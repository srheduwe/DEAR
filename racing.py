import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from scipy.stats import friedmanchisquare, wilcoxon, rankdata, t
from scikit_posthocs import posthoc_conover_friedman, posthoc_nemenyi
import time
import pandas as pd
import sys
import os
import random

# Attacks
from attacks.DACES.DACES import daces_attack
from attacks.CGBA.cgba import cgba_attack
from attacks.HSJA.hsja_wrapper import hsja_attack
from attacks.TA.TA import ta_attack
from attacks.ares.attacking import main as einseins_attack
from attacks.TtBA.main import main as ttba_attack
from loader import loader

# Device and seed
device = "cuda" if torch.cuda.is_available() else "cpu"
split = "test"
samples = int(10)
save_attack_results = False # i.e. images etc. of the single attacks; racing results will always be saved



def frace(dataset_name: str, model_arc: str, resource: str, budget: int, targeted: bool, seeds: int = 1):
    for seed in range(seeds):
        np.random.seed(42+seed)

        if resource == "t":
            time_budget = budget
            query_budget = 10e6
        elif resource == "q":
            query_budget = budget
            time_budget = None
        else:
            raise ValueError("Resource must be 't' or 'q'")

        del sys.argv[1:]  # Removes all user-passed arguments

        os.makedirs("results/", exist_ok=True)

        # Load model and data
        net, dataset, dim = loader(model=model_arc, split=split)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Candidate attacks
        def run_cgba(**kw):
            return cgba_attack(**kw, saving=save_attack_results)

        def run_ta(**kw):
            return ta_attack(**kw, saving=save_attack_results, dim=dim)

        def run_eins(**kw):
            return einseins_attack(**kw, saving=save_attack_results)

        def run_hsja(**kw):
            return hsja_attack(**kw, saving=save_attack_results)

        def run_ttba(**kw):
            return ttba_attack(**kw, saving=save_attack_results)

        def run_daces(**kw):
            return daces_attack(**kw, saving=save_attack_results)

        attacks = {
            "CGBA": run_cgba,
            "HSJA": run_hsja,
            "TtBA": run_ttba
        }

        if targeted == False: # Add untargeted attacks
            attacks.update({
                # "EinsEins": run_eins, #was excluded in paper
                "DACES": run_daces,
                "TA": run_ta,
            })

        cols = attacks.keys()

        # Storage for results
        remaining = list(cols)
        results = pd.DataFrame(columns=cols)
        conover_post_hoc_test_value = pd.DataFrame(columns=cols)
        for name in cols:
            results[f"{name}_rank"] = 0
            
        # Load indices
        indice_path = f'data/correct_indices/{dataset_name}/{model_arc}_{split}.npy'
        indices = np.load(indice_path)
        sampled_indices = np.random.choice(a=indices, size=samples*2, replace=False)

        sampled_indices_targeted = sampled_indices[int(samples):]
        sampled_indices = sampled_indices[:int(samples)]

        # Main F-Race loop
        evaluations = 0

        for i, ii in zip(sampled_indices, sampled_indices_targeted):
            evaluations += 1
            print(f"\nOriginal Instance: {i}")

            image, label = dataloader.dataset[i][0], dataloader.dataset[i][1]

            if targeted == True:
                if dataset_name == 'ImageNet':
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225] 
                    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
                    model = torch.nn.Sequential(normalizer, net).eval()    
                elif dataset_name == 'cifar100':
                    mean = [0.5071, 0.4867, 0.4408]
                    std = [0.2675, 0.2565, 0.2761]
                    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
                    model = torch.nn.Sequential(normalizer, net).eval()
                elif dataset_name == 'GTSRB':
                    mean = [0.0, 0.0, 0.0]
                    std = [1.0, 1.0, 1.0]
                    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
                    model = torch.nn.Sequential(normalizer, net).eval()
                else:
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225] 
                    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
                    model = torch.nn.Sequential(normalizer, net).eval()    

                target_image, target_label = dataloader.dataset[ii][0], dataloader.dataset[ii][1]
                target_image_t = transforms.Compose([transforms.ToTensor()])(target_image).to(device).unsqueeze(0)

                while label == target_label or model(target_image_t).argmax(1) != target_label:
                    ii = random.randint(0, len(dataloader) - 1)
                    target_image, target_label = dataloader.dataset[ii]
                    target_image_t = transforms.Compose([transforms.ToTensor()])(target_image).to(device).unsqueeze(0)
                print(f"\nTarget Instance: {ii}")
            else:
                target_image, target_label = None, None

            for name in remaining:
                print(f"  Running {name} with budget {budget}")
                start_time = time.time()

                score = attacks[name](
                    net=net,
                    image=image,
                    label=label,
                    index=i,
                    dataset_name=dataset_name,
                    query_budget=query_budget,
                    time_budget=time_budget,
                    model_name=model_arc,
                    target_image=target_image,
                    target_label=target_label
                )
                
                elapsed = time.time() - start_time
                print(f"    Score: {score}  Time: {elapsed:.2f}s")
                results.loc[i, name] = score if score is not None else np.inf
            
            results.loc[i, resource] = budget * len(remaining)


            # Elimination step. 
            if all(len(results[n]) > 2 for n in remaining):
                if len(remaining) > 2:
                    _, p = friedmanchisquare(*(results.loc[instance, remaining] for instance, _ in results.iterrows())) #ordering is part of function
                    k = len(results[remaining]) #number of attacked images
                    m = len(remaining) #number of remaining attacks
                    alpha = 0.05

                    sub = k*m*((m+1)**2)/4 # common subtractor in friedman and post-hoc test
                    s_f = k*(m+1)/2 # factor in friedman test

                    rank_cols = [f"{col}_rank" for col in remaining]
                    rank_squared_cols = [f"{col}_rank_squared" for col in remaining]
                    for instance, row in results[remaining].iterrows(): results.loc[row.name, rank_cols] =  rankdata(results.loc[row.name, remaining])
                    
                    results[rank_squared_cols] = results[rank_cols]**2

                    double_sum = sum(sum(results[rank_squared_cols].values)) - sub
                    enumerator = (m-1) * sum((results[rank_cols].sum() - s_f)**2)

                    T = enumerator / double_sum
                    
                    print(f"  Friedman p={p:.4f}")
                    if p < 0.05:
                        f_p = 2*k*(1-(T/(k*(m-1)))) # factor in post-hoc test                
                        d_p = (k-1)*(m-1) # # divisor in post-hoc test   

                        best = results[rank_cols].mean().idxmin()[:-5] # lowest expected rank = lowest mean rank
                        conover_post_hoc_test_value.loc[best, remaining] = np.array(abs(results[f"{best}_rank"].sum()-results[rank_cols].sum())/np.sqrt(double_sum*f_p/d_p))

                        df = (m - 1)*k 

                        significance_threshold_student_t = t.ppf(1-alpha/2, df)

                        losers = [attack for attack in conover_post_hoc_test_value[remaining].columns if conover_post_hoc_test_value.loc[best, attack] > significance_threshold_student_t]

                        for loser in losers:
                            remaining.remove(loser)
                        if len(losers) > 0:
                            print(f"  Eliminated: {losers}")
                        else:
                            print("  No elimination")
                            
                if len(remaining) == 2: 
                    rank_cols = [f"{col}_rank" for col in remaining]

                    for instance, row in results[remaining].iterrows():   
                        results.loc[row.name, rank_cols] =  rankdata(results.loc[row.name, remaining])

                    stat, p = wilcoxon(list(results[remaining[0]].values), list(results[remaining[1]].values))
                    if p < 0.05:
                        loser = results[rank_cols].mean().idxmax()[:-5]
                        remaining.remove(loser)
                
                if len(remaining) == 1:
                    print(f"\nWinner found early: {remaining[0]}")
                    break

                if evaluations >= samples: 
                    best = results[remaining].mean().idxmin()
                    remaining = [best]
                    print(f"\nWinner determined after 10 iterations by rank: {remaining[0]}")
                    break

        print("\nFinal survivors:", remaining)
        results.loc["Winner"] = 0
        results.loc["Winner", remaining[0]] = 1

        results.to_csv(f"results/{dataset_name}_{model_arc}_{resource}_{str(budget)}_{str(targeted)}_{str(seed)}_results.csv")


if __name__ == "__main__":
    frace(dataset_name="cifar100", model_arc="resnet34_cifar100", resource="q", budget=1000, targeted=False, seeds=1)