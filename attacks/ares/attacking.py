import os
from attacks.ares.ares.utils.registry import registry
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import torchvision.transforms as transforms
import torchvision
import time

def main(dataset_name="ImageNet", 
        model_name="resnet50", 
        constraint="l2", 
        attack_type="untargeted", 
        num_samples=1, 
        image=None, 
        label=None, 
        target_label=None, 
        target_image=None, 
        net=None, 
        query_budget=10000, 
        time_budget=None,
        index=None,
        ccov=0.001,
        decay_weight=0.99,
        mu=0.01,
        sigma=3e-2,
        maxlen=30,
        saving=True
        ):

    start_time = time.time()
    if time_budget: query_budget = int(10e10)
    else: time_budget = 10e10
    
    x_test = transforms.Compose([transforms.ToTensor()])(image).to(device)
    if dataset_name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif dataset_name == "GTSRB":
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225] 

    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    model = torch.nn.Sequential(normalizer, net).eval()
    
    attack_name = "evolutionary"
    attacker_cls = registry.get_attack(attack_name)
    
    path = "results/11/" + "/" + dataset_name + "/" + model_name + "/"
    os.makedirs(path, exist_ok=True)

    attacker = attacker_cls(model, max_queries=query_budget, device=device,
                                                time_budget=time_budget,
                                                ccov=ccov,
                                                decay_weight=decay_weight,
                                                mu=mu,
                                                sigma=sigma,
                                                maxlen=maxlen,
                                                start_time=start_time
                            )

    adv_images, norms, times, queries = attacker(x_test.unsqueeze(0), torch.tensor(label).unsqueeze(0))
    if adv_images is None: return None
    
    if saving:
        np.save(path + f"{index}_times.npy", times)
        np.save(path + f"{index}_norms.npy", norms[0].cpu())
        np.save(path + f"{index}_queries.npy", queries)
        torch.save(adv_images, path + f"{index}_images.pt")
        
    return norms[0][-1].item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', "ImageNet", 'cifar100'], default='ImageNet') 
    parser.add_argument('--model_name', type=str, choices=['resnet', 'resnet50', 'resnet101', "ViT", "vgg16", "resnet34_cifar100", "resnet50_cifar100", "vgg16_cifar100", "ViT_cifar100", "ResNet50_robust"], default='resnet50') 
    parser.add_argument('--constraint', type=str, choices=['l2', 'linf'], default='l2') 
    parser.add_argument('--attack_type', type=str, choices=['targeted', 'untargeted'], default='untargeted') 
    parser.add_argument('--num_samples', type=int, default=200) 
    args = parser.parse_args()

    print(args.dataset_name)
    main(args)
