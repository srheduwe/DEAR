from __future__ import absolute_import, division, print_function 

from attacks.HSJA.hsja import hsja
import numpy as np
import os
import imageio
import torch
import torchvision
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"


def hsja_attack(index, 
                net, 
                image, 
                label, 
                target_image=None, 
                target_label=None, 
                dataset_name = "ImageNet", 
                model_name = "ResNet50_robust", 
                saving=True,
                constraint = "l2", 
                attack_type = "untargeted", 
                verbose=False, 
                gamma=1.0, 
                max_num_evals=1e4, 
                init_num_evals=100,
                num_iterations = 200000, 
                stepsize_search = "geometric_progression", 
                query_budget=10000,
                time_budget = None):

    if time_budget: query_budget = 10e10
    else: time_budget = 10e10

    if target_image:
        attack_type = "targeted"
        data_model = "targeted/" + dataset_name + "/" + model_name 
    else:
        data_model = "untargeted/" + dataset_name + "/" + model_name 
    
    x_test = transforms.Compose([transforms.ToTensor()])(image).to(device)
    
    if attack_type == "targeted":
        target_image = transforms.Compose([transforms.ToTensor()])(target_image).to(device)

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
    
    if label != model(x_test.unsqueeze(0)).argmax(1): #change for targeted attack
        print("Image is already misclassified, no need to run attack.")
        return None
    if verbose:
        print(f'Attacking the {index}th sample...')
        print(f"Target: {target_label}")

    perturbed, final_norm = hsja(model, 
                     x_test, 
                     clip_max=1.0, 
                     clip_min=0.0, 
                     constraint=constraint, 
                     num_iterations=num_iterations, 
                     target_label=target_label, 
                     target_image=target_image, 
                     stepsize_search=stepsize_search, 
                     folder=data_model,
                     instance=index,
                     query_budget=query_budget,
                     time_budget=time_budget,
                     verbose=verbose,
                     gamma=gamma, 
                     max_num_evals=max_num_evals,
                     init_num_evals=init_num_evals,
                     saving=saving
                     )

    perturbed = np.transpose(perturbed.cpu().numpy(), (1, 2, 0))
    perturbed = (perturbed * 255).astype(np.uint8)

    if saving:
        os.makedirs("results/HSJA/" + f'{data_model}/figs', exist_ok=True)
        imageio.imwrite("results/HSJA/" + f'{data_model}/figs/{index}.jpg', perturbed)
        np.save("results/HSJA/" + f'{data_model}/figs/{index}.npy', perturbed)

    return final_norm