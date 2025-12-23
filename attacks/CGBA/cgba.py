'''
The provided code demonstrates an attack on the ImageNet dataset targeting four popular classifiers. 
It can be readily converted for attacks against a specific classifier on different datasets.
'''
import torchvision.transforms as transforms
import numpy as np
import torch
import os
from attacks.CGBA.utils import    get_label
from attacks.CGBA.utils import valid_bounds
from torch.autograd import Variable
import time
from attacks.CGBA.proposed_attack import Proposed_attack

    
##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##############################################################################



attack_method = "CGBA_H" # 'CGBA'             # Attacking methods: 'CGBA' or 'CGBA_H'  
dim_reduc_factor = 4               # dim_reduc_factor=1 for full dimensional image space

def cgba_attack(model_name, 
                net, 
                image, 
                label, 
                index, 
                dataset_name, 
                query_budget=10000, 
                time_budget=None,
                attack_method="CGBA_H", 
                verbose=False, 
                iteration=100000,
                saving=True,
                dim_reduc_factor=4,  
                initial_query=30,
                tol=0.0001, 
                sigma=0.0002,
                target_image=None,
                target_label=None,
                ):    
           
    start_time = time.time()
    all_norms = []
    all_queries = []
    image_iter = 0 
    if time_budget: query_budget = 10e10
    else: time_budget = 10e10

    if target_image: 
        targeted = True
    else:
        targeted = False

    t11 = time.time()
    
    delta = 255
    lb, ub = valid_bounds(image, delta)

    if dataset_name == 'ImageNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225] 
    elif dataset_name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif dataset_name == "GTSRB":
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225] 

    net = net.eval()
    
    # Transform data
    im = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                                std = std)
                                ])(image)
    
    lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
    ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)
    
    lb = lb[None, :, :, :].to(device)
    ub = ub[None, :, :, :].to(device)
    
    x_0 = im[None, :, :, :].to(device)
    
    orig_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()


    if targeted:
        # Transform data
        im_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,
                                    std = std)
                                    ])(target_image)
        
        x_0_t = im_t[None, :, :, :].to(device)
        
        orig_label_t = torch.argmax(net.forward(Variable(x_0_t, requires_grad=True)).data).item()
    else:
        x_0_t = None

##############################################################################        
    
                
    if label != int(orig_label) :
        print('Already missclassified ... Lets try another one!')
        return None
        
    else:    
        
    
        image_iter = image_iter + 1
        if verbose:
            print('Image number good to go: ', image_iter)

            print('#################################################################################')
            print(f'Start: {attack_method} non-targeted will be run for {iteration} iterations with dim_reduc_factor: {dim_reduc_factor}')
            print('#################################################################################')
        

        t3 = time.time()
        attack = Proposed_attack(net, x_0, mean, std, lb, ub, 
                                    attack_method = attack_method, 
                                    iteration=iteration, q_limit=query_budget, time_budget=time_budget,
                                    verbose_control=verbose,
                                    tar_img=x_0_t,
                                    dim_reduc_factor=dim_reduc_factor,  
                                    initial_query=initial_query,
                                    tol=tol, 
                                    sigma=sigma,
                                    start_time=start_time
                                    )
        x_adv, n_query, norms, times= attack.Attack()
        t4 = time.time()
        if verbose:
            print(f'##################### End Itetations:  took {t4-t3:.3f} sec #########################')
        
        all_norms.append(norms)
        all_queries.append(n_query)
    
    norm_array = np.array(all_norms)
    query_array = np.array(all_queries)
    norm_median = np.median(norm_array, 0)
    query_median = np.median(query_array, 0)
    
    if saving:
        if targeted:
            path = f'results/CGBA/targeted/{dataset_name}/{model_name}/'
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = f'results/CGBA/untargeted/{dataset_name}/{model_name}/'
            if not os.path.exists(path):
                os.makedirs(path)
                
        np.savez(path + f'{attack_method}_nonTar_imgNum_{index}',
                norm = norm_median,
                query = query_median,
                time = times,
                )
        
        
    
    return norms[-1].item()