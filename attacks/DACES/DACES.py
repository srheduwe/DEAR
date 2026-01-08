import torch
import numpy as np
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch.tools import set_default_logger_config
import logging
import warnings
import os
from attacks.DACES.dim_reduction import downsizer
import time
import torchvision.transforms as transforms
import torchvision
from attacks.DACES.fitness import fitness


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def daces_attack(net,
           image: torch.Tensor = None,
           label: int = None,
           query_budget: int = 10000, 
           time_budget: int = None, 
           seed_inc: int = 0, 
           saving: bool = True, 
           logging_interval: int = 1000,
           index: int = None,
           dataset_name: str = "cifar100",
           model_name: str = "resnet34_cifar100",
           mu: float = None, 
           scaling_factor: int = 14,
           popsize: int = 23, 
           stdev_init: float = 0.0848110498039,
           stdev_max: float = 2.6130781817644,
           c_sigma: float = 0.0082748019543,
           c_sigma_ratio: float = 1.8813217943943,
           c_m: float = 0.5686191705902,
           damp_sigma: float = 1.0082722268148,
           damp_sigma_ratio: float = 0.5801363151528,
           c_c: float = 0.0658367000557,
           c_c_ratio: float = 1.207643245963,
           c_1: float = 0.0009491677108,
           c_1_ratio: float = 2.3526721935377,
           c_mu: float = 0.00415052491,
           c_mu_ratio: float = 2.678917101898,
           e_sigma: float = 0.3970439763507,
           verbose: bool = False,
           hpo: bool = False,
           scaling_method: str = 'nni'
           ):

    start = time.time() 
    
    np.random.seed(seed=42)                          # Seed for images
    torch.manual_seed(42 + seed_inc)                 # Seed for experiment
    torch.cuda.manual_seed(42 + seed_inc)            # Seed for experiment

    if time_budget: query_budget = 10e10
    else: time_budget = 10e10
    
    image = transforms.Compose([transforms.ToTensor()])(image).to(device)
    dim = image.shape[-1]

    if dataset_name == 'ImageNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225] 
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()    
    elif dataset_name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()
    elif dataset_name == 'GTSRB':
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225] 
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
        net = torch.nn.Sequential(normalizer, net).eval()    

    if label != net(image.unsqueeze(0)).argmax(1): 
        print("Image is already misclassified, no need to run DACES attack.")
        return None

    set_default_logger_config(logger_level=logging.WARNING, override=True)

    if saving:
        path = f'results/DACES/{dataset_name}/{scaling_method}/{model_name}/{seed_inc}/'
        os.makedirs(path, exist_ok=True)
    else:
        path = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        queries, norms = np.array([]), np.array([])

        init_queries = 0

        scaling_factor = scaling_factor if dim >= 100 else 1
        _, height, width, upsizer = downsizer(image=image, dim=dim, popsize=popsize, scaling_factor=scaling_factor, scaling_method=scaling_method)

        problem = fitness(
            d=height*width*3,                               # Dimensionality of the problem                
            label=label,                                    # Label of the original image
            I=image,                                        # Original image
            i=index,                                        # index of image
            dim=dim,                                        # Length/Width of original image (for ImageNet=224)  
            height=height,                                  # Height of lower image subspace
            width=width,                                    # Width of lower image subspace
            upsizer=upsizer,                                # upsizer for upscaling image subspace
            net=net,                                        # Neural Network for inference
            exp_update=None,                                # Whether e_sigma is used, defaults to None
            e_sigma=e_sigma,                                # Step size enlarger of dynamic configuration
            device=device,
            start_time=start,
            query_budget=query_budget-init_queries,         # Query budget, most often 30.000
            time_budget=time_budget,                        # Time budget
            abl_c=False,                                    # Whether ablation should be done on the dynamic configuration strategy
            norm=2,                                         # Norm to be used, most often 2 (Euclidian norm)
            path=path,                                      # Path for saving files
            mu=mu, #not used
            verbose=verbose
        )   

        stdev_init = 0.093620629686 if dim >= 100 else 0.0093620629686
        center_init = torch.zeros(size=(height*width*3,))
        
        searcher_args = {
            "problem": problem,
            "stdev_init": stdev_init,
            "stdev_max": stdev_max,
            "center_init": center_init,
            "separable": True, 
            "popsize": popsize,
            "c_sigma": c_sigma,
            "c_sigma_ratio": c_sigma_ratio,
            "c_m": c_m,
            "damp_sigma": damp_sigma,
            "damp_sigma_ratio": damp_sigma_ratio,
            "c_c": c_c,
            "c_c_ratio": c_c_ratio,
            "c_1": torch.tensor(c_1, device=device),
            "c_1_ratio": c_1_ratio,
            "c_mu": torch.tensor(c_mu),
            "c_mu_ratio": c_mu_ratio
        }

        searcher = CMAES(**searcher_args)

        if verbose:
            logger = StdOutLogger(searcher, interval=logging_interval)
        pandas_logger = PandasLogger(searcher)

        searcher.run(10e10) # Arbitrary high number, CMA-ES script has been change so that it stops based on query budget, not generations

        queries = np.concatenate((queries, searcher.problem._query_counter.cpu().numpy()))
        norms = searcher._problem._best_norm.cpu().numpy()

        end = time.time()
        if verbose:
            print(f"Instance {index} took {round(end - start, 3)} seconds")
            print("Final l2 norm: ", norms[-1])
        if path: pandas_logger.to_dataframe().to_csv(path + f"{index}_df.csv")
    
    if hpo:
        area_instances_stepwise_list = [query * norm for query, norm in zip(queries, norms)]
        return np.sum(area_instances_stepwise_list), searcher._problem._best_norm[-1]
    else:
        return norms[-1]