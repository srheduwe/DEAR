import torch 
from torch import linalg as LA
from evotorch import Problem, SolutionBatch
import time
import numpy as np

class fitness(Problem):
    def __init__(self, 
                 d: int,                    # Dimensionality of the problem
                 label: int,                # Label of the original image
                 I: torch.Tensor,           # Original image
                 i: int,                    # Index of image
                 dim: int,                  # Length/Width of original image (for ImageNet=224)  
                 height: int,               # Height of lower image subspace
                 width: int,                # Width of lower image subspace
                 upsizer: list,             # upsizer for upscaling image subspace
                 net,                       # Neural Network for inference
                 exp_update,                # Whether e_sigma is used, defaults to None
                 e_sigma: float,            # Step size enlarger of dynamic configuration
                 device: bool,              
                 start_time: float,
                 query_budget: int,         # Query budget, most often 30.000
                 time_budget: int,          # Time budget
                 abl_c: bool,               # Whether ablation should be done on the dynamic configuration strategy
                 norm,                      # Norm to be used, most often 2 (euclidian norm)
                 path: str,
                 verbose: bool,
                 mu: float
                 ):
        super().__init__(
            objective_sense="min",
            solution_length=d,
            bounds=(torch.zeros(height*width*3), torch.ones(height*width*3)),
            device=device,
            )

        self._label = label
        self._I = I
        self._i = i                 
        self._dim = dim
        self._height = height
        self._width = width
        self._upsizer = upsizer           
        self._net = net
        self._exp_update = exp_update
        self._e_sigma = e_sigma
        self._device = device
        self._query_counter = torch.tensor([], device=device)
        self._time = torch.tensor([], device=device)
        self._label_pun = 1000
        self._C_ben = False                                             # indicates, whether any instance has been classified as benign before
        self._start_time = start_time
        self._query_budget = query_budget
        self._time_budget = time_budget
        self._abl_c = abl_c
        self._norm = norm
        self._path = path
        self._verbose = verbose
        self._mu = mu
        self._best_Instance = None
        self._best_pred = None
        self._best_norm = torch.tensor([1000], device=device)

    @torch.inference_mode()
    def _evaluate_batch(self, solution: SolutionBatch):
        perturbation = solution.values                                  # The current population=perturbation
        popsize = perturbation.shape[0]
        upsizer = self._upsizer
        height, width = self._height, self._width
        pun = self._label_pun
        dim = self._dim
        
        perturbation = (perturbation.clone().reshape(popsize, 3, self._height, self._width)).float() 
        perturbed_image = self._I.unsqueeze(0).repeat(popsize, 1, 1, 1).float() # create tensor with popsize times the reference image

        if isinstance(upsizer, list):                                   # This is the case when using subspace activation and grid downsizing
            upsizer[0] = torch.arange(popsize).reshape(popsize, 1, 1, 1)# Update the popsize in the upsizer (only necessary if popsize is not given before)
            perturbed_image[upsizer] += perturbation                    # Add perturbation to original image
        elif upsizer:                                                   # For nearest neighbour and bilinear interpolation
            perturbation = upsizer(perturbation)                        # Scale perturbation to original image size
            perturbed_image += perturbation                             # Add perturbation to original image
            height = self._I.shape[1]
            width = self._I.shape[2]
        else: 
            perturbed_image += perturbation                             # Only for small images like CIFAR, when we search in the orginal image space

        perturbed_image = torch.clamp(perturbed_image, min=0.0, max=1.0)
        norms = LA.vector_norm(perturbation.reshape(popsize, 3*height*width), ord=float(self._norm), axis=1)
        f = norms.detach().clone()
        queries = torch.zeros(1, device=self._device)

        if not self._C_ben:                                             # Stage 1 of query strategy
            queried_instances = norms == norms.min()                    # We only query the nearest image
            no_of_queried_instances = queried_instances.sum()           # Can not be simply put to 1 in case two instances have the same distance
            
            preds = self._net(perturbed_image[queried_instances].reshape(no_of_queried_instances, 3, dim, dim)).argmax(1)
            label_puns = torch.where(preds == self._label, pun, 0)            
            queries += no_of_queried_instances.unsqueeze(0)

            if label_puns.sum() == pun*no_of_queried_instances:         # True, as soon as nearest image is misclassified
                self._C_ben = True
            else:
                f[queried_instances] += label_puns                      # We only add the label puns, when we do not jump to stage 2 afterwards

            index = f.argmin()                            
            best_instance = perturbed_image[index]
            best_norm = f[index]
            best_pred = preds
              
        if self._C_ben:                                                 # Stage 2 of query strategy
            mu_eff = self.mu_eff/popsize                                # Share of effekcive population
            norm_threshold = torch.quantile(norms, mu_eff)
            queried_instances = norms < norm_threshold                  # We first only query mu_eff
            remaining_instances = norms >= norm_threshold
            no_of_queried_instances = queried_instances.sum()
            no_of_remaining_instances = remaining_instances.sum()

            preds = self._net(perturbed_image[queried_instances].reshape(no_of_queried_instances, 3, dim, dim)).argmax(1)
            label_puns = torch.where(preds == self._label, pun, 0)
            f[queried_instances] += label_puns
            self._adv_share = (torch.mean(label_puns/pun))

            if self._adv_share >= 0.75:                                 # Stage 3 of query strategy
                preds_rest = self._net(perturbed_image[remaining_instances].reshape(no_of_remaining_instances, 3, dim, dim)).argmax(1)    
                preds_total = torch.empty(popsize, device=self._device).to(torch.long)
                preds_total[queried_instances] = preds
                preds_total[remaining_instances] = preds_rest
                
                label_puns_rest = torch.where(preds_rest == self._label, pun, 0)        
                queries += (no_of_queried_instances.unsqueeze(0)+no_of_remaining_instances.unsqueeze(0))
                f[remaining_instances] += label_puns_rest
                self._adv_share = (self._adv_share*mu_eff) + (torch.mean(label_puns_rest/pun)*(1-mu_eff))
                index = f.argmin()                            
                best_instance = perturbed_image[index]
                best_norm = f[index]
                best_pred = preds_total[index]

            else:
                queries += no_of_queried_instances.unsqueeze(0)
                queried_images = perturbed_image[queried_instances]
                index = f[queried_instances].argmin()                            
                best_instance = queried_images[index]
                best_norm = f[queried_instances][index]
                best_pred = preds[index]


        self._query_counter = torch.cat((self._query_counter, queries))
        self._time = torch.cat((self._time, torch.tensor(time.time()-self._start_time, device=self._device).unsqueeze(0)))

        if (best_norm < torch.min(self._best_norm)):  # Only update if nearer and adversarial
            self._best_norm = torch.cat((self._best_norm, best_norm.unsqueeze(0)))
            self._best_Instance = best_instance
            self._best_pred = best_pred
        else:
            self._best_norm = torch.cat((self._best_norm, self._best_norm[-1].unsqueeze(0)))

        if not self._abl_c:                                             
            if self._C_ben and self._adv_share >= 0.95:                 # When the share of adversarial examples is larger than t_adv = 0.95
                self._exp_update = self._e_sigma                        # Manually increase the exponential update for sigma
            else:
                self._exp_update = None

        if (self._query_counter.sum() > (self._query_budget - popsize)) and self._path:
            np.save(self._path + f"{self._i}_queries.npy", self._query_counter.cpu().numpy())
            np.save(self._path + f"{self._i}_times.npy", self._time.cpu().numpy())
            np.save(self._path + f"{self._i}_norms.npy", self._best_norm.cpu().numpy())
            torch.save(self._best_Instance, self._path + f"{self._i}_best_Instance.pt")

            if self._verbose: print(f"Label changed from {self._label} to {best_pred}")

        solution.set_evals(f)   