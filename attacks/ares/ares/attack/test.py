import torch
import numpy as np
from attacks.ares.ares.utils.registry import registry
import time
device = "cuda" if torch.cuda.is_available() else "cpu"


@registry.register_attack('evolutionary')
class Evolutionary(object):
    ''' Evolutionary. A black-box decision-based method.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('evolutionary')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 2.
    - References: https://arxiv.org/abs/1904.04433.
    '''
    def __init__(self, model, device='cuda', ccov=0.001, decay_weight=0.99,
                 max_queries=10000, mu=0.01, sigma=3e-2, maxlen=30, target=False):
        '''The function to initialize evolutionary attack.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            ccov (float): The parameter cconv. Defaults to 0.001.
            decay_weight (float): The decay weight param. Defaults to 0.99.
            max_queries (int): The maximum query number. Defaults to 10000.
            mu (float): The mean for bias. Defaults to 0.01.
            sigma (float): The deviation for bias. Defaults to 3e-2.
            maxlen (int): The maximum length. Defaults to 30.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''
        self.model = model
        self.ccov = ccov
        self.decay_weight = torch.tensor(decay_weight, device=device)
        self.max_queries = max_queries
        self.mu = mu
        self.device =device
        self.sigma = sigma
        self.maxlen = maxlen
        self.targeted = target
        self.min_value = 0
        self.max_value = 1
        if self.targeted:
            raise AssertionError('dont support targeted attack')


    def _is_adversarial(self,x, y, ytarget):
        '''The function to judge if the input image is adversarial.'''
        output = torch.argmax(self.model(x), dim=1)
        if self.targeted:
            return output == ytarget
        else:
            return output != y

    def get_init_noise(self, x_target, y, ytarget):
        '''The function to initialize noise.'''
        while True:
            x_init = torch.rand(x_target.size()).to(self.device)
            x_init = torch.clamp(x_init, min=self.min_value, max=self.max_value)
            if self._is_adversarial(x_init, y, ytarget):
                # print("Success getting init noise",end=' ')
                return x_init

   
    def evolutionary(self, x, y, ytarget):
        '''The function to conduct evolutionary attack.'''
        x_normalised = (x - x.min())/(x.max() - x.min())

        start_time = time.time()
        x = x.to(self.device)
        y = y.to(self.device)
        if ytarget is not None:
            ytarget = ytarget.to(self.device)
        pert_shape = (x.size(0),x.size(1),x.size(2),x.size(3)) 
        m = torch.prod(torch.tensor(pert_shape)).item()
        k = int(m / 20)
        evolutionary_path = torch.zeros(pert_shape, device=device)
        decay_weight = self.decay_weight
        diagonal_covariance = torch.ones(pert_shape, device=device)
        ccov = self.ccov
        if self._is_adversarial(x, y,ytarget):
            return x

        # find an starting point
        x_adv = self.get_init_noise(x , y, ytarget)
            
        norms, times, queries = [], [], []
        mindist = 1e10
        stats_adversarial = torch.tensor([], device=device)
        for _ in range(self.max_queries):
            unnormalized_source_direction = x - x_adv
            source_norm = torch.norm(unnormalized_source_direction)
            if mindist > source_norm:
                mindist = source_norm
                best_adv = x_adv
            
            # import pdb; pdb.set_trace()
            selection_prob = diagonal_covariance.reshape(-1) / torch.sum(diagonal_covariance)
            # selection_indices = np.random.choice(m, k, replace=False, p=selection_prob)
            
            # import pdb; pdb.set_trace()
            selection_indices = torch.arange(m, device=device)[torch.multinomial(selection_prob, num_samples=k, replacement=False)] #[selection_prob.multinomial(num_samples=k, replacement=False)]

            # pert = np.random.normal(0.0, 1.0, pert_shape)
            pert = torch.normal(mean=0.0, std=1.0, size=pert_shape, device=device)

            factor = torch.zeros([m], device=device)
            factor[selection_indices] = True
            # import pdb; pdb.set_trace()
            pert *= factor.reshape(pert_shape) * torch.sqrt(diagonal_covariance)
            pert_large = torch.Tensor(pert).to(self.device)

            biased = (x_adv + self.mu * unnormalized_source_direction).to(self.device)
            candidate = biased + self.sigma * source_norm * pert_large / torch.norm(pert_large)
            candidate = x - (x - candidate) / torch.norm(x - candidate) * torch.norm(x - biased)
            candidate = torch.clamp(candidate, self.min_value, self.max_value)
            
            if self._is_adversarial(candidate, y, ytarget):
                x_adv = candidate
                evolutionary_path = decay_weight * evolutionary_path + torch.sqrt(torch.tensor(1)-decay_weight** 2) * pert
                diagonal_covariance = (1 - ccov) * diagonal_covariance + ccov * (evolutionary_path ** 2)
                torch.cat((stats_adversarial, torch.tensor([1], device=device)))
            else:
                torch.cat((stats_adversarial, torch.tensor([0], device=device)))
            if len(stats_adversarial) == self.maxlen:
                self.mu *= torch.exp(torch.mean(stats_adversarial) - 0.2)
                stats_adversarial = torch.tensor([], device=device)


            x_adv_normalised = (best_adv - best_adv.min())/(best_adv.max() - best_adv.min())

            norm = torch.norm(x_normalised-x_adv_normalised)

            norms.append(norm.unsqueeze(0))
            times.append(time.time() - start_time)
            queries.append(1)

        return best_adv, norms, times, queries


    def __call__(self, images=None, labels=None, target_labels=None):
        '''This function perform attack on target images with corresponding labels 
        and target labels for target attack.

        Args:
            images (torch.Tensor): The images to be attacked. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].
            labels (torch.Tensor): The corresponding labels of the images. The labels should be torch.Tensor with shape [N, ]
            target_labels (torch.Tensor): The target labels for target attack. The labels should be torch.Tensor with shape [N, ]

        Returns:
            torch.Tensor: Adversarial images with value range [0,1].

        '''
        adv_images, all_norms, all_times, all_queries = [], [], [], []
        for i in range(len(images)):
            if target_labels is None:
                target_label = None
            else:
                target_label = target_labels[i].unsqueeze(0)
            adv_x, norms, times, queries = self.evolutionary(images[i].unsqueeze(0), labels[i].unsqueeze(0), target_label)
            adv_images.append(adv_x)
            all_norms.append(torch.cat(norms, 0))
            all_times.append(times)
            all_queries.append(queries)
        adv_images = torch.cat(adv_images, 0)
        return adv_images, all_norms, all_times, all_queries