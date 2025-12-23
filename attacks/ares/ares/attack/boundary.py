import torch
from collections import deque
from ares.utils.registry import registry
import time

@registry.register_attack('boundary')
class BoundaryAttack(object):
    ''' Boundary. A black-box decision-based method.
    
    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('boundary')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)
    
    - Supported distance metric: 2.
    - References: https://arxiv.org/abs/1712.04248.
    '''
    def __init__(self, model, device='cuda', norm=2, spherical_step_eps=20, orth_step_factor=0.5,
                 orthogonal_step_eps=1e-2, perp_step_factor=0.5, max_iter=20, target=False):
        '''
        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to 2.
            spherical_step_eps (float): The spherical step epsilon.
            orth_step_factor (float): The orthogonal step factor.
            orthogonal_step_eps (float): The orthogonal step epsilon.
            perp_step_factor (float): The perpendicular step factor.
            max_iter (int): The maximum iteration.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''
        self.net = model
        self.spherical_step = spherical_step_eps
        self.p = norm
        self.target = target
        self.orthogonal_step = orthogonal_step_eps
        self.max_iter = max_iter
        self.min_value = 0
        self.max_value = 1
        self.orth_step_factor = orth_step_factor
        self.perp_step_factor = perp_step_factor
        self.device = device
        self.orth_step_stats = deque(maxlen=30)
        self.perp_step_stats = deque(maxlen=100)
        if self.p !=2:
            raise AssertionError('boundary attck only support L2 bound')
    
    def _is_adversarial(self,x, y, y_target, queries):
        '''The function to test if the input images are adversarial.'''
        output = torch.argmax(self.net(x), dim=1)
        queries.append(len(x))
        if self.target:
            return output == y_target, queries
        else:
            return output != y, queries
    
    def get_init_noise(self, x_target, y, ytarget, queries):
        '''The function to initialize noise.'''
        while True:
            x_init = torch.rand(x_target.size()).to(self.device)
            x_init = torch.clamp(x_init, min=self.min_value, max=self.max_value)
            isadv, queries = self._is_adversarial(x_init, y, ytarget, queries)
            if isadv:
                # print("Success getting init noise",end=' ')
                return x_init, queries


    def perturbation(self, x, x_adv, y, ytarget):
        '''The function of single attack iteration.'''
        unnormalized_source_direction = x - x_adv
        source_norm = torch.norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm
        perturbation = torch.normal(torch.zeros_like(x_adv), torch.ones_like(x_adv)).to(self.device)
        dot = torch.matmul(perturbation, source_direction)
        perturbation -= dot * source_direction
        perturbation *= self.perp_step_factor * source_norm / torch.norm(perturbation)

        D = (1 / torch.sqrt(torch.tensor(self.perp_step_factor ** 2.0 + 1))).to(self.device)
        direction = perturbation - unnormalized_source_direction
        spherical_candidate = torch.clamp(x + D * direction, self.min_value, self.max_value)

        new_source_direction = x - spherical_candidate
        new_source_direction_norm = torch.norm(new_source_direction)
        length = self.orthogonal_step * source_norm
        deviation = new_source_direction_norm - source_norm
        length = max(0, length + deviation) / new_source_direction_norm
        candidate = torch.clamp(spherical_candidate + length * new_source_direction, self.min_value, self.max_value)
        return candidate
   
  
    def boundary(self, x, y, ytarget):
        '''The function of boundary attack.'''
        x_normalised = (x - x.min())/(x.max() - x.min())
        queries = []

        start_time = time.time()
        isadv, queries = self._is_adversarial(x, y, ytarget, queries)
        if isadv:
            # print("The original image is already adversarial")
            return x
        x_adv, queries = self.get_init_noise(x, y, ytarget, queries)
        norms, times = [], []

        for i in range(self.max_iter):
            pertubed = self.perturbation(x, x_adv,y,ytarget)
            isadv, queries = self._is_adversarial(x, y, ytarget, queries)
            if isadv:
                x_adv = pertubed
            if len(self.perp_step_stats) == self.perp_step_stats.maxlen:
                if torch.Tensor(self.perp_step_stats).mean() > 0.5:
                    # print('Boundary too linear, increasing steps')
                    self.spherical_step /= self.perp_step_factor
                    self.orthogonal_step /= self.orth_step_factor
                elif torch.Tensor(self.perp_step_stats).mean() < 0.2:
                    # print('Boundary too non-linear, decreasing steps')
                    self.spherical_step *= self.perp_step_factor
                    self.orthogonal_step *= self.orth_step_factor
                self.perp_step_stats.clear()

            if len(self.orth_step_stats) == self.orth_step_stats.maxlen:
                if torch.Tensor(self.orth_step_stats).mean() > 0.5:
                    # print('Success rate too high, increasing source step')
                    self.orthogonal_step /= self.orth_step_factor
                elif torch.Tensor(self.orth_step_stats).mean() < 0.2:
                    # print('Success rate too low, decreasing source step')
                    self.orthogonal_step *= self.orth_step_factor
                self.orth_step_stats.clear()

            x_adv_normalised = (x_adv - x_adv.min())/(x_adv.max() - x_adv.min())

            # print(x_adv_normalised.min(), x_adv_normalised.max(), x_adv_normalised.shape)
            # print(x_normalised.min(), x_normalised.max(), x_normalised.shape)

            norm = torch.norm(x_normalised-x_adv_normalised)

            norms.append(norm.unsqueeze(0))
            times.append(time.time() - start_time)

        return x_adv, norms, times, queries
        
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
            adv_x, norms, times, queries = self.boundary(images[i].unsqueeze(0), labels[i].unsqueeze(0), target_label)
            adv_images.append(adv_x)
            all_norms.append(torch.cat(norms, 0))
            all_times.append(times)
            all_queries.append(queries)
        adv_images = torch.cat(adv_images, 0)
        return adv_images, all_norms, all_times, all_queries