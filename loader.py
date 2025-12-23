import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from torch import nn
import detectors

imagenet_path = ' '

def loader(model, split: str ="test"):
    '''Returns the model, the respective dataset and size of the dimension and channel.'''
    if model == "resnet50":
        import torch
        import torchvision
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)    
        testset = torchvision.datasets.ImageNet(root=imagenet_path, split='val', transform=transform)
    
    elif model == "resnet101":
        import torch
        import torchvision
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)    
        testset = torchvision.datasets.ImageNet(root=imagenet_path, split='val', transform=transform)

    elif model == "vgg16":
        import torch
        import torchvision
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)    
        testset = torchvision.datasets.ImageNet(root=imagenet_path, split='val', transform=transform)

    elif model == "ViT":
        import torch
        import torchvision
        import timm
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)    
        testset = torchvision.datasets.ImageNet(root=imagenet_path, split='val', transform=transform)

    elif model == "ViT_GTSRB":
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained("bazyl/gtsrb-model")
        import torch.nn as nn
        import torch
        import torchvision
        class WrappedModel(nn.Module):
            def __init__(self, hf_model):
                super().__init__()
                self.model = hf_model

            def forward(self, x):
                return self.model(x).logits

        net = WrappedModel(model)

        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.GTSRB(root='data', split='train', transform=transform, download=True)    
        testset = torchvision.datasets.GTSRB(root='data', split='test', transform=transform, download=True)

    elif model == "InceptionV3":
        import torch
        import torchvision
        net = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 299, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)    
        testset = torchvision.datasets.ImageNet(root=imagenet_path, split='val', transform=transform)

    elif model == "ResNet50_robust":
        from robustness.datasets import ImageNet
        from robustness.model_utils import make_and_restore_model
        from torch import nn
        import torch
        import torchvision

        ds = ImageNet(imagenet_path)
        net, _ = make_and_restore_model(arch='resnet50', dataset=ds, pytorch_pretrained=True)
                    #  resume_path="https://download.pytorch.org/models/resnet50-19c8e357.pth")
        
        net.normalizer = nn.Identity()
        
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 299, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)    
        testset = torchvision.datasets.ImageNet(root=imagenet_path, split='val', transform=transform)

    elif model == "ResNet101_robust":
        from robustness.datasets import ImageNet
        from robustness.model_utils import make_and_restore_model
        from torch import nn
        import torch
        import torchvision

        ds = ImageNet(imagenet_path)
        net, _ = make_and_restore_model(arch='resnet101', dataset=ds, pytorch_pretrained=True)
                    #  resume_path="https://download.pytorch.org/models/resnet50-19c8e357.pth")
        
        net.normalizer = nn.Identity()
        
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 299, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)    
        testset = torchvision.datasets.ImageNet(root=imagenet_path, split='val', transform=transform)

    elif model == "vgg16_robust":
        from robustness.datasets import ImageNet
        from robustness.model_utils import make_and_restore_model
        from torch import nn
        import torch
        import torchvision

        ds = ImageNet(imagenet_path)
        net, _ = make_and_restore_model(arch='vgg16', dataset=ds, pytorch_pretrained=True)
                    #  resume_path="https://download.pytorch.org/models/resnet50-19c8e357.pth")
        
        net.normalizer = nn.Identity()
        
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 299, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.ImageNet(root=imagenet_path, split='train', transform=transform)    
        testset = torchvision.datasets.ImageNet(root=imagenet_path, split='val', transform=transform)

    elif model == "resnet50_cifar100":
        import timm
        import torch
        import torchvision
        net = timm.create_model("resnet50_cifar100", pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 32, 3
        trainset = torchvision.datasets.CIFAR100(root="data/raw", train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root="data/raw", train=False, download=True)
        
    elif model == "vgg16_cifar100":
        import timm
        import torch
        import torchvision
        net = timm.create_model("vgg16_bn_cifar100", pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 32, 3
        trainset = torchvision.datasets.CIFAR100(root="data/raw", train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root="data/raw", train=False, download=True)
        
    elif model == "ViT_cifar100":
        from torch import nn 
        import timm
        import torch
        import torchvision
        net = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k",
        pretrained=False)
        net.head = nn.Linear(net.head.in_features, 100)
        net.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
                map_location="cpu",
                file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
            )
        )

        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim])])
        trainset = torchvision.datasets.CIFAR100(root="data/raw", train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR100(root="data/raw", train=False, transform=transform, download=True)

    elif model == "resnet34_cifar100":
        import timm
        import torch
        import torchvision
        net = timm.create_model("resnet34_cifar100", pretrained=True)

        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 32, 3
        trainset = torchvision.datasets.CIFAR100(root="data/raw", train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root="data/raw", train=False, download=True)

    elif model == "resnet50_caltech":
        import timm
        import torchvision
        from torch import nn
        import torch

        net = timm.create_model("hf_hub:anonauthors/caltech101-timm-resnet50", pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim]), transforms.Lambda(lambda img: img.convert("RGB"))])
        testset = torchvision.datasets.Caltech101(root='data', transform=transform)    
        trainset = None

        class LogitShifter(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model

            def forward(self, x):
                logits = self.base(x)             # [B, C]
                return logits[:, 1:]              # [B, C-1]
        
        net = LogitShifter(net)

    elif model == "ViT_caltech":
        import timm
        import torchvision
        from torch import nn
        import torch

        net = timm.create_model("hf_hub:anonauthors/caltech101-timm-vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim]), transforms.Lambda(lambda img: img.convert("RGB"))])
        testset = torchvision.datasets.Caltech101(root='data', transform=transform)    
        trainset = None

        class LogitShifter(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model

            def forward(self, x):
                logits = self.base(x)             # [B, C]
                return logits[:, 1:]              # [B, C-1]
        
        net = LogitShifter(net)

    elif model == "ConvNet_caltech":
        import timm
        import torchvision
        from torch import nn
        import torch

        net = timm.create_model("hf_hub:anonauthors/caltech101-timm-convnext_base.fb_in1k", pretrained=True)
        if torch.cuda.is_available():
            net = net.cuda()

        dim, channel = 224, 3
        transform = transforms.Compose([transforms.Resize([dim, dim]), transforms.Lambda(lambda img: img.convert("RGB"))])
        testset = torchvision.datasets.Caltech101(root='data', transform=transform)    
        trainset = None

        class LogitShifter(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model

            def forward(self, x):
                logits = self.base(x)             # [B, C]
                return logits[:, 1:]              # [B, C-1]
        
        net = LogitShifter(net)

    if split == "test":
        return net, testset, dim
    else:
        return net, trainset, dim