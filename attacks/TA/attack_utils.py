import torch
import random
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
from foolbox import PyTorchModel
import torchvision.models as models
from datetime import datetime
import pandas as pd
from torch import nn



def get_model(args,device):
    model_name = args.model_name
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.to(device)
            std = std.to(device)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    
    if model_name == 'resnet101':
        model = models.resnet101(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.to(device)
            std = std.to(device)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.to(device)
            std = std.to(device)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    
    if model_name == 'ViT':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.to(device)
            std = std.to(device)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel


    elif model_name == "resnet50_cifar100":
        import timm
        import detectors
        model = timm.create_model("resnet50_cifar100", pretrained=True)
        mean = torch.Tensor([0.5071, 0.4867, 0.4408])
        std = torch.Tensor([0.2675, 0.2565, 0.2761])
        if torch.cuda.is_available():
            model = model.cuda()
            mean = mean.to(device)
            std = std.to(device)
        model.eval()
        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel

    elif model_name == "vgg16_cifar100":
        import timm
        import detectors
        model = timm.create_model("vgg16_bn_cifar100", pretrained=True)
        mean = torch.Tensor([0.5071, 0.4867, 0.4408])
        std = torch.Tensor([0.2675, 0.2565, 0.2761])
        if torch.cuda.is_available():
            model = model.cuda()
            mean = mean.to(device)
            std = std.to(device)
        model.eval()
        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel

    elif model_name == "ViT_cifar100":
        import timm
        import detectors
        model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k",
        pretrained=False)
        model.head = nn.Linear(model.head.in_features, 100)
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100/resolve/main/pytorch_model.bin",
                map_location="cpu",
                file_name="vit_base_patch16_224_in21k_ft_cifar100.pth",
            )
        )
        mean = torch.Tensor([0.5071, 0.4867, 0.4408])
        std = torch.Tensor([0.2675, 0.2565, 0.2761])
        if torch.cuda.is_available():
            model = model.cuda()
            mean = mean.to(device)
            std = std.to(device)
        model.eval()

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel

    elif model_name == "resnet34_cifar100":
        import timm
        import detectors
        model = timm.create_model("resnet34_cifar100", pretrained=True)
        mean = torch.Tensor([0.5071, 0.4867, 0.4408])
        std = torch.Tensor([0.2675, 0.2565, 0.2761])
        if torch.cuda.is_available():
            model = model.cuda()
            mean = mean.to(device)
            std = std.to(device)
        model.eval()

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel

    if model_name == 'resnet-18':
        
        model = models.resnet18(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.to(device)
            std = std.to(device)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'inception-v3':
        model = models.inception_v3(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    
    elif model_name == "ResNet50_robust":
        from robustness.datasets import ImageNet
        from robustness.model_utils import make_and_restore_model

        ds = ImageNet('/storage/work/duwe/imagenet-1k')
        model, _ = make_and_restore_model(arch='resnet50', dataset=ds, pytorch_pretrained=True)
                    #  resume_path="https://download.pytorch.org/models/resnet50-19c8e357.pth")
        if torch.cuda.is_available():
            model = model.cuda()

        model = model.eval()

        fmodel = PyTorchModel(model, bounds=(0, 1))
        return fmodel

    elif model_name == 'vgg-16':
        model = models.vgg16(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'resnet-101':
        model = models.resnet101(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel
    elif model_name == 'densenet-121':
        model = models.densenet121(pretrained=True).eval().to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            mean = mean.cuda(0)
            std = std.cuda(0)

        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
        return fmodel


def get_label(logit):
    _, predict = torch.max(logit, 1)
    return predict



def save_results(args,my_intermediates, n, times):
    path = "results/TA/" + args.dataset_name + "/" + args.model_name
    if not os.path.exists(path):
        os.makedirs(path)
    numpy_results = np.full((n * 3, args.max_queries), np.nan)
    for i, my_intermediate in enumerate(my_intermediates):
        length = len(my_intermediate)
        for j in range(length):
            numpy_results[3 * i][j] = my_intermediate[j][0]
            numpy_results[3 * i + 1][j] = my_intermediate[j][1]
        numpy_results[3 * i + 2][:len(times)] = times
    pandas_results = pd.DataFrame(numpy_results)
    pandas_results.to_csv(os.path.join(path,f'{args.index}_results.csv'))


def read_imagenet_data_specify(args, device):
    images = []
    labels = []
    info = pd.read_csv(args.csv)
    selected_image_paths = []
    for d_i in range(len(info)):
        image_path = info.iloc[d_i]['ImageName']
        image = Image.open(os.path.join(args.dataset_path,image_path))
        image = image.convert('RGB')
        image = image.resize((args.side_length, args.side_length))
        image = np.asarray(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        groundtruth = info.iloc[d_i]['Label']
        images.append(image)
        labels.append(groundtruth)
        selected_image_paths.append(image_path)
    images = np.stack(images)
    labels = np.array(labels)
    images = images / 255
    images = torch.from_numpy(images).to(device)
    labels = torch.from_numpy(labels).to(device).long()
    return images, labels, selected_image_paths


