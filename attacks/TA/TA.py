import json
import torch
import os
import argparse
import time
import attacks.TA.attack_mask as attack
from attacks.TA.attack_utils import get_model, read_imagenet_data_specify, save_results
from foolbox.distances import l2
import numpy as np
from PIL import Image
import torch_dct
from foolbox import PyTorchModel

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="results", help="Output folder")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="The name of model you want to attack(resnet-18, inception-v3, vgg-16, resnet-101, densenet-121)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="images",
        help="The path of dataset"
    )
    parser.add_argument(
         "--csv",
        type=str,
        default="label.csv",
        help="The path of csv information about dataset"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='The random seed you choose'
    )
    parser.add_argument(
        '--max_queries',
        type=int,
        default=30000,
        help='The max number of queries in model'
    )
    parser.add_argument(
        '--ratio_mask',
        type=float,
        default=0.1,
        help='ratio of mask'
    )
    parser.add_argument(
        '--dim_num',
        type=int,
        default=1,
        help='the number of picked dimensions'
    )
    parser.add_argument(
        '--max_iter_num_in_2d',
        type=int,
        default=2,
        help='the maximum iteration number of attack algorithm in 2d subspace'
    )
    parser.add_argument(
        '--init_theta',
        type=int,
        default=2,
        help='the initial angle of a subspace=init_theta*np.pi/32'
    )
    parser.add_argument(
        '--init_alpha',
        type=float,
        default=np.pi/2,
        help='the initial angle of alpha'
    )
    parser.add_argument(
        '--plus_learning_rate',
        type=float,
        default=0.1,
        help='plus learning_rate when success'
    )
    parser.add_argument(
        '--minus_learning_rate',
        type=float,
        default=0.005,
        help='minus learning_rate when fail'
    )
    parser.add_argument(
        '--half_range',
        type=float,
        default=0.1,
        help='half range of alpha from pi/2'
    )
    return parser.parse_args()


def ta_attack(model_name, dataset_name, query_budget, index, image, label, saving, net, dim, verbose=False, target_image=None, target_label=None,
    ratio_mask = 0.1,
    dim_num = 1,
    max_iter_num_in_2d = 2,
    init_theta = 2,
    init_alpha = np.pi/2,
    plus_learning_rate = 0.1,
    minus_learning_rate = 0.005,
    half_range = 0.1,
    time_budget = None
              ):
    
    time_start = time.time()


    if time_budget: query_budget = 10e10
    else: time_budget = 10e10
    
    args = get_args()
    args.side_length = dim
    args.max_queries = query_budget
    args.time_budget = time_budget
    args.dataset_name = dataset_name
    args.model_name = model_name
    args.index = index
    args.ratio_mask = ratio_mask
    args.dim_num = dim_num
    args.max_iter_num_in_2d = max_iter_num_in_2d
    args.init_theta = init_theta
    args.init_alpha = init_alpha
    args.plus_learning_rate = plus_learning_rate
    args.minus_learning_rate = minus_learning_rate
    args.half_range = half_range

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

    preprocessing = dict(mean=mean, std=std, axis=-3)
    fmodel = PyTorchModel(net.eval(), bounds=(0, 1), preprocessing=preprocessing)

    image = image.convert('RGB')
    image = np.asarray(image, dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))

    image = image / 255
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    label = torch.tensor(label).unsqueeze(0).to(device).long()

    if label != fmodel(image).argmax(1):
        print("Image is already misclassified, no need to run attack.")
        return None

    ###############################
    if verbose: print("Attack !")

    ta_model = attack.TA(fmodel, input_device=device)
    my_advs, q_list, my_intermediates, max_length, times = ta_model.attack(args, image, label, verbose=verbose)
    if verbose: 
        print('TA Attack Done')
        print("{:.2f} s to run".format(time.time() - time_start))
        print("Results")

    my_labels_advs = fmodel(my_advs).argmax(1)
    my_advs_l2 = l2(image, my_advs)

    label_o = int(label)
    label_adv = int(my_labels_advs)
    if verbose:
        print("\t- l2 = {}".format(my_advs_l2.item()))
        print("\t- {} queries\n".format(q_list))

    output_folder = model_name + f"/{index}/"
    if saving:
        save_results(args, my_intermediates, len(image), times)
    return my_advs_l2.item()