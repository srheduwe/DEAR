# coding:utf-8
from attacks.TtBA import DataTools
from attacks.TtBA import TtBA
import os
from datetime import datetime
import argparse
import torch
import random
from attacks.TtBA import dataset
import numpy as np
import torchvision
import time
import torchvision.transforms as transforms
from attacks.TtBA.general_torch_model import GeneralTorchModel
device = "cuda" if torch.cuda.is_available() else "cpu"


bridge_bestKs = []
def main(net, image, label, index, query_budget=30000, time_budget=None, dataset_name="ImageNet", model_name="resnet50", verbose=False, target_image=None, target_label=None, saving=True):
    start_time = time.time()

    torch_model, test_loader = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--dataset', default='mnist', type=str, help='Dataset')
    parser.add_argument('--targeted', default=0, type=int, help='targeted-1 or untargeted-0')
    parser.add_argument('--norm', default='k', type=str, help='Norm for attack, k or l2')
    parser.add_argument('--epsilon', default=0.3, type=float, help='attack strength')
    parser.add_argument('--budget', default=10000, type=int, help='Maximum query for the attack norm k')
    parser.add_argument('--early', default=0, type=int, help='early stopping (stop attack once the adversarial example is found)')
    parser.add_argument('--remember', default=0, type=int, help='if remember adversarial examples.')
    parser.add_argument('--imgnum', default=1, type=int, help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int, help='begin test img number')
    parser.add_argument('--RGB', default=['RGB'], type=str, help='List of RGB channels (e.g., RG)')
    parser.add_argument('--binaryM', default=1, type=int, help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--initDir', default=1, type=int, help='initial direction, 1,-1,and 0 for random, 2 for 1-11-1...')
    parser.add_argument('--model_arc', default="resnet50", type=str, help='model_arc')

    args = parser.parse_args()
    args.budget = int(query_budget)
    
    if time_budget: 
        query_budget = int(10e6)
        args.time_budget = time_budget
    else: args.time_budget = 10e10
    
    args.dataset = dataset_name
    if target_image: args.targeted = 1
    args.norm = "TtBA"
    if verbose: print(args)
    label = torch.tensor(label, device=device)
    image = transforms.Compose([transforms.ToTensor()])(image).to(device).unsqueeze(0)
    if args.targeted == 1: target_image = transforms.Compose([transforms.ToTensor()])(target_image).to(device)


    Folder_name =  str(args.dataset) + f"/{model_name}/" +\
                        "target" + str(args.targeted) +\
                        "_budget" + str(args.budget) +\
                        "_IMG" + str(index)
    

    # ###############################################################################
    Out = dataset.OutResult(args, Folder_name, saving=saving)

    Attacker = None


    if dataset_name == 'ImageNet':
        model = torch.nn.DataParallel(net, device_ids=[0])
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif dataset_name == 'cifar100':
        model = torch.nn.DataParallel(net, device_ids=[0])
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.5071, 0.4867, 0.4408],
                                        im_std=[0.2675, 0.2565, 0.2761])
    elif dataset_name == 'GTSRB':
        model = torch.nn.DataParallel(net, device_ids=[0])
        torch_model = GeneralTorchModel(model, n_class=43, im_mean=[0.0, 0.0, 0.0],
                                        im_std=[1.0, 1.0, 1.0])
    else:
        model = torch.nn.DataParallel(net, device_ids=[0])
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])

    Out.ImgNum_total_tested = Out.ImgNum_total_tested + 1
    out_label = torch_model.predict_label(image.cuda()).cpu().item()
    real_label = label.item()
    if out_label == real_label:
        if args.targeted == 1:
            target_image = target_image.cuda()
        else:
            target_image, target_label = None, None

        if args.norm == "TtBA":
            Attacker = TtBA.Attacker(args, torch_model, image, index, args.norm, target_image, target_label, verbose=verbose, start_time=start_time)
            _, times = Attacker.attack()
        else:
            print("norm is wrong: " + args.norm)
            return

        if saving: 
            if not os.path.exists("results/TtBA/" + Folder_name):
                os.makedirs("results/TtBA/" + Folder_name)
            np.save("results/TtBA/" + Folder_name + "/time.npy", times)
            
        Out.add1Result(Attacker)
        Out.Summary()
        if args.remember == 1:
            combined_file = DataTools.save_images([Attacker.Img_result, Attacker.heatmaps], "results/TtBA/" + Folder_name, Attacker.File_string)
        if verbose: print("")
    else:
        print(f"IMG{index} Originally classify incorrect")
    Out.Summary()
    if verbose and not time_budget: 
        print(args)
        print(f"NATURAL ACCURACY RATE={Out.NATURAL_ACCURACY_RATE:.4f}")
        print(f"ATTACK SUCCESS RATE = {Out.ATTACK_SUCCESS_RATE:.4f}")
        print(f"ROBUST ACCURACY RATE={Out.ROBUST_ACCURACY_RATE:.4f}")
        print(f"AVG(MID)-AccQuery  = {Out.AccQuery_avg:.1f}({Out.AccQuery_mid:.1f})")
        print(f"midAUC-l2(linf) = {Out.AUC_l2:.1f}({Out.AUC_linf:.1f})")
        query = [int(args.budget/5), int(args.budget/2), int(args.budget)]
        for q in query:
            print(f"AVG(MID)-l2 after {q} queries : {torch.mean(Out.L2_LINE_sum[:, q]).item():.3f}"
                f"({torch.median(Out.L2_LINE_sum[:, q]).item():.3f})")
        """"""
        print(f"AVG(MID)-ADB = {Out.EndADB_avg}({Out.EndADB_mid})")
        print(f"AVG(MID)-linf= {Out.Endlinf_avg}()")
    elif verbose and time_budget:
        print("final l2:", Out.L2_LINE_sum[:,-1].item())

    return Out.L2_LINE_sum[:,-1].item()


if __name__ == "__main__":
    main()