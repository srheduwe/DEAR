#!/bin/bash

#SBATCH --partition=KathleenG
#SBATCH --qos=gpu
#SBATCH -J gpu_serial
#SBATCH -o gpu_serial.%J.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Ask for up to 15 Minutes of runtime

# Name the job
#SBATCH --job-name=TtBA
#SBATCH --time=240:00:00
#SBATCH --cpus-per-task=14


# Declare a file where the STDOUT/STDERR outputs will be written
#SBATCH --output=outputs/output.%J.txt

### end of Slurm SBATCH definitions

### your program goes here (hostname is an example, can be any program)
# `srun` runs `ntasks` instances of your programm `hostname`

source /storage/work/duwe/miniconda3/etc/profile.d/conda.sh
conda activate AE
set CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="$PYTHONPATH:$PWD"

# srun python main.py --dataset=imagenet-resnet50 --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=200 --beginIMG=0 --budget=30000 --remember=1 --model_arc=resnet50
# srun python main.py --dataset=imagenet-resnet101 --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=200 --beginIMG=0 --budget=30000 --remember=1 --model_arc=resnet101
# srun python main.py --dataset=imagenet-vgg16 --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=200 --beginIMG=0 --budget=30000 --remember=1 --model_arc=vgg16
# srun python main.py --dataset=imagenet-ViT --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=200 --beginIMG=0 --budget=30000 --remember=1 --model_arc=ViT
# srun python main.py --dataset=imagenet-ResNet50_robust --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=200 --beginIMG=0 --budget=30000 --remember=1 --model_arc=ResNet50_robust
# srun python main.py --dataset=imagenet-ResNet101_robust --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=200 --beginIMG=0 --budget=30000 --remember=1 --model_arc=ResNet101_robust
# srun python main.py --dataset=imagenet-vgg16_robust --targeted=0 --norm=TtBA --epsilon=1.0 --early=0 --imgnum=200 --beginIMG=0 --budget=30000 --remember=1 --model_arc=vgg16_robust
