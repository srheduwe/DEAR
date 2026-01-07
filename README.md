DEAR: Decision-based Ensemble Attacks via Racing
==============================

Welcome to DEAR, an algorithm that combines racing with decision-based attacks to determine the best method for any network/dataset/restriction.

Getting started:
To create an Anaconda environment with the necessary dependencies follow these steps:
```
conda create -n DEAR python=3.9.19
conda activate DEAR
pip install -r requirements.txt
```

To run the attack, specify the experiment, e.g.:
```
python racing.py --dataset_name=cifar100 --model_arc=resnet34_cifar100 --resource=q --budget=1000 --targeted=False --no_seeds=1
```

Resource can be either "q" for queries or "t" for time.
Have at look at loader.py for other networks and datasets.

For running experiments on ImageNet, follow the instructions for obtaining the dataset from the official [website](https://www.image-net.org). Then, place the dataset into data/raw.