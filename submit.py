from __future__ import absolute_import, division, print_function 
import submitit
from racing import frace


saving = True
split = "test"
seeds = 5

dataset_name = "GTSRB"

for model_arc in ["ViT_GTSRB"]: 
    for budget in [500, 1000, 3000, 5000, 10000]:
        for targeted in [True, False]:
            executor = submitit.AutoExecutor(folder=f"submit/frace/{dataset_name}/{model_arc}/{str(budget)}_{targeted}")
            executor.update_parameters(slurm_partition="KathleenG")
            executor.update_parameters(slurm_exclude="kathleengpu01")
            executor.update_parameters(cpus_per_task=14)
            executor.update_parameters(slurm_time='100:00:00')
            executor.update_parameters(name=f"frace")
            executor.update_parameters(slurm_mem='15G')
            executor.update_parameters(slurm_qos='gpu')
            executor.update_parameters(slurm_gres='gpu:1')
            executor.update_parameters(slurm_nodes=1)

            job = executor.submit(frace, dataset_name, model_arc, "q", budget, targeted, seeds=seeds)

    for budget in [3, 6, 9, 12, 15]:
        for targeted in [True, False]:
            executor = submitit.AutoExecutor(folder=f"submit/frace/{dataset_name}/{model_arc}/{str(budget)}_{targeted}")
            executor.update_parameters(slurm_partition="KathleenG")
            executor.update_parameters(slurm_exclude="kathleengpu01")

            executor.update_parameters(cpus_per_task=14)
            executor.update_parameters(slurm_time='100:00:00')
            executor.update_parameters(name=f"frace")
            executor.update_parameters(slurm_mem='15G')
            executor.update_parameters(slurm_qos='gpu')
            executor.update_parameters(slurm_gres='gpu:1')
            executor.update_parameters(slurm_nodes=1)

            job = executor.submit(frace, dataset_name, model_arc, "t", budget, targeted, seeds=seeds)






dataset_name = "caltech101"

for model_arc in ["resnet50_caltech", "ConvNet_caltech", "ViT_caltech"]: 
    for budget in [500, 1000, 3000, 5000, 10000]:
        for targeted in [True, False]:
            executor = submitit.AutoExecutor(folder=f"submit/frace/{dataset_name}/{model_arc}/{str(budget)}_{targeted}")
            executor.update_parameters(slurm_partition="KathleenG")
            executor.update_parameters(slurm_exclude="kathleengpu01")
            executor.update_parameters(cpus_per_task=14)
            executor.update_parameters(slurm_time='100:00:00')
            executor.update_parameters(name=f"frace")
            executor.update_parameters(slurm_mem='15G')
            executor.update_parameters(slurm_qos='gpu')
            executor.update_parameters(slurm_gres='gpu:1')
            executor.update_parameters(slurm_nodes=1)

            job = executor.submit(frace, dataset_name, model_arc, "q", budget, targeted, seeds=seeds)

    for budget in [3, 6, 9, 12, 15]:
        for targeted in [True, False]:
            executor = submitit.AutoExecutor(folder=f"submit/frace/{dataset_name}/{model_arc}/{str(budget)}_{targeted}")
            executor.update_parameters(slurm_partition="KathleenG")
            executor.update_parameters(slurm_exclude="kathleengpu01")
            executor.update_parameters(cpus_per_task=14)
            executor.update_parameters(slurm_time='100:00:00')
            executor.update_parameters(name=f"frace")
            executor.update_parameters(slurm_mem='15G')
            executor.update_parameters(slurm_qos='gpu')
            executor.update_parameters(slurm_gres='gpu:1')
            executor.update_parameters(slurm_nodes=1)

            job = executor.submit(frace, dataset_name, model_arc, "t", budget, targeted, seeds=seeds)





dataset_name = "cifar100"

for model_arc in ["resnet34_cifar100", "resnet50_cifar100", "vgg16_cifar100", "ViT_cifar100"]: 
    for budget in [500, 1000, 3000, 5000, 10000]:
        for targeted in [True, False]:
            executor = submitit.AutoExecutor(folder=f"submit/frace/{dataset_name}/{model_arc}/{str(budget)}_{targeted}")
            executor.update_parameters(slurm_partition="KathleenG")
            executor.update_parameters(slurm_exclude="kathleengpu01")
            executor.update_parameters(cpus_per_task=14)
            executor.update_parameters(slurm_time='100:00:00')
            executor.update_parameters(name=f"frace")
            executor.update_parameters(slurm_mem='15G')
            executor.update_parameters(slurm_qos='gpu')
            executor.update_parameters(slurm_gres='gpu:1')
            executor.update_parameters(slurm_nodes=1)

            job = executor.submit(frace, dataset_name, model_arc, "q", budget, targeted, seeds=seeds)

    for budget in [1, 2, 3, 4, 5]:
        for targeted in [True, False]:
            executor = submitit.AutoExecutor(folder=f"submit/frace/{dataset_name}/{model_arc}/{str(budget)}_{targeted}")
            executor.update_parameters(slurm_partition="KathleenG")
            executor.update_parameters(slurm_exclude="kathleengpu01")
            executor.update_parameters(cpus_per_task=14)
            executor.update_parameters(slurm_time='100:00:00')
            executor.update_parameters(name=f"frace")
            executor.update_parameters(slurm_mem='15G')
            executor.update_parameters(slurm_qos='gpu')
            executor.update_parameters(slurm_gres='gpu:1')
            executor.update_parameters(slurm_nodes=1)

            job = executor.submit(frace, dataset_name, model_arc, "t", budget, targeted, seeds=seeds)



dataset_name = "ImageNet"

for model_arc in ["resnet50", "resnet101", "vgg16", "ViT", "ResNet50_robust", "ResNet101_robust", "vgg16_robust"]: 
    for budget in [500, 1000, 3000, 5000, 10000]:
        for targeted in [True, False]:
            executor = submitit.AutoExecutor(folder=f"submit/frace/{dataset_name}/{model_arc}/{str(budget)}_{targeted}")
            executor.update_parameters(slurm_partition="KathleenG")
            executor.update_parameters(cpus_per_task=14)
            executor.update_parameters(slurm_time='100:00:00')
            executor.update_parameters(name=f"frace")
            executor.update_parameters(slurm_mem='15G')
            executor.update_parameters(slurm_qos='gpu')
            executor.update_parameters(slurm_gres='gpu:1')
            executor.update_parameters(slurm_nodes=1)

            job = executor.submit(frace, dataset_name, model_arc, "q", budget, targeted, seeds=seeds)

    for budget in [3, 6, 9, 12, 15]:
        for targeted in [True, False]:
            executor = submitit.AutoExecutor(folder=f"submit/frace/{dataset_name}/{model_arc}/{str(budget)}_{targeted}")
            executor.update_parameters(slurm_partition="KathleenG")
            executor.update_parameters(cpus_per_task=14)
            executor.update_parameters(slurm_time='100:00:00')
            executor.update_parameters(name=f"frace")
            executor.update_parameters(slurm_mem='15G')
            executor.update_parameters(slurm_qos='gpu')
            executor.update_parameters(slurm_gres='gpu:1')
            executor.update_parameters(slurm_nodes=1)

            job = executor.submit(frace, dataset_name, model_arc, "t", budget, targeted, seeds=seeds)