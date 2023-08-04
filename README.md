# FedKDD: Federated Learning with Diffusion-based Knowledge Distillation in Heterogeneous Networks

![GitHub](https://img.shields.io/github/license/colinlaganier/FedKDD)
![Github](https://img.shields.io/badge/status-under_development-yellow)

Master's Dissertation supervised by Dr. Zhongguo Li

## Abstract

Federated learning (FL) is a distributed machine learning paradigm that enables multiple parties to collaboratively train a shared model without sharing their private data. However, a limitation of FL is the requirement for one singular model to be common to all agents. In this project, we propose a novel FL framework, FedKDD, to allow for training of heterogeneous models. 
First a diffusion model is trained from the local data of the clients. From there the clients are trained on the local data. Then, in the knowledge distilation phase, the clients learn from each other and the server model learns from the aggregated client logits. Our solution proposes a data-free method, leveraging synthetic from the diffusion model to transfer knowledge from the teacher model to the student model. We conduct extensive experiments on two benchmark datasets, i.e., EMNIST, CIFAR-10 and CINIC-10.

## Requirements
Install all the packages from environment.yml file using conda:

```
conda env create -f environment.yaml
```


## Data
* Download train and test datasets using the ```download.sh``` script in the ```dataset/``` directory.
* To use your own dataset: Move your dataset to ```dataset/``` directory and modify ```Dataset``` class accordingly.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment on CINIC-10 dataset with ```ResNet32``` server model and hetergeneous ```ResNet``` clients, run the following command:

```
python main.py --dataset cifar-10 --data --model resnet32 --epochs 10 --gpu 0
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
Training parameters and other options can be set in the ```main.py``` file. The following options are available:

* ```--dataset-id:```  Name of target dataset. Default: 'cifar10'. Options: 'cifar10', 'cinc10', 'emnist'
* ```--data-path:```    Path to the directory containing the dataset.
* ```--data-partition:```   Dataset splitting method. Default: 'iid'. Options: 'iid', 'non-iid'
* ```--server-model:```   Model for server. Default: 'resnet32'. Options: 'resnet32', 'resnet18', 'mobilenetv3', 'shufflenetv2', 'vgg'
* ```--client-model``` Model for server. Default: 'strategy_1'. Options: 'heterogeneous_random', 'homogeneous_random', 'homogenous', 'strategy_1', 'strategy_2' (see ```ClientModelStrategy```)
* ```--epochs:```   Number of rounds of training. Default: 10
* ```--kd-epochs:```   Number of rounds of knowledge distillation. Default: 10
* ```--batch-size:```   Batch size for training. Default: 32
* ```--kd-batch-size:```   Batch size for knowledge distillation. Default: 32
* ```--num-rounds:```   Number of communication rounds. Default: 10
* ```--num-clients:```   Number of clients. Default: 5
* ```--load-diffusion:```   Load diffusion model from file. Default: True
* ```--save-checkpoint:```   Save checkpoint of the model. Default: False

## Further Readings
* [Federated learning by employing knowledge distillation on edge devices with limited hardware resources](https://doi.org/10.1016/j.neucom.2023.02.011)
* [Is Synthetic Data From Diffusion Models Ready for Knowledge Distillation?](https://arxiv.org/abs/2305.12954)


<!-- to do section -->
## Future Work
* Checkpoints, logs and figures will be uploaded soon.
* Implement differential privacy for the diffusion model (see [related work](https://arxiv.org/abs/2302.13861)).