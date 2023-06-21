# FedKDD: Federated Learning with Diffusion-based Knowledge Distillation in Heterogeneous Networks

![GitHub](https://img.shields.io/github/license/colinlaganier/FedKDD)
![Github](https://img.shields.io/badge/status-under_development-yellow)

Master's Dissertation supervised by Dr. Zhongguo Li

## Abstract

Federated learning (FL) is a distributed machine learning paradigm that enables multiple parties to collaboratively train a shared model without sharing their private data. However, the performance of FL is limited by the heterogeneity of the data distribution across clients. In this paper, we propose a novel FL framework, FedKDD, to address the heterogeneity problem in FL. FedKDD consists of two phases: a knowledge distillation phase and a federated learning phase. In the knowledge distillation phase, a teacher model is trained on the data of all clients. Then, the teacher model is used to guide the training of the student model in the federated learning phase. In particular, we propose a diffusion-based knowledge distillation method to transfer knowledge from the teacher model to the student model. We conduct extensive experiments on three benchmark datasets, i.e., CIFAR-10, CIFAR-100 and 

## Requirments
Install all the packages from environment.yml file using conda:

```
conda env create -f environment.yml
```


## Data
* Download train and test datasets using the ```download.sh``` script in the ```dataset/``` directory.
* To use your own dataset: Move your dataset to ```dataset/``` directory and modify ```Dataset``` class accordingly.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment on CIFAR-10 dataset with ```ResNet32``` server model and hetergeneous ```ResNet``` clients, run the following command:

```
python main.py --dataset cifar-10 --data --model resnet32 --epochs 10 --gpu 0
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset-id:```  Default: 'cifar10'. Options: 'cifar10', 'cifar100'
* ```--data-path:```    Path to the directory containing the dataset.
* ```--server-model:```    Default: 'resnet32'. Options: 'resnet18', 'resnet32', 'mobilenet3', 'lenet5', 'vgg'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
<!-- * ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1. 
parser.add_argument("--dataset-id", type=str, choices=["cifar10", "cifar100"], default="cifar10")
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--server-model", type=str, choices=list(Models.available.keys()), default="resnet32")
    parser.add_argument("--client-model", type=str, choices=list(Models.available.keys()) + list(ClientModelStrategy.available.keys()), default="strategy_1")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--load-diffusion", type=bool, default=False)
    print(list(Models.available.keys())) -->

<!-- #### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits. -->



## Further Readings
### Papers:
* [Federated learning by employing knowledge distillation on edge devices with limited hardware resources](https://doi.org/10.1016/j.neucom.2023.02.011)
* [Is Synthetic Data From Diffusion Models Ready for Knowledge Distillation?](https://arxiv.org/abs/2305.12954)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

<!-- The proposed implementation is a data-free method building upon: Tanghatari, Ehsan & Kamal, Mehdi & Afzali-Kusha, Ali & Pedram, Massoud. (2023). Federated Learning by Employing Knowledge Distillation on Edge Devices with Limited Hardware Resources. Neurocomputing. 531. 10.1016/j.neucom.2023.02.011. -->