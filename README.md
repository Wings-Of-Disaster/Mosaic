<h1 align="center">Mosaic: Data-Free Knowledge Distillation via Mixture-of-Experts for Heterogeneous Distributed Environments</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2505.19699">
        <img alt="Build" src="http://img.shields.io/badge/cs.CV-arXiv%3A2505.19699-B31B1B.svg">
    </a>
    <a href="https://github.com/Wings-Of-Disaster/Mosaic">
        <img alt="Build" src="https://img.shields.io/badge/Github-Code-blue">
    </a>
</p>

## Overview
This repository contains the code for **Mosaic**, a novel DFKD framework utilizing Mixture-of-Experts with Advanced Integrated Collaboration for Heterogeneous Distributed Environments.

## Requirements

- Python == 3.10
- PyTorch == 2.1.2
- mpi4py == 4.0.1
- setproctitle == 1.3.4
- pandas == 2.2.3
- scikit-learn == 1.6.0
- scipy == 1.15.0
- matplotlib == 3.9.0

You can use the following command to install.

```bash
python3 -m pip install \
  Python==3.10 \
  torch==2.1.2 \
  mpi4py==4.0.1 \
  setproctitle==1.3.4 \
  pandas==2.2.3 \
  scikit-learn==1.6.0 \
  scipy==1.15.0 \
  matplotlib==3.9.0
```

Please note that the installation of mpi4py may require additional dependencies. If you encounter issues during the installation, you can resolve them by running the following commands:

```bash
sudo apt update
sudo apt install libmpich-dev
sudo apt-get install hwloc libhwloc-dev
sudo apt-get install libfabric-dev
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

Of course, you can also refer to our `environment.yml` file to set up the environment beforehand:

```bash
conda env create -f environment.yml
```

## Run Pipeline for Mosaic

### Entering the Mosaic Directory

```bash
cd Mosaic/experiments/horizontal
cd Mosaic
```

### Set Python Hash Seed
The Hash Seed needs to be set to 0.

```bash
export PYTHONHASHSEED=0
```

### Run Initialization Script
We initially used the FedAvg algorithm to train a global model and an ensemble, which were the initial conditions for our use of the Mosaic method. For the CIFAR-10 dataset, we trained for 300 epochs and set I=10, but for CIFAR-100, we trained for 1000 epochs and set I=20 to converge.

```bash
python main_init.py --data_set CIFAR-10 --comm_round 300 --I 10 --batch_size 64 --eval_step_interval 5 --eval_batch_size 256 --lr_lm 0.01 --weight_decay 1e-4 --data_partition_mode non_iid_dirichlet_unbalanced --non_iid_alpha 0.01 --client_num 10 --selected_client_num 10 --device cuda --seed 0 --app_name FedInit --imageSize 32 --outf GANs_CIFAR-10_0.01 --niter 200
```

### Run Mosaic Script
The Mosaic method first preheats for several epochs to reduce model instability, and then divides it into training meta head stage and knowledge distillation stage. After the knowledge distillation is completed, the global model is issued and the learning rate is reduced for fine-tuning.

```bash
python main_mosaic.py --data_set CIFAR-10 --comm_round 80 --I 10 --batch_size 64 --eval_step_interval 5 --eval_batch_size 256 --lr_lm 0.01 --weight_decay 1e-4 --data_partition_mode non_iid_dirichlet_unbalanced --non_iid_alpha 0.01 --client_num 10 --selected_client_num 10 --device cuda --seed 0 --app_name Mosaic --imageSize 32 --outf GANs_CIFAR-10_0.01 --niter 200 --teacher_init --meta_epochs 30 --KD_epochs 1500 --warmup_epochs 40
```

### Combined Label Counts Table for Seed 657 with non-iid Alpha = 0.01
This is an example of Dirichlet's division into 0.01 on the CIFAR-10 dataset, which is very extreme in the case of 0.01.

|          |   Label 0 |   Label 1 |   Label 2 |   Label 3 |   Label 4 |   Label 5 |   Label 6 |   Label 7 |   Label 8 |   Label 9 |
|:---------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| Client 0 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |        72 |         0 |         0 |
| Client 1 |         0 |         0 |       689 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |
| Client 2 |         0 |         0 |         0 |         0 |         0 |         0 |        90 |         1 |         0 |       974 |
| Client 3 |      4999 |         0 |         0 |      5000 |         0 |         0 |         0 |         0 |         0 |         0 |
| Client 4 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |       782 |         0 |
| Client 5 |         0 |        37 |         0 |         0 |         0 |         0 |         0 |      4927 |         0 |       478 |
| Client 6 |         0 |         0 |         0 |         0 |      4999 |        49 |         0 |         0 |         0 |         0 |
| Client 7 |         0 |         0 |         0 |         0 |         0 |      4950 |      4910 |         0 |         0 |         0 |
| Client 8 |         0 |      4963 |      4310 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |
| Client 9 |         1 |         0 |         1 |         0 |         1 |         1 |         0 |         0 |      4218 |      3548 |

It is worth noting that due to the wide variety of CIFAR-100 datasets, the difference in data distribution between non iid Alpha=0.01 and non iid Alpha=1.0 is relatively small. Therefore, the impact on the performance of the model is relatively minor. 

### Comparative Experiments

Our comparative experiments are based on the following works:

1. **DFRD (DENSE, FedFTG, DFRD, and FedAvg)** from "Data-Free Robustness Distillation for Heterogeneous Federated Learning." The code is available at [DFRD Code Repository](https://anonymous.4open.science/r/DFRD-0C83). Our four methods are all based on this code, and experiments can refer to them.

2. **FedBN** from the repository [med-air/FedBN](https://github.com/med-air/FedBN).

3. **FedProx and FedRS** from [lxcnju/FedRepo](https://github.com/lxcnju/FedRepo).

### Experiment Setup

For fair comparison, we used the same data partitioning strategy and model structure across all experiments.

- **Non-Knowledge Distillation Methods**: We additionally included four methods in the repository: FedAvg, FedProx, FedBN, and FedRS. For all methods, we unified the configuration as follows:
  - CIFAR-10: I=10, training for 300 epochs.
  - CIFAR-100: I=20, training for 1000 epochs.

- **Knowledge Distillation Methods**: For methods like DENSE, FedFTG, and DFRD, in addition to setting I=10/20 for fair comparison, we strictly followed the original code’s experimental setup. This was necessary to ensure the learning curves would converge properly.

This consistent setup ensures that performance differences reflect the algorithmic strengths and limitations of each method rather than variations in data handling or model configuration.

## Citation
If you find this repository useful, please consider giving a star ⭐ and citation.
```
@article{Liu_2025_Mosaic,
  title={Mosaic: Data-Free Knowledge Distillation via Mixture-of-Experts for Heterogeneous Distributed Environments},
  author={Liu, Junming and Gao, Yanting and Meng, Siyuan and Sun, Yifei and Wu, Aoqi and Jin, Yufei and Chen, Yirong and Wang, Ding and Zeng, Guosun},
  journal={arXiv preprint arXiv:2505.19699},
  year={2025}
}
```