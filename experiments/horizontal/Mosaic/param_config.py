import argparse
import logging

import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..','..')))

import torch
from experiments.datasets.data_distributer import DataDistributer
from utils.funcs import consistent_hash, set_seed


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--comm_round', type=int, default=100)

    parser.add_argument('--I', type=int, default=20, help='synchronization interval')

    parser.add_argument('--batch_size', type=int, default=256)  
    
    parser.add_argument('--eval_step_interval', type=int, default=5)
   
    parser.add_argument('--eval_batch_size', type=int, default=256)

    parser.add_argument('--lr_lm', type=float, default=0.01)

    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--model_type', type=str, default='ResNet_18', choices=['ResNet_18', 'ResNet_20', 'ResNet_34', 'ResNet_50'])

    parser.add_argument('--data_set', type=str, default='CIFAR-10',
                        choices=['CIFAR-10', 'CIFAR-100'])

    parser.add_argument('--data_partition_mode', type=str, default='non_iid_dirichlet_unbalanced',
                        choices=['iid', 'non_iid_dirichlet_unbalanced', 'non_iid_dirichlet_balanced'])

    parser.add_argument('--non_iid_alpha', type=float, default=0.01) 

    parser.add_argument('--client_num', type=int, default=10)

    parser.add_argument('--selected_client_num', type=int, default=10)

    parser.add_argument('--device', type=torch.device, default='cuda')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=str, default='FedInversion')

    parser.add_argument('--teacher_init', action='store_true', help='Use Teacher Model Parameters')
    
    #30 epochs for cifar10 0.01, 50 epochs for cifar10 0.1 and 1.0
    parser.add_argument('--meta_epochs', type=int, default=30, help='number of meta training epochs')

    #1500 epochs for cifar10 0.01, 2500 epochs for cifar10 0.1 and 1.0
    parser.add_argument('--KD_epochs', type=int, default=1500, help='number of knowledge distillation training epochs')

    # CIFAR-10: 10 clusters, CIFAR-100: 60 clusters
    parser.add_argument('--cluster_k', type=int, default=10, help='number of clusters')

    parser.add_argument('--warmup_epochs', type=int, default=40, help='number of warmup epochs for initial training')

    parser.add_argument('--iters_mi', default=2000, type=int, help='number of iterations for model inversion')

    parser.add_argument('--cig_scale', default=0.0, type=float, help='competition score')

    parser.add_argument('--di_lr', default=0.1, type=float, help='lr for deep inversion')

    parser.add_argument('--di_var_scale', default=2.5e-5, type=float, help='TV L2 regularization coefficient')

    parser.add_argument('--di_l2_scale', default=3e-8, type=float, help='L2 regularization coefficient')

    parser.add_argument('--r_feature_weight', default=1e2, type=float, help='weight for BN regularization statistic')

    parser.add_argument('--exp_descr', default="try1", type=str, help='name to be added to experiment name')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')

    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--ndf', type=int, default=64)

    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')

    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--netG', default='', help="path to netG (to continue training)")

    parser.add_argument('--netD', default='', help="path to netD (to continue training)")

    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

    parser.add_argument('--label', type=int, help='generated label')

    args = parser.parse_args()

    super_params = args.__dict__.copy()
    del super_params['log_level']
    super_params['device'] = super_params['device'].type
    ff = f"{args.app_name}-{consistent_hash(super_params, code_len=64)}.pkl"
    ff = f"{os.path.dirname(__file__)}/Result/{ff}"
    if os.path.exists(ff):
        print(f"output file existed, skip task")
        exit(0)

    args.data_distributer = _get_data_distributer(args)

    return args

def _get_data_distributer(args):
    set_seed(args.seed + 5363)
    return DataDistributer(args)

