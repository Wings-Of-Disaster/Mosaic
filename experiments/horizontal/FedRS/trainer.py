import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join('..', '..')))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from utils.funcs import set_seed
from utils.model import evaluation, CycleDataloader
from models.model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class ClientTrainer:
    def __init__(self, args, client_id):
        self.args = args
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.lr_lm = args.lr_lm
        self.data_set=args.data_set

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.train_label_list = args.data_distributer.get_client_label_list(client_id)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.alpha = 0.1

        self.model = None
        self.optimizer = None

        if self.data_set == 'CIFAR-10':
            self.label_counts = {i: 0 for i in range(10)} 
        elif self.data_set == 'CIFAR-100':
            self.label_counts = {i: 0 for i in range(100)} 
        else:
            raise ValueError(f"Unsupported dataset: {self.data_set}")
            
        self.res = {}
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)

    def restricted_softmax(self, logits, visible_classes):

        scaling_factors = torch.ones_like(logits, device=self.device) * self.alpha
        scaling_factors[:, visible_classes] = 1.0
        scaled_logits = logits * scaling_factors
        return scaled_logits

    def pull_local_model(self, args):

        if self.data_set == 'CIFAR-10':
            self.model = ResNet18(num_classes=10)
        elif self.data_set == 'CIFAR-100':
            self.model = ResNet18(num_classes=100)
        else:
            raise ValueError(f"Unsupported dataset: {self.data_set}")
        
        self.model = self.model.to(self.device)
    
    def clear(self):
        self.res = {}
        self.model = {}
        self.optimizer = {}

        torch.cuda.empty_cache()

    def train_locally_step(self, I, step, global_params):
        
        #self.alpha = min(0.5, self.alpha + 0.01 * step) #CIFAR10
        self.alpha = min(0.8, self.alpha + 0.01 * step) #CIFAR100
        
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        LOSS = 0
        self.model.train()

        for tau in range(I):
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)

            x, y = next(self.train_batch_data_iter)
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            visible_classes = torch.unique(y).tolist()
            local_outputs = self.restricted_softmax(logits, visible_classes)

            loss = self.criterion(local_outputs, y)
            
            loss.backward()

            self.optimizer.step()

            LOSS += loss.item()

        LOSS = LOSS / I
        self.res.update(m_LOSS=LOSS)
        self.res.update(label_counts=self.label_counts)

    def get_eval_info(self, step, train_time=None):
        self.res.update(train_time=train_time)

        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.test_dataloader,
                                    criterion=self.criterion,
                                    model_params=None,
                                    device=self.device)
        self.res.update(test_loss=loss, test_acc=acc, test_sample_size=num)
