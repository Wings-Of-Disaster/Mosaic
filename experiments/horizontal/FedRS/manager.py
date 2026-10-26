import logging
import time
import glob
import collections

import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
#sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))

import numpy as np
import pandas as pd
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from experiments.models.model import evaluation, get_cpu_param
from experiments.models.model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from experiments.utils.base import BaseServer, ModelStateAvgAgg, NumericAvgAgg
from experiments.utils.funcs import (consistent_hash, save_pkl, model_size, set_seed)

from trainer import ClientTrainer


class ServerManager(BaseServer):
    def __init__(self, ct, args):

        super().__init__(ct)
        self.args = args
        self.super_params = args.__dict__.copy()

        self.app_name = args.app_name
        self.device = args.device
        self.client_num = args.client_num                   
        self.selected_client_num = args.selected_client_num 
        self.comm_round = args.comm_round                   
        self.I = args.I                                
        self.eval_step_interval = args.eval_step_interval
        self.data_set = args.data_set

        self.full_train_dataloader = args.data_distributer.get_train_dataloader()  
        self.full_test_dataloader = args.data_distributer.get_test_dataloader()    
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if self.data_set == 'CIFAR-10':
            self.num_classes = 10
        elif self.data_set == 'CIFAR-100':
            self.num_classes = 100
        else:
            raise ValueError(f"Unsupported dataset: {self.data_set}")

        self.model =  self.initialize_model(self.data_set)

        self.local_sample_numbers = [len(args.data_distributer.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]

        self.global_params_aggregator = ModelStateAvgAgg()

        self.client_label_counts = {i: {j: 0 for j in range(self.num_classes)} for i in range(args.client_num)}

        self.client_test_acc_aggregator = NumericAvgAgg()

        self.comm_load = {client_id: 0 for client_id in range(args.client_num)}

        self.client_eval_info = [] 
        self.global_train_eval_info = [] 

        self.unfinished_client_num = -1

        self.step = -1
        
        self.global_params = get_cpu_param(self.model.state_dict())

        set_seed(args.seed + 657)
    
    def initialize_model(self, dataset_name):
 
        if dataset_name == 'CIFAR-10':
            model = ResNet18(num_classes=10)
        elif dataset_name == 'CIFAR-100':
            model = ResNet18(num_classes=100)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        return model
    
    def save_global_params(self, file_name=None):
        
        file_name=f"global_model_params_{self.args.non_iid_alpha}.pth"
        file_path = os.path.join(os.getcwd(), file_name)
        torch.save(self.model.state_dict(), file_path)
        print(f"Global model parameters saved to {file_path}")
    
    def start(self):
        logging.info("start...")

        self.next_step()

    def end(self):
        logging.info("end...")

        self.super_params['device'] = self.super_params['device'].type

        dataset_name = self.super_params.get('data_set', 'dataset')

        ff = f"{self.app_name}_{dataset_name}_{consistent_hash(self.super_params, code_len=64)}_fedavg.pkl"
        logging.info(f"output to {ff}")

        result = {'super_params': self.super_params,
                  'global_train_eval_info': pd.DataFrame(self.global_train_eval_info),
                  'client_eval_info': pd.DataFrame(self.client_eval_info),
                  'comm_load': self.comm_load}
        save_pkl(result, f"{os.path.dirname(__file__)}/Result/{ff}")
        #self.save_global_params()

        self._ct_.shutdown_cluster()

    def end_condition(self):
        return self.step > self.comm_round - 1

    def next_step(self):
        self.step += 1

        self.selected_clients = self._new_train_workload_arrage()
        self.unfinished_client_num = self.selected_client_num
        self.global_params_aggregator.clear()
        self.model = self.model.to('cuda')

        for client_id in self.selected_clients:
            self._ct_.get_node('client', client_id) \
                .fed_client_train_step(step=self.step, global_params=self.global_params)

    def _new_train_workload_arrage(self):

        self.selected_client_num = np.random.randint(self.client_num // 2, self.client_num + 1)

        if self.selected_client_num < self.client_num:
            selected_client = np.random.choice(range(self.client_num), self.selected_client_num, replace=False)
        elif self.selected_client_num == self.client_num:
            selected_client = np.array([i for i in range(self.client_num)])
        return selected_client


    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     client_model_params,
                                     eval_info):
        assert self.step == step
        self.client_eval_info.append(eval_info)

        client_label_counts = eval_info.get('label_counts', {i: 0 for i in range(self.num_classes)})

        weight = self.local_sample_numbers[client_id]

        self.client_test_acc_aggregator.put(eval_info['test_acc'], weight)

        self.client_label_counts[client_id] = client_label_counts
        self.global_params_aggregator.put(client_model_params, weight)

        if self.comm_load[client_id] == 0:
            self.comm_load[client_id] = model_size(client_model_params) / 1024 / 1024 

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.server_train_test_res = {'comm. round': self.step, 'client_id': 'server'}

            self.global_params = self.global_params_aggregator.get_and_clear()

            client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
            print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))

            self.model.load_state_dict(self.global_params, strict=True)

            self.server_train_test_res['client_test_acc_avg'] = client_test_acc_avg
            self._set_global_train_eval_info()

            self.global_train_eval_info.append(self.server_train_test_res)
            self.server_train_test_res = {}

            self.next_step()


    def _set_global_train_eval_info(self):
        
        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.full_test_dataloader,
                                    criterion=self.criterion,
                                    device=self.device,
                                    eval_full_data=True)
        torch.cuda.empty_cache()
        self.server_train_test_res.update(test_loss=loss, test_acc=acc, test_sample_size=num)

        logging.info(f"global eval info:{self.server_train_test_res}")


class ClientManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.I = args.I
        self.device = args.device
        self.client_id = self._ct_.role_index
        self.model_type = args.model_type
        self.data_set = args.data_set

        if self.data_set == 'CIFAR-10':
            self.num_classes = 10
        elif self.data_set == 'CIFAR-100':
            self.num_classes = 100
        else:
            raise ValueError(f"Unsupported dataset: {self.data_set}")

        self.args = args.__dict__.copy()
        del self.args['data_distributer']

        self.trainer = ClientTrainer(args, self.client_id)
        self.step = 0

        self.client_train_test_res = {'comm. round': self.step, 'client_id': self.client_id}

        self.model_params = None
        self.full_test_dataloader = args.data_distributer.get_test_dataloader() 
        self.criterion = nn.CrossEntropyLoss().to(self.device)  

        self.bn_running_mean = {}
        self.bn_running_var = {}
    
    def start(self):
        logging.info("start...")

    def end(self):
        logging.info("end...")
        
    def end_condition(self):
        return False

    def fed_client_train_step(self, step, global_params):
        self.step = step

        self.trainer.res = {'communication round': step, 'client_id': self.client_id}

        self.trainer.pull_local_model(self.args) 

        self.trainer.model.load_state_dict(global_params, strict=True)

        self.timestamp = time.time()

        self.trainer.train_locally_step(self.I, step, global_params)
        curr_timestamp = time.time()
        train_time = curr_timestamp - self.timestamp

        self.model_params = get_cpu_param(self.trainer.model.state_dict())

        self.finish_train_step(self.model_params, train_time)
        self.trainer.clear()
        torch.cuda.empty_cache()

    def finish_train_step(self, model_params, train_time):
        self.trainer.get_eval_info(self.step, train_time)
        logging.debug(f"finish_train_step comm. round:{self.step}, client_id:{self.client_id}")

        self._ct_.get_node("server") \
            .set(deepcopy=False) \
            .fed_finish_client_train_step(self.step,
                                          self.client_id,
                                          model_params,
                                          self.trainer.res)