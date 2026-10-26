import logging
import time
import glob
import collections

import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

import numpy as np
import pandas as pd
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.model import evaluation, get_cpu_param
from models.model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils.dcgan_model import Generator, Discriminator
from utils.base import BaseServer, ModelStateAvgAgg, NumericAvgAgg
from utils.funcs import (consistent_hash, save_pkl, model_size, set_seed)
from utils.deepinversion_cifar import DeepInversionFeatureHook, get_images

from trainer_mosaic import ClientTrainer

import torch
import torch.nn as nn

class CustomAttentionEnsemble_For_CIFAR10(nn.Module):
    def __init__(self, models, device, num_classes):
        super(CustomAttentionEnsemble_For_CIFAR10, self).__init__()
        self.models = models
        self.device = device
        self.num_classes = num_classes

        for i in range(num_classes):
            self.models[i] = self.models[i].to(self.device)
        for model in self.models.values():
            model.eval()

        self.fc_expand = nn.Linear(num_classes, num_classes * num_classes)

        self.attention_weights = nn.Parameter(torch.ones(num_classes * num_classes))
        self.diag_weights = nn.Parameter(torch.ones(num_classes) * 2.0)

        self.transformer = nn.Transformer(
            d_model=num_classes * num_classes, 
            nhead=num_classes, 
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.2
        ).to(self.device)

        self.fc = nn.Linear(num_classes * num_classes, num_classes)

    def forward(self, images):
        batch_size = images.size(0)
        all_outputs = torch.zeros((batch_size, self.num_classes * self.num_classes), device=self.device)

        with torch.no_grad():
            for i in range(self.num_classes):
                if images.dim() == 2:
                    outputs = self.models[i].linear(images)
                elif images.dim() == 4:
                    outputs = self.models[i](images)
                all_outputs[:, i*self.num_classes:(i+1)*self.num_classes] = outputs

        #To enhance mapping capability and expand model dimensions
        all_outputs = self.fc_expand(all_outputs.view(batch_size, self.num_classes, self.num_classes))
        all_outputs = all_outputs.permute(1, 0, 2)

        transformer_outputs = self.transformer(all_outputs, all_outputs)
        #(Batchsize, Number of models, Number of labels) -> (Number of models, Batchsize, Number of labels)

        final_outputs = torch.zeros((batch_size, self.num_classes * self.num_classes), device=self.device)
        for i in range(self.num_classes):
            #The first dimension represents the i-th model
            final_outputs += transformer_outputs[i, :, :]

            #The i-th model, the i-th column class, and all samples in the entire batch are multiplied by weights
            #diag_attention = transformer_outputs[i, :, :] * self.attention_weights
            #diag_attention[:, i] *= self.diag_weights[i]
            #final_outputs += diag_attention

        #final_outputs /= self.num_classes
        final_outputs = self.fc(final_outputs)

        return final_outputs

class CustomAttentionEnsemble_For_CIFAR100(nn.Module):
    
    # By default, the 10×100 layer configuration shows better performance when non_iid_alpha = 0.01
    def __init__(self, models, device, num_classes):
        super(CustomAttentionEnsemble_For_CIFAR100, self).__init__()
        self.models = models
        self.device = device
        self.num_classes = num_classes

        for i in range(num_classes):
            self.models[i] = self.models[i].to(self.device)
        for model in self.models.values():
            model.eval()

        self.fc_expand = nn.Linear(num_classes, 10 * num_classes)

        self.attention_weights = nn.Parameter(torch.ones(num_classes))
        self.diag_weights = nn.Parameter(torch.ones(num_classes) * 2.0)

        self.transformer = nn.Transformer(
            d_model=10 * num_classes, 
            nhead=num_classes, 
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.2
        ).to(self.device)

        self.fc = nn.Linear(10 * num_classes, num_classes)

    def forward(self, images):
        batch_size = images.size(0)
        all_outputs = torch.zeros((batch_size, self.num_classes * self.num_classes), device=self.device)

        with torch.no_grad():
            for i in range(self.num_classes):
                if images.dim() == 2:
                    outputs = self.models[i].linear(images)
                elif images.dim() == 4:
                    outputs = self.models[i](images)
                all_outputs[:, i*self.num_classes:(i+1)*self.num_classes] = outputs

        all_outputs = self.fc_expand(all_outputs.view(batch_size, self.num_classes, self.num_classes))
        all_outputs = all_outputs.permute(1, 0, 2)

        transformer_outputs = self.transformer(all_outputs, all_outputs)

        final_outputs = torch.zeros((batch_size, 10 * self.num_classes), device=self.device)
        for i in range(self.num_classes):
            final_outputs += transformer_outputs[i, :, :]
        
        final_outputs /= self.num_classes
        final_outputs = self.fc(final_outputs)

        return final_outputs

class MetaTransformerTrainer:
    def __init__(self, models, device, criterion, num_classes, dataset, ema_decay=0.99):
        self.models = models
        self.device = device
        self.criterion = criterion
        self.num_classes = num_classes

        if dataset == 'CIFAR-10':
            self.meta_transformer = CustomAttentionEnsemble_For_CIFAR10(models, device, num_classes).to(self.device)
            self.ema_transformer = CustomAttentionEnsemble_For_CIFAR10(models, device, num_classes).to(self.device)
        elif dataset == 'CIFAR-100':
            self.meta_transformer = CustomAttentionEnsemble_For_CIFAR100(models, device, num_classes).to(self.device)
            self.ema_transformer = CustomAttentionEnsemble_For_CIFAR100(models, device, num_classes).to(self.device)

        self.optimizer = torch.optim.Adam(self.meta_transformer.parameters(), lr=0.001)
        self.ema_decay = ema_decay
        self._initialize_ema()

    def _initialize_ema(self):
        for ema_param, param in zip(self.ema_transformer.parameters(), self.meta_transformer.parameters()):
            ema_param.data.copy_(param.data)

    def _update_ema(self):
        for ema_param, param in zip(self.ema_transformer.parameters(), self.meta_transformer.parameters()):
            ema_param.data = self.ema_decay * ema_param.data + (1.0 - self.ema_decay) * param.data

    def eval_meta_transformer_ema_nograd(self, images):
        self.ema_transformer.eval()
        with torch.no_grad():
            final_outputs = self.ema_transformer(images)
        return final_outputs
    
    def eval_meta_transformer_grad(self, images):
        self.meta_transformer.eval()
        final_outputs = self.meta_transformer(images)
        return final_outputs

    def train_transformer(self, train_loader, num_epochs=5, idx=0):
        self.meta_transformer.train()
        
        for epoch in range(num_epochs):
            num_epochs_sum = num_epochs
            running_loss = 0.0
            
            for batch_idx, data in enumerate(train_loader):
                if batch_idx != idx:
                    continue
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.meta_transformer(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                self._update_ema()

                num_epochs_sum -= 1
                if num_epochs_sum == 0:
                    break

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}")

        print("Transformer training completed with EMA.")

def jensen_shannon_divergence(p, q):

    m = 0.5 * (p + q)
    jsd = 0.5 * (F.kl_div(F.log_softmax(p, dim=1), F.softmax(m, dim=1), reduction='batchmean') + 
                 F.kl_div(F.log_softmax(q, dim=1), F.softmax(m, dim=1), reduction='batchmean'))
    return jsd

def knowledge_distillation(student_logits, teacher_logits, targets, temperature=1.0, alpha=0.8, beta=1.0):

    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(student_logits, targets)
    jsd_loss = jensen_shannon_divergence(student_logits / temperature, teacher_logits / temperature)

    return alpha * soft_loss + (1. - alpha) * hard_loss + beta * jsd_loss

def modify_dataloader_batch_size(train_loader, new_batch_size):

    dataset = train_loader.dataset
    sampler = train_loader.sampler
    drop_last = train_loader.drop_last
    num_workers = train_loader.num_workers
    pin_memory = train_loader.pin_memory

    new_train_loader = DataLoader(
        dataset=dataset,
        batch_size=new_batch_size,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return new_train_loader

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

        self.prototype={}

        if self.data_set == 'CIFAR-10':
            self.num_classes = 10
        elif self.data_set == 'CIFAR-100':
            self.num_classes = 100
        else:
            raise ValueError(f"Unsupported dataset: {self.data_set}")

        self.model =  self.initialize_model(self.data_set)
        self.models = {i: self.initialize_model(self.data_set) for i in range(self.num_classes)}
        self.meta_trainer = None

        self.ensemble_global_params = {i: get_cpu_param(self.models[i].state_dict()) for i in range(self.num_classes)}

        self.local_sample_numbers = [len(args.data_distributer.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]

        self.global_params_aggregator = ModelStateAvgAgg()
        self.ensemble_global_params_aggregator = {i: ModelStateAvgAgg() for i in range(self.num_classes)}

        self.client_label_counts = {i: {j: 0 for j in range(self.num_classes)} for i in range(args.client_num)}

        self.client_test_acc_aggregator = NumericAvgAgg()

        self.comm_load = {client_id: 0 for client_id in range(args.client_num)}

        self.client_eval_info = [] 
        self.global_train_eval_info = [] 

        self.unfinished_client_num = -1

        self.step = -1

        self.teacher_init = args.teacher_init
        
        if self.teacher_init:
            self.model_params_path = f'./global_model_params_{self.data_set}_{self.args.non_iid_alpha}.pth'
            model_params = torch.load(self.model_params_path)
            self.model.load_state_dict(model_params, strict=True)
            print("Model parameters loaded successfully")
            loss, acc, num = evaluation(model=self.model,
                                        dataloader=self.full_test_dataloader,
                                        criterion=self.criterion,
                                        device=self.device,
                                        eval_full_data=True)
            print(acc)

            for i, model in self.models.items():
                weight_path = f'./model_group_params_{self.data_set}_{self.args.non_iid_alpha}/model_{i}_params.pth'
                checkpoint = torch.load(weight_path)
                model.load_state_dict(checkpoint)
            print("Model Groups parameters loaded successfully")
            
            for i in range(self.num_classes):
                self.models[i] = self.models[i].to('cuda')
            model_group_accuracy = self.evaluate_model_group()
            print(f'model_group_accuracy: {model_group_accuracy:.2f}%')
        
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
        
        file_name=f"global_model_params_mosaic_{self.data_set}_{self.args.non_iid_alpha}.pth"
        file_path = os.path.join(os.getcwd(), file_name)
        torch.save(self.model.state_dict(), file_path)
        print(f"Global model parameters saved to {file_path}")

    def save_model_group_params(self, dir_name=None):

        dir_name=f"model_group_params_mosaic_{self.data_set}_{self.args.non_iid_alpha}"
        dir_path = os.path.join(os.getcwd(), dir_name)
        os.makedirs(dir_path, exist_ok=True)

        for model_idx, model in self.models.items():
            model_file_path = os.path.join(dir_path, f"model_{model_idx}_params.pth")
            torch.save(model.state_dict(), model_file_path)
            print(f"Model {model_idx} parameters saved to {model_file_path}")

        print(f"All model parameters saved to {dir_path}")
    
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
        #self.save_model_group_params()

        self._ct_.shutdown_cluster()

    def end_condition(self):
        return self.step > self.comm_round - 1

    def next_step(self):
        self.step += 1

        self.selected_clients = self._new_train_workload_arrage()
        self.unfinished_client_num = self.selected_client_num
        self.global_params_aggregator.clear()
        for i in range(self.num_classes):
            self.ensemble_global_params_aggregator[i].clear()

        self.model = self.model.to('cuda')

        # Allocate to CUDA if your GPUs have sufficient memory.
        for i in range(self.num_classes):
            self.models[i] = self.models[i].to('cuda')

        for client_id in self.selected_clients:
            self._ct_.get_node('client', client_id) \
                .fed_client_train_step(step=self.step, global_params=self.global_params)

    def _new_train_workload_arrage(self):
        if self.selected_client_num < self.client_num:
            selected_client = np.random.choice(range(self.client_num), self.selected_client_num, replace=False)
        elif self.selected_client_num == self.client_num:
            selected_client = np.array([i for i in range(self.client_num)])
        return selected_client
        
    def evaluate_model_group(self):
        correct = 0
        total = 0

        for model in self.models.values():
            model.eval()

        with torch.no_grad():
            for data in self.full_test_dataloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                model_outputs = torch.zeros((images.size(0), self.num_classes)).to(self.device)

                for i in range(self.num_classes):
                    
                    # Allocate to CUDA if your GPUs have sufficient memory.
                    outputs = self.models[i](images)
                    model_outputs += outputs
                
                _, predicted = torch.max(model_outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy

    def test_meta(self, test_loader, log_file_path="validation_log.txt"):
        print('==> Meta validation')
        total_loss = 0
        correct = 0
        total = 0
        device = torch.device("cuda")
        criterion = nn.CrossEntropyLoss()

        with open(log_file_path, 'a') as log_file:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    meta_transformer_outputs = torch.zeros((inputs.size(0), 10)).to(device)
                    with torch.no_grad():
                        meta_transformer_outputs = self.meta_trainer.eval_meta_transformer_ema_nograd(inputs)

                    _, predicted = torch.max(meta_transformer_outputs, 1)

                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    loss = criterion(meta_transformer_outputs, targets)
                    total_loss += loss.item()

                avg_loss = total_loss / (batch_idx + 1)
                acc = 100. * correct / total

                print(f'Loss: {avg_loss:.3f} | Acc: {acc:.3f}% ({correct}/{total})')
                log_file.write(f'Loss: {avg_loss:.3f} | Acc: {acc:.3f}% ({correct}/{total})\n')

        print("Validation results saved to log.")

    def meta_train(self):
        self.meta_trainer = MetaTransformerTrainer(self.models, self.device, nn.CrossEntropyLoss().to(self.device), self.num_classes, self.data_set)

        for round_num in range(self.args.meta_epochs):

            all_prototypes = []
            all_labels = []

            for label, proto_dict in self.prototype.items():
                for prototype in proto_dict:
                    all_prototypes.append(prototype)
                    all_labels.append(label)
            all_prototypes = torch.stack(all_prototypes).to(self.device)
            all_labels = torch.tensor(all_labels, dtype=torch.long).to(self.device)
            train_loader = torch.utils.data.DataLoader(
                list(zip(all_prototypes, all_labels)),
                batch_size=128,
                shuffle=True
            )

            self.meta_trainer.train_transformer(train_loader, num_epochs=5)
            #self.meta_trainer.train_transformer(self.full_train_dataloader, num_epochs=1)

            if (round_num + 1) % 10 == 0:
                self.test_meta(self.full_test_dataloader)
    
    def test_model(self, model, data_loader, device):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in data_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
        
        with open("KD.txt", "a") as file:
            file.write(f'Accuracy of the model on the test images: {accuracy:.2f}%\n')

        return accuracy

    def KD(self, weight, step):

        if self.data_set == 'CIFAR-10':
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        elif self.data_set == 'CIFAR-100':
            optimizer = optim.Adam(self.model.parameters(), lr=0.00001)

        netG_list = []
        for i in range(self.client_num):
            netG = Generator(1).to(self.device)
            netG_path = f'./GANs_{self.args.data_set}_{self.args.non_iid_alpha}/netG_client_{i}_epoch_199_step_0.pth'
            netG.load_state_dict(torch.load(netG_path))
            netG.eval()
            netG_list.append(netG)

        accuracy = self.test_model(self.model, self.full_test_dataloader, self.device)

        for epoch in range(self.args.KD_epochs):

            final_images = None

            for i, netG in enumerate(netG_list):
                
                z = torch.randn(32, 100, 1, 1, device=self.device)

                images = netG(z)

                if final_images is None:
                    final_images = images
                else:
                    final_images = torch.cat((final_images, images), dim=0)

            with torch.no_grad():
                meta_transformer_outputs = self.meta_trainer.eval_meta_transformer_ema_nograd(final_images)
                pred_class_argmax_teacher = meta_transformer_outputs.max(1, keepdim=True)[1]
                pred_class_argmax_teacher = pred_class_argmax_teacher.squeeze(1)

            optimizer.zero_grad()
            self.model.eval()
            student_logits = self.model(final_images)
            self.model.train()
            
            distillation_loss = knowledge_distillation(student_logits, meta_transformer_outputs, pred_class_argmax_teacher)
            print(f"Distillation Loss: {distillation_loss.item()}")
            distillation_loss.backward()

            optimizer.step()
            if (epoch+1)%10 ==0:
                accuracy = self.test_model(self.model, self.full_test_dataloader, self.device)

        for netG in netG_list:
            del netG
        del netG_list, optimizer, final_images, student_logits, meta_transformer_outputs, pred_class_argmax_teacher
        torch.cuda.empty_cache()

    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     client_model_params,
                                     eval_info):
        assert self.step == step
        self.client_eval_info.append(eval_info)

        client_prototypes = eval_info.get('prototypes', {})
        for label, proto_dict in client_prototypes.items():
            if label not in self.prototype:
                self.prototype[label] = []

            if isinstance(proto_dict, dict):
                for feature in proto_dict.values():
                    if isinstance(feature, torch.Tensor):
                        self.prototype[label].append(feature)

        client_label_counts = eval_info.get('label_counts', {i: 0 for i in range(self.num_classes)})

        weight = self.local_sample_numbers[client_id]

        self.client_test_acc_aggregator.put(eval_info['test_acc'], weight)

        self.client_label_counts[client_id] = client_label_counts
        self.global_params_aggregator.put(client_model_params, weight)

        for i in range(self.num_classes):
            ensemble_weight = client_label_counts[i]
            ensemble_weight = ensemble_weight.to('cpu')
            self.ensemble_global_params_aggregator[i].put(client_model_params, ensemble_weight)

        if self.comm_load[client_id] == 0:
            self.comm_load[client_id] = model_size(client_model_params) / 1024 / 1024 

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.server_train_test_res = {'comm. round': self.step, 'client_id': 'server'}

            self.global_params = self.global_params_aggregator.get_and_clear()

            client_test_acc_avg = self.client_test_acc_aggregator.get_and_clear()
            print('comm. round: {}, client_test_acc: {}'.format(self.step, client_test_acc_avg))

            self.model.load_state_dict(self.global_params, strict=True)

            for i in range(self.num_classes):
                self.ensemble_global_params[i] = self.ensemble_global_params_aggregator[i].get_and_clear()
                self.models[i].load_state_dict(self.ensemble_global_params[i], strict=True)

            if self.step % 10 == 0:
                model_group_accuracy = self.evaluate_model_group()
                print(f'model_group_accuracy: {model_group_accuracy:.2f}%')

            if self.step == self.args.warmup_epochs:
                self.meta_train()
            
            if self.step == self.args.warmup_epochs:
                self.KD(self.local_sample_numbers, self.step)
                self.global_params = get_cpu_param(self.model.state_dict())

            self.server_train_test_res['client_test_acc_avg'] = client_test_acc_avg
            #self.server_train_test_res['model_group_accuracy'] = model_group_accuracy

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