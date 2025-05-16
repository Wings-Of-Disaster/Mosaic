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
from utils.dcgan_model import Generator, Discriminator
from utils.degan import train_generator
from utils.model import evaluation, CycleDataloader
from models.model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils.deepinversion_cifar import DeepInversionFeatureHook

from sklearn.cluster import KMeans

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def test_model(model, data_loader, device):
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

    return accuracy

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
        self.train_batch_data_iter_new = CycleDataloader(self.train_dataloader)
        self.train_label_list = args.data_distributer.get_client_label_list(client_id)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.model = None
        self.optimizer = None

        if self.data_set == 'CIFAR-10':
            self.label_counts = {i: 0 for i in range(10)} 
        elif self.data_set == 'CIFAR-100':
            self.label_counts = {i: 0 for i in range(100)} 
        else:
            raise ValueError(f"Unsupported dataset: {self.data_set}")
        
        n_classes = len(self.label_counts)
        for i in range(n_classes):
            self.label_counts[i] = 0
        label_counts_in_batch = torch.zeros(n_classes, dtype=torch.long)
        for batch in self.train_dataloader:
            _, y = batch
            label_counts_in_batch += torch.bincount(y, minlength=n_classes)
        for i in range(len(label_counts_in_batch)):
            self.label_counts[i] += label_counts_in_batch[i]
            
        self.res = {}
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)

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

    def degan(self, step):
        try:
            os.makedirs(self.args.outf)
        except OSError:
            pass
        cudnn.benchmark = True
        nc=3
        ngpu = int(self.args.ngpu)
        nz = int(self.args.nz)
        ngf = int(self.args.ngf)
        ndf = int(self.args.ndf)

        if self.data_set == 'CIFAR-10':
            classifier = ResNet18(num_classes=10)
        elif self.data_set == 'CIFAR-100':
            classifier = ResNet50(num_classes=100)
        else:
            raise ValueError(f"Unsupported dataset: {self.data_set}")

        optimizer_c = optim.SGD(params=classifier.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)

        #By default, our classifier undergoes training for 50 epochs.
        num_epochs = 50

        classifier = classifier.to(self.device)
        criterion = nn.CrossEntropyLoss()
        train_loader = self.train_dataloader

        for epoch in range(num_epochs):
            classifier.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer_c.zero_grad()
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_c.step()
        print('Finished Training')
        
        netG = Generator(ngpu).to(self.device)
        netG.apply(weights_init)
        if self.args.netG != '':
            netG.load_state_dict(torch.load(self.args.netG))

        netD = Discriminator(ngpu).to(self.device)
        netD.apply(weights_init)
        if self.args.netD != '':
            netD.load_state_dict(torch.load(self.args.netD))

        criterion = nn.BCELoss()
        criterion_sum = nn.BCELoss(reduction = 'sum')

        fixed_noise = torch.randn(self.args.batch_size, nz, 1, 1, device=self.device)

        optimizerD = optim.Adam(netD.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))
        threshold = []
        
        inc_classes = [index for index, count in enumerate(self.label_counts) if count != 0]
        
        best_errG = float('inf')
        best_epoch = -1
        
        #We initialize a classifier at step = 0, or use the global model as the classifier after training for a certain number of epochs.
        if step == 0:
            train_generator(step=step, num_classes=self.num_classes, client_id=self.client_id, niter=self.args.niter, inc_classes=inc_classes, dataloader=self.train_dataloader, nz=nz, 
            device=self.device, netD=netD, netG=netG, classifier=self.model, optimizerD=optimizerD, optimizerG=optimizerG,
            outf=self.args.outf, criterion=criterion, criterion_sum=criterion_sum, inversion_loss=False, inversion_model=self.model, flag=0)
        
        if step == 249:
            train_generator(step=step, num_classes=self.num_classes, client_id=self.client_id, niter=self.args.niter, inc_classes=inc_classes, dataloader=self.train_dataloader, nz=nz, 
            device=self.device, netD=netD, netG=netG, classifier=self.model, optimizerD=optimizerD, optimizerG=optimizerG,
            outf=self.args.outf, criterion=criterion, criterion_sum=criterion_sum, inversion_loss=False, inversion_model=self.model, flag=0)

        del classifier, optimizer_c, optimizerD, optimizerG, netD, netG
        torch.cuda.empty_cache()

    def train_locally_step(self, I, step, global_params):

        if step <= self.args.warmup_epochs:
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr_lm, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=0.0001, weight_decay=self.weight_decay)
        
        LOSS = 0
        feature_dict = {i: [] for i in range(len(self.label_counts))}
        self.model.eval()

        for tau in range(I):
            x, y = next(self.train_batch_data_iter)
            x = x.to(self.device)
            y = y.to(self.device)
            _, feature = self.model(x, out_feature=True)

            for i in range(len(y)):
                label = y[i].item()
                feature_dict[label].append(feature[i].detach().cpu())

        self.model.train()

        for tau in range(I):
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad(set_to_none=True)

            if step <= self.args.warmup_epochs:
                x, y = next(self.train_batch_data_iter)
                x = x.to(self.device)
                y = y.to(self.device)
            else:
                x, y = next(self.train_batch_data_iter_new)
                x = x.to(self.device)
                y = y.to(self.device)

            local_outputs = self.model(x)
            
            loss = self.criterion(local_outputs, y)
            loss.backward()

            self.optimizer.step()
            LOSS += loss.item()

        LOSS = LOSS / I
        prototypes = {}

        k = self.args.cluster_k
        
        for label, features in feature_dict.items():
            if features:
                stacked_features = torch.stack(features)
                if len(stacked_features) < k:
                    mean_feature = stacked_features.mean(dim=0)
                    prototypes[label] = {
                        "mean": mean_feature
                    }
                else:
                    stacked_features_np = stacked_features.numpy()
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(stacked_features_np)
                    cluster_centers = kmeans.cluster_centers_
                    cluster_prototypes = [torch.tensor(center) for center in cluster_centers]
                    prototypes[label] = {
                        #"mean": stacked_features.mean(dim=0),
                        f"kmeans_{i}": cluster_prototypes[i] for i in range(10)
                    }

        self.res.update(m_LOSS=LOSS)
        self.res.update(label_counts=self.label_counts)
        self.res.update(prototypes=prototypes)

    def get_eval_info(self, step, train_time=None):
        self.res.update(train_time=train_time)

        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.test_dataloader,
                                    criterion=self.criterion,
                                    model_params=None,
                                    device=self.device)
        self.res.update(test_loss=loss, test_acc=acc, test_sample_size=num)
