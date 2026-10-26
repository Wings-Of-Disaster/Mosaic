import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import random

from experiments.utils.deepinversion_cifar import get_loss
from experiments.utils.dcgan_model import Generator, Discriminator

def train_generator(step, num_classes, client_id, niter, inc_classes, dataloader, nz, device, netD, netG, classifier, optimizerD, outf, optimizerG, criterion, criterion_sum, inversion_loss, inversion_model, flag):  

    real_label = 1
    fake_label = 0

    for epoch in range(niter):
            num_greater_thresh = 0
            count_class = [0]*num_classes
            count_class_less = [0]*num_classes
            count_class_hist = [0]*num_classes
            count_class_less_hist = [0]*num_classes
            classification_loss_sum = 0
            errD_real_sum = 0
            errD_fake_sum = 0
            errD_sum = 0
            errG_adv_sum = 0
            data_size = 0 
            accD_real_sum = 0
            accD_fake_sum = 0
            accG_sum = 0
            accD_sum = 0
            div_loss_sum = 0
            for i, data in enumerate(dataloader, 0):

                netD.zero_grad()
                real_cpu = torch.from_numpy(data[0].numpy()[np.isin(data[1],inc_classes)]).to(device)
                batch_size = real_cpu.size(0)
                if batch_size > 0:
                    data_size += batch_size
                    label = torch.full((batch_size,), real_label, device=device).float()
                else:
                    print("No samples found for in this batch.")
                    continue

                output = netD(real_cpu)
                
                errD_real = criterion(output, label)
                errD_real_sum = errD_real_sum + (criterion_sum(output,label)).cpu().data.numpy()

                accD_real = (label[output>0.5]).shape[0]
                accD_real_sum = accD_real_sum + float(accD_real)

                errD_real.backward()
                
                D_x = output.mean().item()

                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)

                fake_class = classifier(fake)
                sm_fake_class = F.softmax(fake_class, dim=1)
                
                class_max = fake_class.max(1,keepdim=True)[0]
                class_argmax = fake_class.max(1,keepdim=True)[1]
                
                classification_loss = torch.mean(torch.sum(-sm_fake_class*torch.log(sm_fake_class+1e-5),dim=1))
                classification_loss_add = torch.sum(-sm_fake_class*torch.log(sm_fake_class+1e-5))
                classification_loss_sum = classification_loss_sum + (classification_loss_add).cpu().data.numpy() 
                
                sm_batch_mean = torch.mean(sm_fake_class,dim=0)
                div_loss = torch.sum(sm_batch_mean*torch.log(sm_batch_mean))
                div_loss_sum = div_loss_sum + div_loss*batch_size

                label.fill_(fake_label)
                label = label.float()

                output = netD(fake.detach())

                errD_fake = criterion(output, label)
                errD_fake_sum = errD_fake_sum + (criterion_sum(output, label)).cpu().data.numpy()

                accD_fake = (label[output<=0.5]).shape[0]
                accD_fake_sum = accD_fake_sum + float(accD_fake)

                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                errD_sum = errD_real_sum + errD_fake_sum

                accD = accD_real + accD_fake
                accD_sum = accD_real_sum + accD_fake_sum

                optimizerD.step()

                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake)
                c_l = 0
                d_l = 5
                i_l = 10

                errG_adv = criterion(output, label) 
                errG_adv_sum = errG_adv_sum + (criterion_sum(output, label)).cpu().data.numpy()

                accG = (label[output>0.5]).shape[0]
                accG_sum = accG_sum + float(accG)
                
                if inversion_loss:
                    feature_loss = get_loss(inversion_model, fake, bn_reg_scale=0.001)
                    if flag == 1:
                        errG = errG_adv + c_l * classification_loss + d_l * div_loss + i_l * feature_loss
                    else: 
                        errG = errG_adv + c_l * classification_loss + d_l * div_loss - i_l * feature_loss
                else:
                    errG = errG_adv + c_l * classification_loss + d_l * div_loss

                errG_sum = errG_adv_sum + c_l * classification_loss_sum + d_l * div_loss_sum
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, niter, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                pred_class = F.softmax(fake_class,dim=1).max(1, keepdim=True)[0]
                pred_class_argmax = F.softmax(fake_class,dim=1).max(1, keepdim=True)[1]
                num_greater_thresh = num_greater_thresh + (torch.sum(pred_class > 0.9).cpu().data.numpy())
                for argmax, val in zip(pred_class_argmax, pred_class):
                    if val > 0.9:
                        count_class_hist.append(argmax)
                        count_class[argmax] = count_class[argmax] + 1
                    else:
                        count_class_less_hist.append(argmax)
                        count_class_less[argmax] = count_class_less[argmax] + 1
          
            current_errG = errG_sum / data_size

            if (epoch+1) % 100 == 0:

                os.makedirs(outf, exist_ok=True)
                torch.save(netG.state_dict(), '%s/netG_client_%d_epoch_%d_step_%d.pth' % (outf, client_id, epoch, step))
                torch.save(netD.state_dict(), '%s/netD_client_%d_epoch_%d_step_%d.pth' % (outf, client_id, epoch, step))