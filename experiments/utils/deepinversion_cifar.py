import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.utils as vutils
import collections
import glob

class DeepInversionFeatureHook():


    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def get_images(net, bs=256, epochs=1000, idx=-1, var_scale=0.00005,
               net_student=None, prefix=None, competitive_scale=0.01, train_writer = None, global_iteration=None,
               optimizer = None, inputs = None, bn_reg_scale = 0.0, random_labels = False, l2_coeff=0.0):
  
    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
    best_cost = 1e6
    inputs.data = torch.randn((bs, 3, 32, 32), requires_grad=True, device='cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer.state = collections.defaultdict(dict)

    if random_labels:
        targets = torch.LongTensor([random.randint(0,9) for _ in range(bs)]).to('cuda')
    else:
        targets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 25 + [0, 1, 2, 3, 4, 5]).to('cuda')

    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    lim_0, lim_1 = 2, 2

    for epoch in range(epochs):

        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))

        optimizer.zero_grad()
        net.zero_grad()
        outputs = net(inputs_jit)
        loss = criterion(outputs, targets)
        loss_target = loss.item()

        diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
        diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
        diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
        diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale*loss_var

        # R_feature loss
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
        loss = loss + bn_reg_scale*loss_distr
        # l2 loss
        if 1:
            loss = loss + l2_coeff * torch.norm(inputs_jit, 2)

        if epoch % 200==0:
            print(f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")
            vutils.save_image(inputs.data.clone(),
                              './{}/output_{}.png'.format(prefix, epoch//200),
                              normalize=True, scale_each=True, nrow=10)

        if best_cost > loss.item():
            best_cost = loss.item()
            best_inputs = inputs.data

        loss.backward()

        optimizer.step()

    name_use = "best_images"
    if prefix is not None:
        name_use = prefix + name_use
    next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1

    vutils.save_image(best_inputs[:20].clone(),
                      './{}/output_{}.png'.format(name_use, next_batch),
                      normalize=True, scale_each = True, nrow=10)
    
    del net
    torch.cuda.empty_cache()

    return best_inputs, targets


def get_loss(net, fake_images, bs=256, epochs=1, var_scale=0.00005, 
             bn_reg_scale=0.0):
    
    net.eval()

    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    with torch.no_grad():
        net(fake_images)

    loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
    loss = bn_reg_scale * loss_distr

    for hook in loss_r_feature_layers:
        hook.close()

    net.train()
    
    return loss