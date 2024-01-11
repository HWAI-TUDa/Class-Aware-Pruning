import torch.optim as optim
import torch
from models.vgg import *
# from utils.utils import *
from utils import *
# from utils.cifar10_loader import *
# from data.cifar10_loader_Hrank import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):

        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        activation_index = 0

        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        return self.activation_to_layer, self.model.classifier(x)

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        taylor = torch.abs(activation * grad)

        score_imp = torch.gt(taylor, 1e-50) # importance score

        score_imp = torch.where(score_imp, 1., 0.)
        score_imp = score_imp.reshape(score_imp.size()[0], score_imp.size()[1], -1).cuda()
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(score_imp.size()).zero_()
            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
        self.filter_ranks[activation_index] += score_imp
        self.grad_index += 1

    def get_filter_ranks(self):
        return self.filter_ranks


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PrunningFineTuner:
    def __init__(self, model, batch_size=5000, num_class =100):
        self.batch_size = batch_size
        self.train_data_loader = loader(self.batch_size)
        self.test_data_loader = test_loader()
        self.classes_loader = classes_loader()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_class = num_class
        self.prunner = FilterPrunner(self.model)
        self.model.train()

        # self.AverageMeter = AverageMeter()
    def deconv_orth_dist(self, kernel, stride = 2, padding = 1):
        [o_c, i_c, w, h] = kernel.shape
        output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
        ct = int(np.floor(output.shape[-1]/2))
        target[:,:,ct,ct] = torch.eye(o_c).cuda()
        return torch.norm(output - target )

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def test(self):
        self.model.eval()
        total_loss = 0
        total_loss_ce = 0
        correct = 0
        total = 0
        loss = 0
        for i, (batch, label) in enumerate(self.test_data_loader):
            # if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()
            input = Variable(batch)
            output = self.model(input)
            pred = output.data.max(1)[1]
            correct += pred.cuda().eq(label).sum()
            total += label.size(0)

            loss = self.criterion(output, label)

            total_loss += loss.item() * batch.size(0)
            
        average_loss = total_loss / total
        average_acc = float(correct) / total
        
        print('[Test]\t'
            'acc {top1:.4f}\t'
            'loss {loss:.4f}\t'.format(top1=100*average_acc, loss=average_loss))
        print('')
        self.model.train()
        return average_acc


    def train(self, optimizer = None, scheduler = None, epoches=10,
              early_stop = False, early_stop_acc =0.90, prune_stop_acc = 0.9,
              orth=False,  r_orth = 0.01, l1_reg =False, l1_lambda = 1e-4):

        stop_prune = False
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(),lr = 0.01,momentum = 0.9, weight_decay = 5e-4)
        if scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)

        best_accuracy = 0.0
        stop_training = False
        for i in range(epoches):
            lr_1 = optimizer.param_groups[0]['lr']
            print("[Epoch]:", i, 'LR: ', "{:.0e}".format(lr_1))
            self.train_epoch(optimizer, rank_filters = False,
                             orth = orth, r_orth = r_orth, l1_reg =l1_reg, l1_lambda = l1_lambda)
            scheduler.step()
            acc_test = self.test()

            if early_stop:
                if acc_test > early_stop_acc:
                  print('acc_test:',acc_test, 'early_stop_acc:', early_stop_acc)
                  stop_training = True
                  print("Early stopping triggered. Accuracy > {}%. Stopping training.".format(early_stop_acc))
                  break
                if acc_test > best_accuracy:
                    est_accuracy = acc_test

        if acc_test < prune_stop_acc:
              stop_prune = True

        print("Finished fine tuning.")
        return stop_prune

    def rank_train_batch(self, optimizer, batch, label, orth=False, r_orth = 0.01, l1_reg =False, l1_lambda = 1e-4):
        batch = batch.cuda()
        label = label.cuda()
        self.model.zero_grad()
        input = Variable(batch)
        self.model.eval()

        activation_to_layer, output = self.prunner.forward(input)
        self.criterion(output, Variable(label)).backward()
        score_imp_bm_batch = self.prunner.get_filter_ranks()
        #------------------------------------------------------
        predicted_classes = output.data.max(1)[1]
        correct_mask = torch.eq(predicted_classes, label)
        keep_indices = torch.nonzero(correct_mask).squeeze()
        if keep_indices.numel() == 0:
            keep_indices = torch.arange(len(correct_mask))

        for key, value in score_imp_bm_batch.items():
            score_imp_bm_batch[key] = value.mean(dim=(0)).data
        return activation_to_layer, score_imp_bm_batch

    def train_epoch(self, optimizer = None,
                    orth=False, r_orth = 0.01,
                    l1_reg =False, l1_lambda = 1e-4
                    ):
        correct = 0
        total = 0

        losses = AverageMeter()
        top1 = AverageMeter()

        for i, (batch, label) in enumerate(self.train_data_loader):
            batch = batch.cuda()
            label = label.cuda()
            self.model.zero_grad()
            input = Variable(batch)
            optimizer.zero_grad()
            output = self.model(input)
            target = Variable(label)

            loss = self.criterion(output, target)
            #---------------- conv_orth  -----------------
            if orth:
                diff = 0
                for layer, (name, module) in enumerate(self.model.features._modules.items()):
                    if isinstance(module, torch.nn.modules.conv.Conv2d):
                        if module.kernel_size[0] == 3:
                            diff += self.deconv_orth_dist(module.weight, stride=1)
                # print('r_orth * diff: ', r_orth * self.diff)
                loss += r_orth * diff
            #---------------- L1  -----------------------
            if l1_reg:
                l1_regularization = torch.tensor(0.).to(device)
                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, p=1)
                loss += l1_lambda * l1_regularization

            #--------------------------------------------------------

            loss.backward()
            optimizer.step()
            # measure accuracy and record loss
            output = output.float()
            loss = loss.float()
            prec1 = self.accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

        print('[Train]\t'
            'acc {top1.avg:.4f}\t'
            'loss {loss.avg:.4f}\t'.format(loss=losses,top1=top1))


    def train_classes_epoch(self, class_idx, optimizer = None,
                            # rank_filters = True
                            ):

      for i, (batch, label) in enumerate(classes_loader(class_idx)):
        activation_to_layer, self.score_imp_bm_batch_tempo = self.rank_train_batch(optimizer, batch, label)
        n_batch = i+1

      for j in self.score_imp_bm_batch_tempo: # 对多个batch 进行平均
        self.score_imp_bm_batch_tempo[j] = self.score_imp_bm_batch_tempo[j]/n_batch
      return activation_to_layer, self.score_imp_bm_batch_tempo, batch, label#####


    def get_candidates_to_prune(self, class_idx):
        self.prunner.reset()
        activation_to_layer, score_imp_ave_one_class, batch, label = self.train_classes_epoch(class_idx)
        return activation_to_layer, score_imp_ave_one_class , batch, label

    def get_important_score_of_filters(self):
      #Get the accuracy before prunning
      # self.test()
      self.model.train()
      #Make sure all the layers are trainable
      for param in self.model.features.parameters():
          param.requires_grad = True
      activation_to_layer, self.accumulate_importance_for_classes, batch0, label0 = self.get_candidates_to_prune(0)
      # print(accumulate_importance_for_classes)
      for key_init,value_init in self.accumulate_importance_for_classes.items():
        self.accumulate_importance_for_classes[key_init] = torch.FloatTensor(value_init.size()).zero_()

      print_freq =  int(self.num_class/5)
      for class_index in range(self.num_class): #对类的循环
        if class_index % print_freq == 0:
          print(f'Processing {class_index + print_freq}/{self.num_class} classes  ')
        activation_to_layer, self.importance_for_classes, batch, label = self.get_candidates_to_prune(class_index)

        for key in self.importance_for_classes.keys():
          self.accumulate_importance_for_classes[key] = (
                  self.accumulate_importance_for_classes[key] + self.importance_for_classes[key].cpu())

      for key in self.accumulate_importance_for_classes:
        self.accumulate_importance_for_classes[key] = torch.max(self.accumulate_importance_for_classes[key],1).values
      return activation_to_layer, self.accumulate_importance_for_classes
    
#========================================================================================

def total_num_filters(model_1):
    filters = 0
    for name, module in model_1.features._modules.items():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            filters = filters + module.out_channels
    return filters

def generate_mask_top_n(lst, n):
    # 找出最大的 n 个数的索引
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:n]

    # 生成 mask
    mask = torch.zeros(len(lst))
    for idx in indices:
        mask[idx] = 1
    return mask

def get_conv2d_layers(model):
    conv2d_layers = []
    # 获取模型的迭代器
    model_iter = iter(model)
    while True:
        try:
            layer = next(model_iter)
            if isinstance(layer, nn.Conv2d):
                conv2d_layers.append(layer)
        except StopIteration:
            break
    return conv2d_layers


def get_conv_BN_fc_layers(model):
    conv2d_layers = []
    bn_layers = []
    fc_layers = []
    model_modules = model.modules()
    # 获取模型的迭代器
    model_iter = iter(model_modules)
    while True:
        try:
          layer = next(model_iter)
          if isinstance(layer, nn.Conv2d):
            conv2d_layers.append(layer)

          if isinstance(layer, nn.BatchNorm2d):
            bn_layers.append(layer)

          if isinstance(layer, nn.Linear):
            fc_layers.append(layer)

        except StopIteration:
            break
    return conv2d_layers, bn_layers, fc_layers

def my_zip_conv_bn_fc(net1, net2, pruned_layer_by_conv2d):
  # matched_layers = []
  matched_conv_layers = []
  matched_BN_layers = []
  matched_fc_layers = []

  modules_conv_1, modules_BN_1, modules_fc_1 = get_conv_BN_fc_layers(net1)
  modules_conv_2, modules_BN_2, modules_fc_2 = get_conv_BN_fc_layers(net2)
  idx1_conv2d = 0
  idx2_conv2d = 0

  for layer1 in modules_conv_1:
      if idx1_conv2d in pruned_layer_by_conv2d:# 被完全裁掉的层
          idx1_conv2d +=1

      else:
          layer2 = modules_conv_2[idx2_conv2d]
          matched_conv_layers.append((layer1, layer2))

          bn_layer1 = modules_BN_1[idx1_conv2d]
          bn_layer2 = modules_BN_2[idx2_conv2d]

          matched_BN_layers.append((bn_layer1,bn_layer2))

          idx1_conv2d += 1
          idx2_conv2d += 1

  matched_fc_layers.append((modules_fc_1[0], modules_fc_2[0]))
  matched_fc_layers.append((modules_fc_1[1], modules_fc_2[1]))

  return matched_conv_layers, matched_BN_layers, matched_fc_layers


import torch
from torch.autograd import Variable
from functools import reduce
import operator

count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)

def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_layer_info(layer):
    layer_str = str(layer)
    # print(layer_str)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_layer_param(model, is_conv=True):
    if is_conv:
        total=0.
        for idx, param in enumerate(model.parameters()):
            assert idx<2
            f = param.size()[0]
            # pruned_num = int(model.cp_rate * f)
            pruned_num = 0
            if len(param.size())>1:
                c=param.size()[1]
                if hasattr(model,'last_prune_num'):
                    last_prune_num=model.last_prune_num
                    total += (f - pruned_num) * (c-last_prune_num) * param.numel() / f / c
                else:
                    total += (f - pruned_num) * param.numel() / f
            else:
                total += (f - pruned_num) * param.numel() / f
        return total
    else:
        return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

## The input batch size should be 1 to call this function
def measure_layer(layer, x, print_name):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        # pruned_num = int(layer.cp_rate * layer.out_channels)
        pruned_num = 0
        if hasattr(layer,'tmp_name') and 'trans' in layer.tmp_name:
            delta_ops = (layer.in_channels-layer.last_prune_num) * (layer.out_channels - pruned_num) * layer.kernel_size[0] * \
                        layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        else:
            delta_ops = layer.in_channels * (layer.out_channels-pruned_num) * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add

        delta_ops_ori = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add

        delta_params = get_layer_param(layer)


    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer, is_conv=False)

        # print('linear:',layer, delta_ops, delta_params)

    elif type_name in ['DenseBasicBlock', 'ResBasicBlock']:
        measure_layer(layer.conv1, x)

    elif type_name in ['Inception']:
        measure_layer(layer.conv1, x)

    elif type_name in ['DenseBottleneck', 'SparseDenseBottleneck']:
        measure_layer(layer.conv1, x)

    elif type_name in ['Transition', 'SparseTransition']:
        measure_layer(layer.conv1, x)

    elif type_name in ['ReLU', 'BatchNorm1d','BatchNorm2d', 'Dropout2d', 'DropChannel',
                       'Dropout', 'AdaptiveAvgPool2d', 'AvgPool2d', 'MaxPool2d', 'Mask',
                       'channel_selection', 'LambdaLayer', 'Sequential']:
        return
    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return

def measure_model(model, device, C, H, W, print_name=False):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, C, H, W)).to(device)
    model = model.to(device)
    model.eval()

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model, print_name):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x, print_name)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child, print_name)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model, print_name)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params


# -*- coding:utf8 -*-
import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable

import numpy as np


def print_model_param_nums(model=None):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.3fM' % (total / 1e6))
    return total


def print_model_param_flops(model=None, input_res=[32, 32], multiply_adds=True):
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = ((kernel_ops * (2 if multiply_adds else 1) + bias_ops) *
                 output_channels * output_height * output_width * batch_size)

        list_conv.append(flops)


    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)
    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(3,input_res[1],input_res[0]).unsqueeze(0), requires_grad = True)
    out = model(input.cuda())

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) +
                   sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    print('  + Number of FLOPs: %.3fG' % (total_flops / 1e9))
    return total_flops