'''
This is an implementation of the algorithm in
"Learning Efficient Convolutional Networks through Network Slimming".

I implemented the resnet56 network pruning. For each residual block,
I only cropped the first layer.
'''
import torch.nn as nn
import math  
import torch.nn.functional as F

class ZeroPaddingX2(nn.Module):
    def __init__(self):
        super(ZeroPaddingX2, self).__init__()
    def forward(self, x):
        num_channels = x.shape[1]
        y = F.pad(x[:, :, ::2, ::2], # stride = 2, per Kaiming's ResNet (CVPR'16) paper
            (0, 0, 0, 0, num_channels // 2, num_channels // 2), "constant", 0)
        return y

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,v,w,stride,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, v, kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(v)
        self.relu = nn.ReLU(inplace=True)
        inplanes =v
        self.conv2 = nn.Conv2d(inplanes, w, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(w)
        # self.residual=
        inplanes = w
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
#         out = self.relu(out) #############
        if self.downsample is not None:
            # if out.size()[1] != residual.size()[1]:
                # residual=nn.Conv2d(residual.size()[1], out.size()[1],kernel_size=1,bias=False)(x)
            residual = self.downsample(x)
                # print("++++++",residual.size())
        out += residual
        out = self.relu(out)
        return out

class resnet_56 (nn.Module):
    def __init__(self, dataset='cifar10', init_weights=True, cfg=None, option_ZeroPaddingX2 = False):
        super(resnet_56, self).__init__()
        self.option_ZeroPaddingX2 = option_ZeroPaddingX2
        if cfg is None:

            cfg = [16,
                16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
                32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
                64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
                ]
        self.feature = self.make_layers(cfg)
        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'cifar10':
            num_classes = 10
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg):
        inplanes = 64
        layers = []
        in_channels = 3
        for i,v in enumerate(cfg):
            downsample=None
            if i==0:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                inplanes = v
            elif i>0:
                i=2*i-1
                if i==19 or i==37:
                    stride=2
                else:
                    stride=1
                if i<55:
                    if inplanes != cfg[i+1] or stride != 1:
                        if self.option_ZeroPaddingX2:
                            downsample = ZeroPaddingX2()
                    layers.append(BasicBlock(inplanes, cfg[i], cfg[i+1], stride, downsample))
                    inplanes = cfg[i+1]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()