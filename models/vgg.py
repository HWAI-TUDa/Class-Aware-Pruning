import math
import torch.nn as nn
from collections import OrderedDict

norm_mean, norm_var = 0.0, 1.0

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
cfg_19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
#-------------------------------------------------------------------------------
class VGG(nn.Module):
    def __init__(self, num_classes=100, init_weights=True, cfg=None, fc_option = 'B'):
        super(VGG, self).__init__()
        self.features = nn.Sequential()

        if cfg is None:
            # cfg = defaultcfg
            cfg = cfg_19
        self.features = self.make_layers(cfg[:-1], True)

        if fc_option == 'A': # A for HRanking VGG16 cifar10
          self.features = self.make_layers(cfg[:-1], True)

          self.classifier = nn.Sequential(OrderedDict([
              # ('dropout',nn.Dropout(0.5)),
              ('linear1', nn.Linear(cfg[-2], cfg[-1])),
              ('norm1', nn.BatchNorm1d(cfg[-1])),
              ('relu1', nn.ReLU(inplace=True)),
              ('linear2', nn.Linear(cfg[-1], num_classes)),
          ]))

        if fc_option == 'B': # B for TPP VGG19 cifar100
          self.features = self.make_layers(cfg, True)
          self.classifier = nn.Sequential(OrderedDict([
              # ('dropout',nn.Dropout(0.5)),
              ('linear1', nn.Linear(cfg[-1], num_classes)),
          ]))


        if init_weights:
            self._initialize_weights()
    def make_layers(self, cfg, batch_norm=True):
        layers = nn.Sequential()
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = v
        return layers

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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

# def cfg_19():
#     cfg_19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
#     return VGG(cfg=cfg_19, num_classes=100, init_weights=True, fc_option = 'B')