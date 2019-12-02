import torch as t 
from torchvision import models
from torchsummary import summary
import re
import os
import glob
import datetime
import time
import numpy as np
import torch.nn as nn
import json


class Conv3D_Net(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=100, use_bnorm=True, kernel_size=3):
        super(Conv3D_Net, self).__init__()
        self.config = json.load(open('config.json'))
        # padding = kernel_size//2
        # layers = []
        
        # layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
        #                         kernel_size=kernel_size, padding=padding, bias=True))
        # layers.append(nn.ReLU(inplace=True))
        # for _ in range(depth-2):
        #     layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
        #                             kernel_size=kernel_size, padding=padding, bias=False))
        #     layers.append(nn.BatchNorm2d(
        #         n_channels, eps=0.0001, momentum=0.95))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
        #                         kernel_size=kernel_size, padding=padding, bias=False))
        # self.Conv3D_Net = nn.Sequential(*layers)
        self.layer = nn.Sequential(
            nn.Conv3d(2, 8, 3)
        )
        self._initialize_weights()

    def forward(self, x):
        # y = x
        # out = self.Conv3D_Net(x)
        out = self.layer(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    model = Conv3D_Net().to(device)
    summary(model, (2, 100, 100, 100))
