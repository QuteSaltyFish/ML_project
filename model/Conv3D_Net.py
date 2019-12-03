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
        self.preprocess = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(1, 32, 3, padding=8),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2)
        )
        self.Conv = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 16, kernel_size=2, stride=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc = nn.Sequential(
            # nn.Linear(6912, 128),
            nn.Linear(16,2),
            nn.Dropout(),
            nn.LogSoftmax()
        )
        self._initialize_weights()

        # self.W = nn.Parameter(t.)
    def forward(self, x):
        # y = x
        # out = self.Conv3D_Net(x)
        out = self.preprocess(x)
        out = self.Conv(out)
        # print(out.shape)
        tmp = out.view(out.shape[0],-1)
        out = self.fc(tmp)
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
    summary(model, (1, 100, 100, 100))
