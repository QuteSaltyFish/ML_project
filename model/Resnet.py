import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.preprocess = torch.nn.Sequential(
            # nn.Conv3d(2, 4, 3),
            nn.AvgPool3d(2),
            nn.Conv3d(2, 32, 3, padding=8),
            nn.AvgPool3d(2)
            # nn.Conv3d(8, 8, 3),
            # nn.Conv3d(8, 16, 3),
            # nn.Conv3d(16, 32, 3),
            # nn.Conv3d(32, 32, 3),
            # nn.Conv3d(32, 32, 3),
            # nn.Conv3d(32, 32, 3),
            # nn.Conv3d(32, 32, 4),
        )
        self.conv1 = nn.Conv3d(32, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 16, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8, num_blocks[3], stride=2)
        self.linear = nn.Sequential(
            nn.Linear(8*block.expansion, num_classes),
            nn.Dropout(),
            nn.LogSoftmax()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.preprocess(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


# class testnet(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer = torch.nn.Sequential(
#             nn.Conv3d(200, 256, 3),
#             nn.MaxPool3d(2),
#             nn.Conv3d(256, 256, 3),
#             nn.Conv3d(256, 256, 3),
#             nn.Conv3d(256, 512, 3),
#             nn.Conv3d(512, 512, 3),
#             nn.Conv3d(512, 512, 3),
#             nn.Conv3d(512, 512, 3),
#             nn.Conv3d(512, 512, 3),
#             nn.Conv3d(512, 512, 4),
#         )

#     def forward(self, x):
#         return self.layer(x)
def test_net():
    return ResNet(BasicBlock, [1,1,1,1])
if __name__ == "__main__":
    
    net = test_net()
    # testnet = testnet()
    # y = net(torch.randn(1,3,32,32))
    # print(y.size())
    # model = DnCNN().to(device)
    net = net.to('cuda')
    summary(net, (2, 100, 100, 100))