import sys 
sys.path.append('/home/wangmingke/Desktop/HomeWork/ML_project')
import model
import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader import *
from model.DnCNN import DnCNN
from model import Resnet
from model import Conv3D_Net
from model.VoxNet import VoxNet
from model.baseline import FC_Net
from model.func import save_model, eval_model_new_thread, eval_model, load_model
import argparse
import torchvision as tv
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

if __name__ == "__main__":
    data_set = MyDataSet()
    z = [] 
    x = []
    y = []
    z2 = [] 
    x2 = []
    y2 = []
    for idx, data in enumerate(data_set):
        try:# print(t.where(data!=0))
            tmpz = t.where(data!=0)[0]
            tmpx = t.where(data!=0)[1]
            tmpy = t.where(data!=0)[2]
            z.append(tmpz.min())
            x.append(tmpx.min())
            y.append(tmpy.min())
            z2.append(tmpz.max())
            x2.append(tmpx.max())
            y2.append(tmpy.max())
            if x[-1]<25:
                print(x[-1], x2[-1])
            if y[-1]<25:
                print(y[-1], y2[-1])
            if z[-1]<25:
                print(z[-1], z2[-1])
        except:
            break
lens = len(x)
x = sorted(x)
y = sorted(y)
z = sorted(z)
x2 = sorted(x2)
y2 = sorted(y2)
z2 = sorted(z2)

idx1 = (int)(0.05*lens)
idx2 = (int)(0.95*lens)
x = x[idx1:idx2]
y = y[idx1:idx2]
z = z[idx1:idx2]
x2 = x2[idx1:idx2]
y2 = y2[idx1:idx2]
z2 = z2[idx1:idx2]

print(min(z),min(x),min(y),min(z2),min(x2),min(y2))
print('done')

        
    
    