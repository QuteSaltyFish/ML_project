#%%
import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader_v4 import *
from model.DnCNN import DnCNN
from model import Resnet
from model import Conv3D_Net
from model.VoxNet_66 import VoxNet
from model.baseline import FC_Net
from model.func import save_model, eval_model_new_thread, eval_model, load_model
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

#%%
t.backends.cudnn.benchmark=True
time_start = time.time()
config = json.load(open("config.json"))
# os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
DEVICE = t.device(config["DEVICE"])
LR = config['lr']
LR = 1e-4
EPOCH = config['epoch']
WD = config['Weight_Decay']
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
parser.add_argument("--epoch", default=0, type=int,
                    help="The epoch to be tested")
parser.add_argument("--name", default='VoxNet_DA2', type=str,
                    help="Whether to test after training")
args = parser.parse_args()

DataSet = MyDataSet()

# using K-fold
np.random.seed(1998)
kf = KFold(n_splits=5)
idx = np.arange(len(DataSet))
# writer = SummaryWriter('runs/{}_final'.format(args.name))

train_data = data_set(idx, train=True, name=args.name)
# train_data.data_argumentation()

train_loader = DataLoader.DataLoader(
    train_data, batch_size=config["batch_size"], shuffle=True, drop_last=False)

model = VoxNet(2).to(DEVICE)


optimizer = t.optim.SGD(model.parameters(), lr=LR)
print(optimizer.param_groups[0]['lr'])
# optimizer = t.optim.Adam(model.parameters())
criterian = t.nn.CrossEntropyLoss().to(DEVICE)

# Test the train_loader
for epoch in range(args.epoch, EPOCH):
    model = model.train()
    train_loss = 0
    correct = 0
    if epoch>40:
        optimizer.param_groups[0]['lr'] = 1e-5
    if epoch>80:
        optimizer.param_groups[0]['lr'] = 1e-6
    for batch_idx, [data, label] in enumerate(train_loader):
        data, label = data.to(DEVICE), label.to(DEVICE)
        out = model(data).squeeze()
        loss = criterian(out, label.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(label.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)

    # train_l.append(train_loss)
    # train_a.append(train_acc)
    save_model(model, epoch, '{}_final'.format(args.name))

    print('\nEpoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, train_loss, correct, len(train_loader.dataset), train_acc))


# %%
