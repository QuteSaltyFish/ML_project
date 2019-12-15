#%%
import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader_v3 import *
from model.DnCNN import DnCNN
from model import Resnet
from model import Conv3D_Net
from model.VoxNet_v1 import VoxNet
from model.baseline import FC_Net
from model.func import save_model, eval_model_new_thread, eval_model, load_model
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

#%%
t.backends.cudnn.benchmark=True
time_start = time.time()
config = json.load(open("config.json"))
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DEVICE = t.device(config["DEVICE"])
LR = config['lr']
LR = 1e-3
EPOCH = config['epoch']
WD = config['Weight_Decay']
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
parser.add_argument("--epoch", default=0, type=int,
                    help="The epoch to be tested")
parser.add_argument("--lr", default=LR, type=int,
                    help="The epoch to be tested")
parser.add_argument("--name", default='VoxNet_v1_{}_DA'.format(LR), type=str,
                    help="Whether to test after training")
args = parser.parse_args()
LR = args.lr
DataSet = MyDataSet()

# using K-fold
np.random.seed(1998)
kf = KFold(n_splits=5)
idx = np.arange(len(DataSet))
np.random.shuffle(idx)
print(args.name, kf.get_n_splits(idx))
# shuffle the data before the
for K_idx, [train_idx, test_idx] in enumerate(kf.split(idx)):
    writer = SummaryWriter('runs/{}_{}_Fold'.format(args.name, K_idx+1))

    train_data, test_data = data_set(train_idx, train=True), data_set(test_idx, train=False)
    # train_data.data_argumentation()
    
    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

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
        # if epoch>50:
        #     optimizer.param_groups[0]['lr'] = 1e-5
        for batch_idx, [data, label] in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            out = model(data).squeeze()
            loss = criterian(out, label)
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
        

        print('\nEpoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss, correct, len(train_loader.dataset), train_acc))

        model = model.eval()

        with t.no_grad():
            # Test the test_loader
            test_loss = 0
            correct = 0
            for batch_idx, [data, label] in enumerate(test_loader):
                data, label = data.to(DEVICE), label.to(DEVICE)
                out = model(data)
                # monitor the upper and lower boundary of output
                # out_max = t.max(out)
                # out_min = t.min(out)
                # out = (out - out_min) / (out_max - out_min)
                test_loss += criterian(out, label)
                pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct += pred.eq(label.view_as(pred)).sum().item()
            # store params
            for name, param in model.named_parameters():
                writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), epoch)

            test_loss /= len(test_loader.dataset)
            test_acc = 100. * correct / len(test_loader.dataset)

            # test_l.append(test_loss)
            # test_a.append(test_acc)
            
            print('Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, test_loss, correct, len(test_loader.dataset), test_acc))
        save_model(model, epoch, '{}_{}_folds'.format(args.name, K_idx+1))
        # eval_model_new_thread(epoch, 0)
        # LZX pls using the following code instead
        # multiprocessing.Process(target=eval_model(epoch, '0'), args=(multiprocess_idx,))
        # multiprocess_idx += 1
        writer.add_scalar('Training/Training_Loss', train_loss, epoch)
        writer.add_scalar('Training/Training_Acc', train_acc, epoch)
        writer.add_scalar('Testing/Testing_Loss', test_loss, epoch)
        writer.add_scalar('Testing/Testing_Acc', test_acc, epoch)
    writer.close()

#%%  
training_loss = np.zeros(EPOCH, dtype=np.float)
testing_loss = np.zeros(EPOCH, dtype=np.float)
training_acc = np.zeros(EPOCH, dtype=np.float)
testing_acc = np.zeros(EPOCH, dtype=np.float)

# compute the mean acc and loss
import os
from tensorboard.backend.event_processing import event_accumulator
dirs = ['runs/{}_{}_Fold'.format(args.name, i+1) for i in range(5)]
writer = SummaryWriter('runs/{}'.format(args.name))
for dir in dirs:
    try:
        data = os.listdir(dir)
    except:
        break
    print(data)
    ea = event_accumulator.EventAccumulator(os.path.join(dir, data[0]))
    ea.Reload()
    # print(ea.scalars.Keys())
    train_loss = ea.scalars.Items('Training/Training_Loss')
    training_loss += np.array([i.value for i in train_loss])

    train_acc = ea.scalars.Items('Training/Training_Acc')
    training_acc += np.array([i.value for i in train_acc])

    test_loss = ea.scalars.Items('Testing/Testing_Loss')
    testing_loss += np.array([i.value for i in test_loss])

    test_acc = ea.scalars.Items('Testing/Testing_Acc')
    testing_acc += np.array([i.value for i in test_acc])

training_loss /= 5
training_acc /= 5
testing_loss /=5 
testing_acc /= 5
print(training_acc)
for epoch in range(EPOCH):
    writer.add_scalar('Training/Training_Loss', training_loss[epoch], epoch)
    writer.add_scalar('Training/Training_Acc', training_acc[epoch], epoch)
    writer.add_scalar('Testing/Testing_Loss', testing_loss[epoch], epoch)
    writer.add_scalar('Testing/Testing_Acc', testing_acc[epoch], epoch)
writer.close()


# %%
