import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader_v2 import *
from model.DnCNN import DnCNN
from model import Resnet_v2
from model import Conv3D_Net
from model.VoxNet import VoxNet
from model.baseline import FC_Net
from model.func import save_model, eval_model_new_thread, eval_model, load_model
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

if __name__ == "__main__":
    time_start = time.time()
    config = json.load(open("config.json"))
    # os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=0, type=int,
                        help="The epoch to be tested")
    parser.add_argument("--name", default='SimpleResnet_V2_main', type=str,
                        help="Whether to test after training")
    args = parser.parse_args()

    DataSet = MyDataSet()

    # using K-fold
    kf = KFold(n_splits=5)
    idx = np.arange(len(DataSet))
    print(kf.get_n_splits(idx))
    for K_idx, [train_idx, test_idx] in enumerate(kf.split(idx)):
        writer = SummaryWriter('runs/{}_{}_Fold'.format(args.name, K_idx+1))

        train_data, test_data = data_set(train_idx), data_set(test_idx)
        train_loader = DataLoader.DataLoader(
            train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
        test_loader = DataLoader.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

        model = Resnet_v2.test_net().to(DEVICE)

        if args.epoch != 0:
            model = load_model(model, args.epoch)
        # Multi GPU setting
        # model = t.nn.DataParallel(model,device_ids=[0,1])

        # optimizer = t.optim.SGD(model.parameters(), lr=LR)
        optimizer = t.optim.Adam(model.parameters())
        criterian = t.nn.CrossEntropyLoss().to(DEVICE)

        # Test the train_loader
        for epoch in range(args.epoch, EPOCH):
            model = model.train()
            train_loss = 0
            correct = 0
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
            writer.add_scalar('Training/Training_Loss', train_loss, epoch)
            writer.add_scalar('Training/Training_Acc', train_acc, epoch)
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
                writer.add_scalar('Testing/Testing_Loss', test_loss, epoch)
                writer.add_scalar('Testing/Testing_Acc', test_acc, epoch)
                print('Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    epoch, test_loss, correct, len(test_loader.dataset), test_acc))
            save_model(model, epoch, '{}_{}_folds'.format(args.name, K_idx+1))
            # eval_model_new_thread(epoch, 0)
            # LZX pls using the following code instead
            # multiprocessing.Process(target=eval_model(epoch, '0'), args=(multiprocess_idx,))
            # multiprocess_idx += 1
            writer.close()

    time_end = time.time()
    print('time cost', time_end-time_start)
