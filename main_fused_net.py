import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader_for_Unet import *
from model.DnCNN import DnCNN
from model import Resnet
from model import Conv3D_Net
from model.fused_U_net import FuseNet
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
    parser.add_argument("--name", default='Fused_Net', type=str,
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

        model = FuseNet(1).to(DEVICE)

        if args.epoch != 0:
            model = load_model(model, args.epoch)
        # Multi GPU setting
        # model = t.nn.DataParallel(model,device_ids=[0,1])

        # optimizer = t.optim.SGD(model.parameters(), lr=LR)
        optimizer = t.optim.Adam(model.parameters())

        # Test the train_loader
        for epoch in range(args.epoch, EPOCH):
            model = model.train()
            train_loss = 0
            correct = 0
            for batch_idx, [data, label, seg, voxel] in enumerate(train_loader):
                data, label, seg, voxel = data.to(DEVICE), label.to(DEVICE), seg.to(DEVICE), voxel.to(DEVICE)
                [class_pred, seg_pred] = model(data, voxel)
                seg_loss = t.nn.functional.smooth_l1_loss(seg_pred, seg)
                class_loss = t.nn.functional.cross_entropy(class_pred, label)
                loss = seg_loss + class_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                pred = class_pred.max(1, keepdim=True)[1]  # 找到概率最大的下标
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
                for batch_idx, [data, label, seg, voxel] in enumerate(test_loader):
                    data, label, seg, voxel = data.to(DEVICE), label.to(DEVICE), seg.to(DEVICE), voxel.to(DEVICE)
                    [class_pred, seg_pred] = model(data, voxel)
                    seg_loss = t.nn.functional.smooth_l1_loss(seg_pred, seg)
                    class_loss = t.nn.functional.cross_entropy(class_pred, label)
                    loss = seg_loss + class_loss

                    test_loss += loss
                    pred = class_pred.max(1, keepdim=True)[1]  # 找到概率最大的下标
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
