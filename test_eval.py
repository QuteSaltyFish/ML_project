import argparse
import json
import os
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import dataloader
from model.DnCNN import DnCNN
from model.func import load_model
from model import Resnet

if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=153, type=int, help="The epoch to be tested")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    DataSet = dataloader.MyDataSet()
    train_data, test_data = DataSet.test_trian_split()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    
    
    criterian = t.nn.NLLLoss()

    model = Resnet.ResNet18().to(DEVICE)
    # Test the train_loader
    model = load_model(model, args.epoch)
    model = model.eval()

    with t.no_grad():
        # Test the test_loader
        test_loss = 0
        correct = 0
        for batch_idx, [data,label] in enumerate(test_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            out = model(data)
            # monitor the upper and lower boundary of output
            # out_max = t.max(out)
            # out_min = t.min(out)
            # out = (out - out_min) / (out_max - out_min)
            test_loss += criterian(out, label)
            pred = out.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            # monitor the upper and lower boundary of output
            # out_max = t.max(out)
            # out_min = t.min(out)
            # out = (out - out_min) / (out_max - out_min)
            # DIR = 'result/test_result/epoch_{}'.format(args.epoch)
            # if not os.path.exists(DIR):
            #     os.makedirs(DIR)
            # OUTPUT = t.cat([data, out], dim=3)
            # tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save('good_output.jpg') 
            # tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save(DIR + '/idx_{}.jpg'.format(batch_idx))
