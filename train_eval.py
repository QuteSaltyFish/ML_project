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

if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=1, type=int, help="The epoch to be tested")
    parser.add_argument("--gpu", default='1', type=str, help="choose which DEVICE U want to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    DataSet = dataloader.MyDataSet()
    train_data, test_data = DataSet.test_trian_split()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])


    criterian = t.nn.MSELoss()
    model = DnCNN(n_channels=8).to(DEVICE)
    # Test the train_loader
    model = load_model(model, args.epoch)
    model = model.eval()

    with t.no_grad():
        # Test the test_loader
        for batch_idx, [data, label] in enumerate(train_loader):
            data = data.to(DEVICE)
            out = model(data)
            # monitor the upper and lower boundary of output
            out_max = t.max(out)
            out_min = t.min(out)
            out = (out - out_min) / (out_max - out_min)
            DIR = 'result/train_result/epoch_{}'.format(args.epoch)
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            OUTPUT = t.cat([data, out], dim=3)
            tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save(DIR + '/idx_{}.jpg'.format(batch_idx))
