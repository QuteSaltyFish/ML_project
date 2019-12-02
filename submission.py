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
import pandas as pd 
if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=153, type=int, help="The epoch to be tested")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_set = dataloader.In_the_wild_set()
    test_set.sort()
    test_loader = DataLoader.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    
    criterian = t.nn.NLLLoss()

    model = Resnet.ResNet18().to(DEVICE)
    # Test the train_loader
    model = load_model(model, args.epoch)
    model = model.eval()

    with t.no_grad():
        # Test the test_loader
        test_loss = 0
        correct = 0
        idx = []
        Name = []
        Score = []
        
        for batch_idx, [data,name] in enumerate(test_loader):
            data= data.to(DEVICE)
            out = model(data)
            # monitor the upper and lower boundary of output
            # out_max = t.max(out)
            # out_min = t.min(out)
            # out = (out - out_min) / (out_max - out_min)
            out = t.exp(out).squeeze()
            Name.append(name[0])
            Score.append(out[1].item())
        test_dict = {'name':Name, 'Score':Score}
        test_dict_df = pd.DataFrame(test_dict)
        print(test_dict_df)
        test_dict_df.to_csv('result/Submission.csv', index=False)