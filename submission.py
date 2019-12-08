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
from model.VoxNet import VoxNet
import pandas as pd
if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=28, type=int,
                        help="The epoch to be tested")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_set = dataloader.In_the_wild_set()
    test_set.sort()
    test_loader = DataLoader.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    criterian = t.nn.NLLLoss()

    model1 = VoxNet(2).to(DEVICE)
    model2 = VoxNet(2).to(DEVICE)
    model3 = VoxNet(2).to(DEVICE)
    model4 = VoxNet(2).to(DEVICE)
    # model5 = VoxNet(2).to(DEVICE)
    # Test the train_loader
    model1.load_state_dict(
        t.load("saved_model/VoxNet_initialized_1_folds/9.pkl"))
    model1 = model1.eval()
    model2.load_state_dict(
        t.load("saved_model/VoxNet_initialized_2_folds/87.pkl"))
    model2 = model2.eval()
    model3.load_state_dict(
        t.load("saved_model/VoxNet_initialized_3_folds/64.pkl"))
    model3 = model3.eval()
    model4.load_state_dict(
        t.load("saved_model/VoxNet_initialized_4_folds/58.pkl"))
    model4 = model4.eval()
    # model5 = model5.load_state_dict(
    #     t.load("saved_model/VoxNet(150epoch)_5_folds/20.pkl"))
    # model5 = model5.eval()

    with t.no_grad():
        # Test the test_loader
        test_loss = 0
        correct = 0
        idx = []
        Name = []
        Score = []
        for batch_idx, [data, name] in enumerate(test_loader):
            data = data.to(DEVICE)
            out1 = t.nn.functional.softmax(model1(data))
            out2 = t.nn.functional.softmax(model2(data))
            out3 = t.nn.functional.softmax(model3(data))
            out4 = t.nn.functional.softmax(model4(data))
            
            # out5 = model5(data)
            out = out1 + out2 + out3 + out4#+out5
            out /= 4
            out = out.squeeze()
            # monitor the upper and lower boundary of output
            # out_max = t.max(out)
            # out_min = t.min(out)
            # out = (out - out_min) / (out_max - out_min)
            # out = t.exp(out).squeeze()
            Name.append(name[0])
            Score.append(out[1].item())
        test_dict = {'Id': Name, 'Predicted': Score}
        test_dict_df = pd.DataFrame(test_dict)
        print(test_dict_df)
        path = 'result'
        if not os.path.exists(path):
            os.makedirs(path)
        test_dict_df.to_csv('result/Submission.csv', index=False)
