import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model.dataloader_v3 import *
from model.DnCNN import DnCNN
from model import Resnet
from model import Conv3D_Net
from model.VoxNet_v2 import VoxNet
from model.baseline import FC_Net
from model.func import save_model, eval_model_new_thread, eval_model, load_model
import argparse
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
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

    test_set = In_the_wild_set()
    test_set.sort()
    test_loader = DataLoader.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    criterian = t.nn.NLLLoss()

    model1 = VoxNet(2).to(DEVICE)
    model2 = VoxNet(2).to(DEVICE)
    model3 = VoxNet(2).to(DEVICE)
    model4 = VoxNet(2).to(DEVICE)
    model5 = VoxNet(2).to(DEVICE)
    # Test the train_loader
    model1.load_state_dict(
        t.load("saved_model/VoxNet_V2_no_DA_1_folds/37.pkl"))
    model1.eval()

    model2.load_state_dict(
        t.load("saved_model/VoxNet_V2_no_DA_2_folds/56.pkl"))
    model2.eval()

    model3.load_state_dict(
        t.load("saved_model/VoxNet_V2_no_DA_3_folds/38.pkl"))
    model3.eval()

    model4.load_state_dict(
        t.load("saved_model/VoxNet_V2_no_DA_4_folds/55.pkl"))
    model4.eval()

    model5.load_state_dict(
        t.load("saved_model/VoxNet_V2_no_DA_5_folds/22.pkl"))
    model5.eval()
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
            out5 = t.nn.functional.softmax(model5(data))
            # out5 = model5(data)
            out = out1 + out2 + out3 + out4 +out5
            out /= 5
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
