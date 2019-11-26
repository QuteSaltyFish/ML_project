import json
import os

import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import dataloader
from model.DnCNN import DnCNN


def save_model(model, epoch):
    dir = 'saved_model/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    t.save(model.state_dict(), dir + '{}.pkl'.format(epoch))


def load_model(model, epoch):
    path = 'saved_model/''{}.pkl'.format(epoch)
    model.load_state_dict(t.load(path))
    return model


def eval_model_new_thread(epoch, gpu):
    config = json.load(open("config.json"))
    path = 'result/nohup_result'
    if not os.path.exists(path):
        os.makedirs(path)
    python_path = config['python_path']
    os.system('nohup {} -u test_eval.py --epoch={} --gpu={} > {} 2>&1 &'.format(python_path, epoch, gpu,
                                                                                path + '/{}.out'.format(epoch)))
    os.system('nohup {} -u train_eval.py --epoch={} --gpu={} > {} 2>&1 &'.format(python_path, epoch, gpu,
                                                                                 path + '/{}.out'.format(epoch)))


def eval_model(epoch, gpu='0'):
    """
    evaluate the model using multi threading
    :param epoch: the model stored in the nth epoch
    :param gpu: which gpu U tried to use
    :return:
    """
    config = json.load(open('config.json'))
    DEVICE = config['DEVICE'] + ':' + gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    train_data = Mydataloader.TrainingData()
    train_loader = DataLoader.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    test_data = Mydataloader.TestingData()
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    criterian = t.nn.MSELoss()
    model = myNetwork.MyCNN(n_channels=8).to(DEVICE)
    # Test the train_loader
    model = load_model(model, epoch)
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
            DIR = 'result/train_result/epoch_{}'.format(epoch)
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            OUTPUT = t.cat([data, out], dim=3)
            tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save(
                DIR + '/idx_{}.jpg'.format(batch_idx))

    with t.no_grad():
        # Test the test_loader
        for batch_idx, data in enumerate(test_loader):
            data = data.to(DEVICE)
            out = model(data)
            # monitor the upper and lower boundary of output
            out_max = t.max(out)
            out_min = t.min(out)
            out = (out - out_min) / (out_max - out_min)
            DIR = 'result/test_result/epoch_{}'.format(epoch)
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            OUTPUT = t.cat([data, out], dim=3)
            tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save(
                DIR + '/idx_{}.jpg'.format(batch_idx))


if __name__ == '__main__':
    eval_model_new_thread(0, 1)
