import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

from model import dataloader
from model.DnCNN import DnCNN
from model.func import save_model, eval_model_new_thread, eval_model

if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    # os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']

    DataSet = dataloader.MyDataSet()
    train_data, test_data = DataSet.test_trian_split()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    model = DnCNN(n_channels=8).to(DEVICE)

    # Multi GPU setting
    # model = t.nn.DataParallel(model,device_ids=[0,1])

    optimizer = t.optim.Adam(model.parameters())

    criterian = t.nn.MSELoss()

    # Test the train_loader
    model = model.train()
    multiprocess_idx = 2
    for epoch in range(EPOCH):
        for batch_idx, [data, label] in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            out = model(data)
            loss = criterian(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss)
        save_model(model, epoch)
        eval_model_new_thread(epoch, 1)
        # LZX pls using the following code instead
        # multiprocessing.Process(target=eval_model(epoch, '0'), args=(multiprocess_idx,))
        # multiprocess_idx += 1
    time_end = time.time()
    print('time cost', time_end-time_start)
