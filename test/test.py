#%%
import numpy as np
import torch as t 
import pandas as pd 
from tensorboardX import SummaryWriter

#%%
a = t.arange(10)
# data = t.randn(1000, 1000, 1000)
# data[a,a,a]
writer = SummaryWriter()
for i in range(10):
    writer.add_scalar('Training/Training_Loss', a[i], i)

writer.close()
# %%
import os
from tensorboard.backend.event_processing import event_accumulator
EPOCH=50
name = 'VoxNet_v1_0.0001'
training_loss = np.zeros(EPOCH, dtype=np.float)
testing_loss = np.zeros(EPOCH, dtype=np.float)
training_acc = np.zeros(EPOCH, dtype=np.float)
testing_acc = np.zeros(EPOCH, dtype=np.float)

dirs = ['runs/{}_{}_Fold'.format(name, i+1) for i in range(5)]
writer = SummaryWriter('runs/{}'.format(name))
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


# %%
