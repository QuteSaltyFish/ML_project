'''
used to read the data from the data folder
'''
import torch as t
import torchvision as tv
from torchvision import transforms
import os
from PIL import Image
import json
import numpy as np


class data_set(t.utils.data.Dataset):
    def __init__(self, names):
        self.names = names
        self.config = json.load(open('config.json'))
        self.data_root = self.config["Taining_Dir"]
        self.label_root = self.config['Label_Path']
        self.init_transform()

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_root, self.names[index]))
        data['voxel'] = self.transform(data['voxel'].astype(np.float32))
        data['seg'] = data['seg'].astype(np.float32)
        return data

    def __len__(self):
        return len(self.names)


class MyDataSet():
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.data_root = self.config["Taining_Dir"]
        self.data_names = np.array(os.listdir(self.data_root))
        self.DEVICE = t.device(self.config["DEVICE"])
        self.gray = self.config["gray"]

    def test_trian_split(self, p=0.8):
        '''
        p is the portation of the training set
        '''
        length = len(self.data_names)

        # create a random array idx
        idx = np.array(range(length))
        np.random.shuffle(idx)
        print(idx[0])
        train_idx = idx[:(int)(length*p)]
        test_idx = idx[(int)(length*p):]
        print(self.data_names[[1, 2, 3, 4]])
        self.train_name = self.data_names[train_idx]
        self.test_name = self.data_names[test_idx]
        print(length)

        self.train_set = data_set(self.train_name)
        self.test_set = data_set(self.test_name)
        return self.train_set, self.test_set



class In_the_wild_set(t.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.test_root = self.config["Test_Dir"]
        self.test_names = os.listdir(self.test_root)
        self.DEVICE = t.device(self.config["DEVICE"])
        self.gray = self.config["gray"]
        self.init_transform()

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        data = np.load(os.path.join(self.test_root, self.test_names[index]))\
        return data

    def __len__(self):
        return len(self.test_names)


if __name__ == "__main__":

    DataSet = MyDataSet()
    train_set, test_set = DataSet.test_trian_split()
    wild = In_the_wild_set()
    print(len(train_set))
    print(len(test_set))
    print(train_set[0][0].shape)
    print(test_set[0][0].shape)
    print(wild[0][0].shape)
    # test_data = TestingData()
    # for i in range(len(train_data)):
    #     img, label = train_data[i]
    #     tv.transforms.ToPILImage()(img).save('result/input.jpg')
    #     tv.transforms.ToPILImage()(label).save('result/test.jpg')
