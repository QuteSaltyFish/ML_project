'''
used to read the data from the data folder return 32*32*32
'''
import torch as t
import torchvision as tv
from torchvision import transforms
import os
from PIL import Image
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class data_set(t.utils.data.Dataset):
    def __init__(self, idx):
        self.idx = idx
        self.config = json.load(open('config.json'))
        self.data_root = self.config["Taining_Dir"]
        self.names = np.array(os.listdir(self.data_root))
        self.sort()
        self.names = self.names[idx]
        self.label_path = self.config['Label_Path']
        self.init_transform()
        self.read_label()

    def sort(self):
        d = self.names
        sorted_key_list = sorted(d, key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.names = np.array(sorted_key_list)
        # print(self.data_names)

    def read_label(self):
        dataframe = pd.read_csv(self.label_path)
        data = dataframe.values
        self.label = data[:,1][self.idx]

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # print(self.names[index].split('.')[0])
        data = np.load(os.path.join(self.data_root, self.names[index]))
        voxel = (self.transform(data['voxel'].astype(np.float32))).unsqueeze(0)/255
        seg =  self.transform(data['seg'].astype(np.float32)).unsqueeze(0)
        # print(t.max(voxel), t.max(seg))
        label = self.label[index]
        
        data = (voxel*seg)
        
        return [data, label]

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
        self.sort()

    def sort(self):
        d = self.data_names
        sorted_key_list = sorted(d, key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.data_names = np.array(sorted_key_list)
        # print(self.data_names)

    def test_trian_split(self, p=0.8):
        '''
        p is the portation of the training set
        '''
        length = len(self.data_names)

        # create a random array idx
        idx = np.array(range(length))
        np.random.shuffle(idx)
        self.train_idx = idx[:(int)(length*p)]
        self.test_idx = idx[(int)(length*p):]

        self.train_set = data_set(self.train_idx)
        self.test_set = data_set(self.test_idx)
        return self.train_set, self.test_set

    def __len__(self):
        return len(self.data_names)

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

    def sort(self):
        d = self.test_names
        sorted_key_list = sorted(d, key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.test_names = sorted_key_list

        # sorted_dict = map(lambda x:{x:(int)(os.path.splitext(x)[0].strip('candidate'))}, d)
        # print(sorted_dict)

    def __getitem__(self, index):
        data = np.load(os.path.join(self.test_root, self.test_names[index]))
        voxel = (self.transform(data['voxel'].astype(np.float32))).unsqueeze(0)/255
        seg =  self.transform(data['seg'].astype(np.float32)).unsqueeze(0)

        # label = self.label[index]
        
        data = (voxel*seg)
        
        name = os.path.basename(self.test_names[index])
        name = os.path.splitext(name)[0]
        return data, name

    def __len__(self):
        return len(self.test_names)


if __name__ == "__main__":

    # DataSet = MyDataSet()
    # train_set, test_set = DataSet.test_trian_split()
    # wild = In_the_wild_set()
    # print(len(train_set))
    # print(len(test_set))
    # print(train_set[0][0].shape)
    # print(test_set[0][0].shape)
    # print(wild[0].shape)
    # test_data = TestingData()
    # for i in range(len(train_data)):
    #     img, label = train_data[i]
    #     tv.transforms.ToPILImage()(img).save('result/input.jpg')
    #     tv.transforms.ToPILImage()(label).save('result/test.jpg')
    kf = KFold(n_splits=2)
    a = np.arange(100)
    print(kf.get_n_splits(a))
    for idx, [train_index, test_index] in enumerate(kf.split(a)):
        print(idx)
        print("TRAIN:", train_index, "TEST:", test_index)