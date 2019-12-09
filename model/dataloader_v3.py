'''
used to read the data from the data folder return 32*32*32, add data argumentation
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
        self.load_data()
        self.load_label()

    def sort(self):
        d = self.names
        sorted_key_list = sorted(d, key=lambda x: (int)(
            os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.names = np.array(sorted_key_list)
        # print(self.data_names)

    def data_argumentation(self):
        old_len = len(self.names)
        new_label = []
        for idx in range(old_len):
            voxel = self.voxel[idx]
            seg = self.seg[idx]

            # use permute to change the point of view
            self.voxel.append(np.transpose(voxel, [0, 2, 1]))
            self.seg.append(np.transpose(seg, [0, 2, 1]))
            new_label.append(self.label[idx])

            # mirror the image
            # x axis
            self.voxel.append(voxel[..., ::-1].copy())
            self.seg.append(seg[..., ::-1].copy())
            new_label.append(self.label[idx])
            # y axis
            self.voxel.append(voxel[..., ::-1, :].copy())
            self.seg.append(seg[..., ::-1, :].copy())
            new_label.append(self.label[idx])

        # sum the labels
        self.label = np.concatenate([self.label, np.array(new_label)])
        print('data argumentation done.')

    def load_data(self):
        voxel = []
        seg = []
        for idx in range(len(self.names)):
            data = np.load(os.path.join(self.data_root, self.names[idx]))
            voxel.append((data['voxel'].astype(np.float32))/255)
            seg.append(data['seg'].astype(np.float32))
        self.voxel = voxel
        self.seg = seg

    def load_label(self):
        dataframe = pd.read_csv(self.label_path)
        data = dataframe.values
        self.label = data[:, 1][self.idx]

    def init_transform(self):
        """
        The preprocess of the img and label
        """
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        voxel = self.transform(self.voxel[index])
        seg = self.transform(self.seg[index])
        # label = self.label.astype(np.float32)[index]
        label = self.label[index]
        # data = np.expand_dims(seg, axis=0)
        data = (voxel*seg).unsqueeze(0).unsqueeze(0)
        data = t.nn.functional.interpolate(
            data, [32, 32, 32], mode='trilinear').squeeze(0)
        return data, label

    def __len__(self):
        return len(self.label)


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
        sorted_key_list = sorted(d, key=lambda x: (int)(
            os.path.splitext(x)[0].strip('candidate')))
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
        sorted_key_list = sorted(d, key=lambda x: (int)(
            os.path.splitext(x)[0].strip('candidate')))
        # sorted_key_list = sorted(d, key=lambda x:d[x], reverse=True)   倒序排列
        # print(sorted_key_list)
        self.test_names = sorted_key_list

        # sorted_dict = map(lambda x:{x:(int)(os.path.splitext(x)[0].strip('candidate'))}, d)
        # print(sorted_dict)

    def __getitem__(self, index):
        left_bound = 24
        right_bound = 56
        data = np.load(os.path.join(self.test_root, self.test_names[index]))
        voxel = self.transform(data['voxel'].astype(np.float32))/255
        seg = self.transform(data['seg'].astype(np.float32))
        data = (voxel*seg).unsqueeze(0).unsqueeze(0)
        data = t.nn.functional.interpolate(
            data, [32, 32, 32], mode='trilinear').squeeze(0)
        name = os.path.basename(self.test_names[index])
        name = os.path.splitext(name)[0]
        return data, name

    def __len__(self):
        return len(self.test_names)


if __name__ == "__main__":

    DataSet = MyDataSet()
    train_set, test_set = DataSet.test_trian_split()
    train_set.data_argumentation()
    print(train_set[0])
    print(len(train_set))
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
    # kf = KFold(n_splits=2)
    # a = np.arange(100)
    # print(kf.get_n_splits(a))
    # for idx, [train_index, test_index] in enumerate(kf.split(a)):
    #     print(idx)
    #     print("TRAIN:", train_index, "TEST:", test_index)
    # dataset = 
