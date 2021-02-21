import os
import random as rand
import numpy as np
import torch as t
from torch.utils import data
from PIL import Image
from torchvision import transforms as T
import utils

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class NYUDV2Dataset(data.Dataset):
    def __init__(self, imgs_ang_folder, imgs_master_folder, imgs_slave_folder, labels_folder, depths_folder, namelist_file):
        '''
        initialize the class
        :param imgs_folder (str):original images folder
        :param labels_folder (str):original labels folder
        :param namelist_flie (str):train list file for training, or test list file for testing
        '''
        f = open(namelist_file, "r")
        namelist = f.readlines()

        self.transform = transform

        self.imgs_ang = [os.path.join(imgs_ang_folder, '{0}.png'.format(file.strip())) for file in namelist]
        self.imgs_master = [os.path.join(imgs_master_folder, '{0}.png'.format(file.strip())) for file in namelist]
        self.imgs_slave = [os.path.join(imgs_slave_folder, '{0}.png'.format(file.strip())) for file in namelist]
        self.labels = [os.path.join(labels_folder, '{0}.png'.format(file.strip())) for file in namelist]
        self.depths = [os.path.join(depths_folder, '{0}.png'.format(file.strip())) for file in namelist]

    def __getitem__(self, index):

        img_ang_path = self.imgs_ang[index]
        img_master_path = self.imgs_master[index]
        img_slave_path = self.imgs_slave[index]
        label_path = self.labels[index]
        depth_path = self.depths[index]

        img_ang = Image.open(img_ang_path)
        img_master = Image.open(img_master_path)
        img_slave = Image.open(img_slave_path)
        label = Image.open(label_path)
        depth = Image.open(depth_path)

        #left = rand.randint(0, 480)
        #up = rand.randint(0, 320)

        #img = img.crop((left, up, left + 160, up + 160))
        #label = label.crop((left, up, left + 160, up + 160))
        #depth = depth.crop((left, up, left + 160, up + 160))

        img_ang = np.asarray(img_ang)
        img_master = np.asarray(img_master)
        img_slave = np.asarray(img_slave)
        label = np.asarray(label)
        depth = np.asarray(depth)
        #depth =depth / 255

        label = utils.make_one_hot2d(label[:, :, 1], 3)

        img_ang = self.transform(img_ang)
        img_master = self.transform(img_master)
        img_slave = self.transform(img_slave)

        return img_ang, img_master, img_slave, t.Tensor(label), t.Tensor(depth)

    def __len__(self):

        return len(self.imgs_ang)


class PredictINputDataset(data.Dataset):
    def __init__(self, imgs_ang_folder, imgs_master_folder, imgs_slave_folder,):

        self.transform = transform

        namelist = os.listdir(imgs_ang_folder)
        self.namelist = [name[:-4] for name in namelist]
        self.imgs_ang = [os.path.join(imgs_ang_folder, '{0}.png'.format(file.strip())) for file in self.namelist]
        self.imgs_master = [os.path.join(imgs_master_folder, '{0}.png'.format(file.strip())) for file in self.namelist]
        self.imgs_slave = [os.path.join(imgs_slave_folder, '{0}.png'.format(file.strip())) for file in self.namelist]

    def __getitem__(self, index):

        img_ang_path = self.imgs_ang[index]
        img_master_path = self.imgs_master[index]
        img_slave_path = self.imgs_slave[index]

        img_ang = Image.open(img_ang_path)
        img_master = Image.open(img_master_path)
        img_slave = Image.open(img_slave_path)

        img_ang = np.asarray(img_ang)
        img_master = np.asarray(img_master)
        img_slave = np.asarray(img_slave)

        img_ang = self.transform(img_ang)
        img_master = self.transform(img_master)
        img_slave = self.transform(img_slave)

        img_name = self.namelist[index]
        return img_ang, img_master, img_slave, img_name

    def __len__(self):

        return len(self.imgs_ang)


if __name__=="__main__":

    # dataSet = NYUDV2Dataset('data/nyu_images', 'data/nyu_labels40', 'data/nyu_depths', 'data/train.txt')
    # loader = data.DataLoader(dataSet, batch_size=1)
    #
    # for i, (img, label, depth) in enumerate(loader):
    #     if i > 1:
    #         break
    #     print(img.shape)
    #     print(label.shape)
    #     print(depth.shape)

    dataSet = PredictINputDataset('data/predict/images')
    loader = data.DataLoader(dataSet, batch_size=1)


    for i, img in enumerate(loader):
        print(img.shape)