import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class DigitsDataset(Dataset):
    def __init__(self, args, data_path, channels, filename=None, train=True, transform=None):
        percent = args.percent
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            # self.images, self.labels = torch.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)))
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images, images], axis=0)
                            self.labels = np.concatenate([self.labels, labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]

            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)

        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

        self.images, self.labels = torch.from_numpy(self.images), torch.from_numpy(self.labels)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        if self.channels == 1:
            image = Image.fromarray(image.numpy(), mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image.numpy(), mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(base_path+'{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(base_path+'{}_test.pkl'.format(site), allow_pickle=True)

        for i in range(len(self.paths)):
            tmp = self.paths[i].split('/')[1:]
            self.paths[i] = '/'.join(tmp)

        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class Cifar10Dataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        x_all = np.load('./data/cifar10/{}.npy'.format(site))
        y_all = np.load('./data/cifar10/labels.npy')

        if train:
            self.images = x_all[0:40000, :, :]
            self.labels = y_all[0:40000,]
        else:
            self.images = x_all[40000:, :, :]
            self.labels = y_all[40000:,]

        # for i in range(len(self.paths)):
        #     tmp = self.paths[i].split('/')[1:]
        #     self.paths[i] = '/'.join(tmp)
        #
        # label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
        #               'windmill': 7, 'wine_glass': 8, 'zebra': 9}
        #
        # self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.channels = 3
        # self.base_path = base_path if base_path is not None else '../data'

        self.labels = self.labels.astype(np.long).squeeze()

        # self.images, self.labels = torch.from_numpy(self.images), torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        # image = Image.open(img_path)
        image = self.images[idx]

        image = Image.fromarray(image)

        # if len(image.split()) != 3:
        #     image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load(os.path.join(base_path, '{}_train.pkl'.format(site)), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load(os.path.join(base_path, '{}_test.pkl'.format(site)), allow_pickle=True)

        for i in range(len(self.paths)):
            tmp = self.paths[i].split('/')[1:]
            self.paths[i] = '/'.join(tmp)

        label_dict = {'bird': 0, 'feather': 1, 'headphones': 2, 'ice_cream': 3, 'teapot': 4, 'tiger': 5, 'whale': 6,
                      'windmill': 7, 'wine_glass': 8, 'zebra': 9}

        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
