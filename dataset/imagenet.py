import os
import cv2
import pickle
import sys
import random
from io import BytesIO

import numpy as np
from PIL import Image

from torch.utils.data import Dataset

# -------------------------imageNet------------------------
class ImageNetFew(Dataset):
    def __init__(self, root, num_per_class=1000, transform=None):
        self.root = root

        self.transform = transform

        filename = 'imagenet-random-1-per-class.txt'#.format(num_per_class)

        file_path = os.path.join(self.root, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        self.metas = []
        for i, line in enumerate(lines):
            path = line.rstrip()
            # self.metas.append(path)
            self.metas.append((path, int(i)))

        self.root_dir = '/data/imagenet/train' 


    def __getitem__(self, index):
        filepath = os.path.join(self.root_dir, self.metas[index][0])
        with Image.open(filepath) as img:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, self.metas[index][1]

    def __len__(self):
        return len(self.metas)

class ImageNet(Dataset):
    def __init__(self, root, num_per_class=1000, transform=None):
        self.root = root

        self.transform = transform
        
        filename = 'val.txt' #'val.txt
        file_path = os.path.join(self.root, filename)

        # file_path = os.path.join(filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))

        self.root_dir = '/data/imagenet/val'

    def __getitem__(self, index):
        filepath = os.path.join(self.root_dir, self.metas[index][0])

        with Image.open(filepath) as img:
            img = img.convert('RGB')

        cls = self.metas[index][1]

        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape)
        return img, cls

    def __len__(self):
        return len(self.metas)

def categorize_imagenet(num_class):
    gt = 'images/meta/train.txt'
    data_by_class = [[] for i in range(num_class)]
    with open(gt,'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # print(line)
            path, label = line.strip().split()
            data_by_class[int(label)].append(path)
    # for i in range(num_class):
        # data_by_class[i] = np.vstack(data_by_class[i])

    return data_by_class

def extract_dataset_from_imagenet(num_per_class):
    num_class = 1000
    data_by_class = categorize_imagenet(num_class)

    random.seed(10)
    data_select = []
    for i in range(num_class):
        idxs = list(range(len(data_by_class[i])))
        random.shuffle(idxs)
        idx_select = idxs[:num_per_class]
        # print(idx_select)
        for idx in idx_select:
            # with Image.open(path_select) as img:
                # img = img.convert('RGB')
            data_select.append(data_by_class[i][idx])

    # data_select = np.vstack(data_select).reshape(-1, 3, 224, 224)
    # print(data_select.shape)
    with open('data/imagenet-random-{}-per-class.txt'.format(num_per_class), \
              'w') as f:
        for i, path in enumerate(data_select):
            f.write(path)
            if i != len(data_select)-1:
                f.write('\n')
        # pickle.dump(data_select, f)

if __name__ == '__main__':
    num_samples = list(range(1, 11)) + [20, 50]
    for i in num_samples:
        extract_dataset_from_cifar10(i)


