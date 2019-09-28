import os
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MiniImageNet(Dataset):

    def __init__(self, root='', mode='train'):
        data = []
        label = []

        self.root = root
        self.data = data
        self.label = label
        self.mode = mode
        self._load_dataset()


    def _load_dataset(self):
        fw = open(os.path.join(self.root, self.mode+'.txt'))
        lines = fw.readlines()
        for line in lines:
            line = line.strip().split()
            labels = int(line[-1])
            self.data.append(line)
            self.label.append(labels)
        fw.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = np.array([float(item) for item in path[:-1]]).astype(np.float32)
        return image, label

