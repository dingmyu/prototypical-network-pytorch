import os
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms



class MiniImageNet(Dataset):

    def __init__(self, root='', dataset='', mode='train'):
        data = []
        label = []

        self.root = root
        self.data = data
        self.label = label
        self.dataset = dataset
        self.mode = mode
        self._load_dataset()

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_dataset(self):
        path = self.root
        fw = open(os.path.join('dataset', self.dataset, self.mode+'.txt'))
        lines = fw.readlines()
        for line in lines:
            img_path = os.path.join(path, line.split()[0])
            labels = int(line.split()[1])
            self.data.append(img_path)
            self.label.append(labels)
        fw.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

