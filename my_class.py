from torch.utils.data import Dataset
import torch
from config import opt
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, split='PersonImageData', transform=None, target_transform=None):
        img_dir = opt.database_dir
        txt_path = os.path.join(opt.database_dir, '{0}.txt'.format(split))
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # print(line)
            line = line.rstrip()
            words = line.split()
            img_file_path = os.path.join(img_dir, words[1])
            # print(imgA_file_path)
            imgs.append(img_file_path)
            labels.append(words[0])
            # print(words[0])

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        label = self.labels[index]
        img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)

