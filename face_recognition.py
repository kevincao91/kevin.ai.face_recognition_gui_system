#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
import matplotlib.pyplot as plt

from config import opt
from my_class import MyDataset
import time
import cv2
from PIL import Image


class FaceRecognition:
    # 配置程序所有的设置数据
    def __init__(self):
        #  初始化程序的设置
        self.net = None
        self.net_best_margin = opt.net_best_margin
        self.net_input_img_size = opt.net_input_img_size
        self.database_loader = None
        self.database = None
        self.img_preprocessing = self.get_test_img_preprocessing()
        self.load_finish = False

    # 测试图像预处理变换
    def get_test_img_preprocessing(self):
        # 数据预处理设置
        normMean = [0.5, 0.5, 0.5]
        normStd = [0.5, 0.5, 0.5]
        normTransform = transforms.Normalize(normMean, normStd)

        testTransform = transforms.Compose([
            transforms.Resize(self.net_input_img_size),
            transforms.ToTensor(),
            normTransform
        ])
        return testTransform

    def load_database(self):
        # 数据预处理设置
        testTransform = self.img_preprocessing

        # 构建MyDataset实例
        import_data = MyDataset(split='PersonImageData', transform=testTransform)
        import_data_num = len(import_data)
        print('解析数据库，包含人员%d人' % import_data_num)

        # 构建DataLoder
        import_loader = DataLoader(dataset=import_data, batch_size=import_data_num)

        self.database_loader = import_loader

    def load_model(self):
        # 调用已有模型
        model = resnet18(pretrained=False)
        # 提取fc层中固定的参数
        fc_features = model.fc.in_features

        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        # 修改类别为128
        model.fc = nn.Linear(fc_features, 128)

        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        # 载入数据
        load_data = torch.load(opt.net_data_path)

        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        model.load_state_dict(load_data)

        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        model.eval()

        self.net = model

    def create_person_data(self):
        database = {}
        self.net.eval()
        it = iter(self.database_loader)
        images, labels = it.next()
        images = Variable(images)

        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        outputs = self.net(images)

        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        for i in range(len(labels)):
            database[labels[i]] = outputs[i].data.numpy()
            # print(labels[i], outputs[i].data.numpy())

        # print(database)
        self.database = database

    def who_is_it(self, test_img):
        """
        Implements face recognition for the happy house by finding who is the person on the image_path image.

        Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras

        Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
        """

        # START CODE HERE #
        # 适应网络输入尺寸, 预处理变换
        test_img = self.img_preprocessing(test_img)
        # print(np.shape(test_img))
        test_img = torch.unsqueeze(test_img, 0)
        # print(np.shape(test_img))
        # Step 1: Compute the target "encoding" for the image.
        images = Variable(test_img)
        self.net.eval()

        # Step 1: Compute the encoding for the image.
        encodings = self.net(images)
        encoding = torch.squeeze(encodings)
        # print(np.shape(encoding))
        encoding = encoding.data.numpy()

        # Step 2: Find the closest encoding #

        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 100
        identity = 'None'
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in self.database.items():

            # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
            dist = np.linalg.norm(encoding - db_enc)

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name

        # END CODE HERE #

        time_str = time.strftime('%m月%d日-%H时%M分%S秒')
        if min_dist > self.net_best_margin:
            string = time_str + ' : ' + '非数据库中人员'
            sys_string = '检测事件（监测到人员不在数据库当中，检测距离%0.2f）' % min_dist
        else:
            string = time_str + ' : ' + identity + '到访'
            sys_string = '检测事件（监测到人员%s，检测距离%0.2f）' % (identity, min_dist)
        print(opt.WARNING + string + opt.ENDC)
        opt.print_view_gui(string)
        opt.print_system_gui(sys_string)

    def login_main(self):

        string = '数据加载中，请稍后...'
        print(string)
        opt.print_system_gui(string)
        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        string = '正在加载人员数据'
        print(string)
        opt.print_system_gui(string)
        self.load_database()
        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        string = '正在加载深度神经网络'
        print(string)
        opt.print_system_gui(string)
        self.load_model()
        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        string = '正在编码人员数据'
        print(string)
        opt.print_system_gui(string)
        self.create_person_data()
        # 强制处理其他事物，防止GUI卡死
        QApplication.processEvents()

        string = '数据加载成功，可以开始监测。'
        print(string)
        opt.print_system_gui(string)

        opt.is_net_load_done = True


fr = FaceRecognition()

if __name__ == '__main__':
    fr.load_database()
    print('load data done !')

    fr.load_model()
    print('load net done !')

    fr.create_person_data()
    print('load person data done !')

    # test_img = cv2.imread('database/yiyi_0006.jpg')
    test_img = cv2.imread('300.bmp')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    # test_img = cv2.resize(test_img, (200, 200))
    test_img = Image.fromarray(test_img)
    # fn = 'database/yiyi_0006.jpg'
    # test_img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
    print(np.shape(test_img))
    fr.who_is_it(test_img)
