import glob
import warnings
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
import torch.optim as optim

from dataset import LandmarkDataset
from cnn_finetune import make_model
from train import train_model

n_cat = 5

def main():
    train_info = LandmarkDataset()
    dataloaders = {
        'train': data.DataLoader(train_info, batch_size=32, shuffle=True, num_workers=4),
        'validation': data.DataLoader(train_info, batch_size=32, shuffle=False, num_workers=4)
    }

    model = make_model('xception', num_classes=n_cat)
    x = 0
    for param in model.parameters(): # 156
        if x > 85: break
        x += 1
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model_trained = train_model(dataloaders, model, criterion, optimizer, num_epochs=3)

    # Save
    # torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5')
    print(model_trained)


main()

#
#
#
#
#
#
# # train_path = './train-highres/'
# # non_landmark_train_path = './distractors/*/'
# # dev_path = './dev/'
# # non_landmark_dev_path = './distractors-dev/'
# # test_path = './test-highres/'
#
# # n_cat = 14942
#
# # batch_size = 48
# # batch_size_predict = 128
# # input_shape = (299,299)
#
# basic_version = True
# if not basic_version:
#     non_landmark_image_files = glob.glob(non_landmark_train_path + '*.jp*g')
#     nlm_df = pd.DataFrame({'filename': non_landmark_image_files})
#     nlm_df['landmark_id'] = -1
#
#     dev_image_files = glob.glob(dev_path + '*.jpg')
#     dev_image_ids = [image_file.replace('.jpg', '').replace(dev_path, '') \
#                         for image_file in dev_image_files]
#     dev_info = train_info_full.loc[dev_image_ids]
#     dev_info['filename'] = pd.Series(dev_image_files, index=dev_image_ids)
#
#     non_landmark_dev_image_files = glob.glob(non_landmark_dev_path + '*.jpg')
#     nlm_dev_df = pd.DataFrame({'filename': non_landmark_dev_image_files})
#     nlm_dev_df['landmark_id'] = -1
#
#     test_info_full = pd.read_csv('test.csv', index_col='id')
#
#     test_image_files = glob.glob(test_path + '*.jpg')
#     test_image_ids = [image_file.replace('.jpg', '').replace(test_path, '') \
#                         for image_file in test_image_files]
#
#     test_info = test_info_full.loc[test_image_ids]
#     test_info['filename'] = pd.Series(test_image_files, index=test_image_ids)
