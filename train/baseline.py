import glob
import warnings
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from util import load_images, load_cropped_images

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from cnn_finetune import make_model
from train import train_model

data_path = 'data/'
train_path = 'data/images/train-subset/' # on macs, change backslash
dev_path = train_path

n_cat = 5 # classes examining

batch_size = 16
batch_size_predict = 16
input_shape = (299, 299)

train_image_files = glob.glob(train_path + '*.jpg')
train_image_ids = [image_file.replace('.jpg', '').replace(train_path, '') for image_file in train_image_files]
train_info_full = pd.read_csv(data_path + 'train-subset.csv', index_col='id')
train_info = train_info_full.loc[train_image_ids]
train_info['filename'] = pd.Series(train_image_files, index=train_image_ids)

# Heidi: commented out b/c our subset should not be missing any images
# train_info_correct = pd.read_csv('train_info_correct.csv', index_col='id')
# train_info = train_info[train_info['landmark_id'].isin(train_info_correct['landmark_id'])]

n_cat_train = train_info['landmark_id'].nunique()
if n_cat_train != n_cat:
    warnings.warn('Warning: The training data is not compatible.')

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=True, n_values=n_cat)

train_info['label'] = label_encoder.fit_transform(train_info['landmark_id'].values)
train_info['one_hot'] = one_hot_encoder.fit_transform(train_info['label'].values.reshape(-1, 1))


# TODO THESE NEED TO WORK FIRST
dataloaders = {
    'train': torch.utils.data.DataLoader(train_info, batch_size=32, shuffle=True, num_workers=4),
    'validation': torch.utils.data.DataLoader(train_info, batch_size=32, shuffle=False, num_workers=4)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = make_model('xception', num_classes=n_cat)
x = 0
for param in model.parameters():
    if x > 85: break
    x += 1
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model_trained = train_model(dataloaders, model, criterion, optimizer, num_epochs=3)

# Save
torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5')
print(model_trained)






# train_path = './train-highres/'
# non_landmark_train_path = './distractors/*/'
# dev_path = './dev/'
# non_landmark_dev_path = './distractors-dev/'
# test_path = './test-highres/'

# n_cat = 14942

# batch_size = 48
# batch_size_predict = 128
# input_shape = (299,299)

basic_version = True
if not basic_version:
    non_landmark_image_files = glob.glob(non_landmark_train_path + '*.jp*g')
    nlm_df = pd.DataFrame({'filename': non_landmark_image_files})
    nlm_df['landmark_id'] = -1

    dev_image_files = glob.glob(dev_path + '*.jpg')
    dev_image_ids = [image_file.replace('.jpg', '').replace(dev_path, '') \
                        for image_file in dev_image_files]
    dev_info = train_info_full.loc[dev_image_ids]
    dev_info['filename'] = pd.Series(dev_image_files, index=dev_image_ids)

    non_landmark_dev_image_files = glob.glob(non_landmark_dev_path + '*.jpg')
    nlm_dev_df = pd.DataFrame({'filename': non_landmark_dev_image_files})
    nlm_dev_df['landmark_id'] = -1

    test_info_full = pd.read_csv('test.csv', index_col='id')

    test_image_files = glob.glob(test_path + '*.jpg')
    test_image_ids = [image_file.replace('.jpg', '').replace(test_path, '') \
                        for image_file in test_image_files]

    test_info = test_info_full.loc[test_image_ids]
    test_info['filename'] = pd.Series(test_image_files, index=test_image_ids)
