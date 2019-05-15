import glob
import warnings
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np

n_cat = 5 # classes examining


def load_data():
    return pd.read_csv('data/train_info.csv')
    data_path = 'data/'
    train_path = 'data/images/train-subset/' # on macs, change backslash
    dev_path = train_path


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
    # one_hot_encoder = OneHotEncoder(sparse=True, n_values=n_cat)
    train_info['label'] = label_encoder.fit_transform(train_info['landmark_id'].values)
    # train_info['one_hot'] = one_hot_encoder.fit_transform(train_info['label'].values.reshape(-1, 1))
    train_info.to_csv('data/train_info.csv', index=None, header=True)
    return train_info



class LandmarkDataset(data.Dataset):

    def __init__(self): #, data_path):
        super().__init__()
        self.dataset = load_data()

    def __getitem__(self, index):
        img = self.load_image(index)
        y = self.dataset['label'].values[index]
        # y_oh = np.zeros((1, n_cat))
        # y_oh[y >= 0., :] = one_hot_encoder.transform(y_l.reshape(-1,1)).todense()
        # y_oh = y_oh.squeeze()
        return img, y

    def __len__(self):
        return self.dataset.shape[0]

    def load_image(self, index, input_shape=(299, 299)):
        fname = self.dataset.iloc[index]['filename']
        try:
            img = cv2.cvtColor(
                  cv2.resize(cv2.imread(fname), input_shape),
                  cv2.COLOR_BGR2RGB)
        except:
            warnings.warn('Warning: could not read image: ' + fname +
                          '. Using black img instead.')
            img = np.zeros((input_shape[0], input_shape[1], 3))

        img = img.reshape((3, input_shape[0], input_shape[1]))
        return torch.from_numpy(img).type('torch.FloatTensor')

    # TODO Should only load one image now
    # Does not work!
    def load_cropped_images(self, crop_p=0.2, crop='random'):
        new_res = torch.tensor([int(input_shape[0]*(1+crop_p)), int(input_shape[1]*(1+crop_p))])
        if crop == 'random':
            cx0 = np.random.randint(new_res[0] - input_shape[0], size=len(dataset))
            cy0 = np.random.randint(new_res[1] - input_shape[1], size=len(dataset))
        else:
            if crop == 'central':
                cx0, cy0 = (new_res - input_shape) // 2
            if crop == 'upper left':
                cx0, cy0 = 0, 0
            if crop == 'upper right':
                cx0, cy0 = new_res[1] - input_shape[1], 0
            if crop == 'lower left':
                cx0, cy0 = 0, new_res[0] - input_shape[0]
            if crop=='lower right':
                cx0, cy0 = new_res - input_shape
            cx0 = np.repeat(np.expand_dims(cx0, 0), len(dataset))
            cy0 = np.repeat(np.expand_dims(cy0, 0), len(dataset))

        cx1 = cx0 + input_shape[0]
        cy1 = cy0 + input_shape[1]

        raw_imgs = load_images(dataset, input_shape=tuple(new_res)) # TODO

        cropped_imgs = torch.zeros((len(dataset), input_shape[0], input_shape[1], 3))
        for ind in range(len(dataset)):
            cropped_imgs[ind,:,:,:] = raw_imgs[ind,
                                               cy0[ind]:cy1[ind],
                                               cx0[ind]:cx1[ind], :]
        return cropped_imgs
