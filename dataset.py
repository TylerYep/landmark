import os
import multiprocessing
from typing import Any, Optional, Tuple
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import LabelEncoder
import const

class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str):
        print(f'creating data loader - {mode}')
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        transforms_list = [transforms.Resize(const.INPUT_SHAPE)]
        if self.mode == 'train':
            transforms_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.RandomResizedCrop(const.INPUT_SHAPE[0]),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                            scale=(0.8, 1.2), shear=15,
                                            resample=Image.BILINEAR)
                ])
            ])

        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index):
        ''' Returns: tuple (sample, target) '''
        filename = self.df.id.values[index]

        part = 1 if self.mode == 'test' else 2
        sample = Image.open(f'data/images/{self.mode}/{filename}.jpg')
        while sample.mode != 'RGB':
            index += 1
            filename = self.df.id.values[index]
            sample = Image.open(f'data/images/{self.mode}/{filename}.jpg')

        image = self.transforms(sample)

        if self.mode == 'test':
            return image
        else:
            return image, self.df.landmark_id.values[index]

    def __len__(self):
        return self.df.shape[0]

def load_data() -> 'Tuple[DataLoader[np.ndarray], DataLoader[np.ndarray], LabelEncoder, int]':
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    # only use classes which have at least const.MIN_SAMPLES_PER_CLASS samples
    print('Loading data...')
    df = pd.read_csv(const.TRAIN_CSV)
    df.drop(columns='url', inplace=True)

    counts = df.landmark_id.value_counts()
    selected_classes = counts[counts >= const.MIN_SAMPLES_PER_CLASS].index
    num_classes = selected_classes.shape[0]
    print('Classes with at least N samples:', num_classes)

    train_df = df.loc[df.landmark_id.isin(selected_classes)].copy()
    print('train_df', train_df.shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df.landmark_id.values)
    print('Found classes', len(label_encoder.classes_))
    assert len(label_encoder.classes_) == num_classes

    train_df.landmark_id = label_encoder.transform(train_df.landmark_id)

    train_dataset = ImageDataset(train_df, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=const.BATCH_SIZE,
                              shuffle=True, num_workers=multiprocessing.cpu_count(), drop_last=True)

    ### DEV SET ###

    dev_df = pd.read_csv(const.DEV_CSV)
    dev_df.drop(columns='url', inplace=True)
    dev_df = dev_df.loc[dev_df.landmark_id.isin(selected_classes)].copy()
    print('dev_df', dev_df.shape)

    # filter non-existing test images
    exists = lambda img: os.path.exists(f'data/images/dev/{img}.jpg')
    dev_df = dev_df.loc[dev_df.id.apply(exists)].copy()
    print('dev_df after filtering', dev_df.shape)

    dev_dataset = ImageDataset(dev_df, mode='test')
    dev_loader = DataLoader(dev_dataset, batch_size=const.BATCH_SIZE,
                             shuffle=False, num_workers=multiprocessing.cpu_count())

    return train_loader, dev_loader, label_encoder, num_classes


def load_test_data() -> 'DataLoader[np.ndarray]':
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.benchmark = True

    # only use classes which have at least const.MIN_SAMPLES_PER_CLASS samples
    print('Loading data...')
    test_df = pd.read_csv(const.TEST_CSV, dtype=str)
    print('test_df', test_df.shape)

    # filter non-existing test images
    exists = lambda img: os.path.exists(f'data/images/test/{img}.jpg')
    test_df = test_df.loc[test_df.id.apply(exists)].copy()
    print('test_df after filtering', test_df.shape)

    test_dataset = ImageDataset(test_df, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=const.BATCH_SIZE,
                             shuffle=False, num_workers=multiprocessing.cpu_count())
    return test_loader
