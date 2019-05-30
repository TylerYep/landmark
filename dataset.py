import os
import glob
import cv2
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import const

'''
#### Data preparation
Most of the code lines deal with missing images and the fact that
I had started with low resolution images and that the high resolution
image collection had different missing images compared to the low resolution collection.
Basically, the following lines load the dataframes provided by kaggle,
remove all missing images and add a field `filename` with a path to the downloaded jpg file.
There are 5 dataframes:
* train_info: train, landmark images
* nlm_df: train, non-landmark images
* dev_info: dev, landmark images
* nlm_dev_df: dev, non-landmark images
* test_info: test images
'''
def load_data(type='train'):
    """
    Returns pandas df of data plus encoders used (if not test)
    """
    train_info = pd.read_csv(const.TRAIN_CSV, index_col='id')
    dev_info = pd.read_csv(const.DEV_CSV, index_col='id')

    total_df = pd.concat([train_info, dev_info])
    label_encoder = LabelEncoder()
    label_encoder.fit(total_df['landmark_id'].values)
    one_hot_encoder = OneHotEncoder(sparse=True, n_values=const.N_CAT)

    if type in ('train', 'dev'):
        ### Takes all images from the data/train folder ###
        # train_image_files = [const.TRAIN_PATH + file for file in os.listdir(const.TRAIN_PATH) if file.endswith('.jpg')]
        # train_image_ids = [image_file.replace('.jpg', '').replace(const.TRAIN_PATH, '') for image_file in train_image_files]
        # train_info = train_info_full.loc[train_image_ids]
        info = train_info if type == 'train' else dev_info
        path = const.TRAIN_PATH if type == 'train' else const.DEV_PATH      # Equivalent

        image_ids = info.index.values
        image_files = [path + id + '.jpg' for id in image_ids]
        info['filename'] = pd.Series(image_files, index=image_ids)

        info['label'] = label_encoder.transform(info['landmark_id'].values)
        info['one_hot'] = one_hot_encoder.fit_transform(info['label'].values.reshape(-1, 1))
        return info, (label_encoder, one_hot_encoder)

    elif type == 'test':
        info = test_info
        path = const.TEST_PATH
        info = pd.read_csv(const.TEST_CSV, index_col='id')

        image_ids = info.index.values
        image_files = [path + id + '.jpg' for id in image_ids]
        info['filename'] = pd.Series(image_files, index=image_ids)

        return test_info, label_encoder



# ### Image i/o and image data augmentation
# Standard keras image augmentation is used and in addition random crops (with slighter additional augmentation) are scaled to full resolution. Since the original images have a higher resolution than this model, the crops will contain additional information.
def load_images(info, input_shape=const.INPUT_SHAPE):
    input_shape = tuple(input_shape)
    imgs = np.zeros((len(info), input_shape[0], input_shape[1], 3))

    for i in range(len(info)):
        fname = info.iloc[i]['filename']
        try:
            img = cv2.cvtColor(
                  cv2.resize(cv2.imread(fname),input_shape),
                  cv2.COLOR_BGR2RGB)
        except:
            warnings.warn('Warning: could not read image: '+ fname +
                          '. Use black img instead.')
            img = np.zeros((input_shape[0], input_shape[1], 3))
        imgs[i,:,:,:] = img
    return imgs


def load_cropped_images(info, crop_p=0.2, crop='random', input_shape=const.INPUT_SHAPE):
    new_res = np.array([int(input_shape[0]*(1+crop_p)), int(input_shape[1]*(1+crop_p))])
    if crop == 'random':
        cx0 = np.random.randint(new_res[0] - input_shape[0], size=len(info))
        cy0 = np.random.randint(new_res[1] - input_shape[1], size=len(info))
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
        cx0 = np.repeat(np.expand_dims(cx0, 0), len(info))
        cy0 = np.repeat(np.expand_dims(cy0, 0), len(info))

    cx1 = cx0 + input_shape[0]
    cy1 = cy0 + input_shape[1]

    raw_imgs = load_images(info, input_shape=tuple(new_res))

    cropped_imgs = np.zeros((len(info), input_shape[0], input_shape[1], 3))
    for ind in range(len(info)):
        cropped_imgs[ind,:,:,:] = raw_imgs[ind, cy0[ind]:cy1[ind], cx0[ind]:cx1[ind], :]
    return cropped_imgs


# Create the image data generator which is used for training
def get_image_gen(info_arg, encoders, shuffle=True, image_aug=True, eq_dist=False, n_ref_imgs=16,
                  crop_prob=0.5, crop_p=0.5):
    label_encoder, one_hot_encoder = encoders

    if image_aug:
        datagen = ImageDataGenerator(rotation_range=4.,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.5,
                                    channel_shift_range=25,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

        if crop_prob > 0:
            datagen_crop = ImageDataGenerator(rotation_range=4.,
                                            shear_range=0.2,
                                            zoom_range=0.1,
                                            channel_shift_range=20,
                                            horizontal_flip=True,
                                            fill_mode='nearest')

    count = len(info_arg)
    while True:
        if eq_dist:
            def sample(df):
                return df.sample(min(n_ref_imgs, len(df)))
            info = info_arg.groupby('landmark_id', group_keys=False).apply(sample)
        else:
            info = info_arg
        print('Generate', len(info), 'for the next round.')

        # shuffle data
        if shuffle and count >= len(info):
            print('Shuffling Data...')
            # info = info df.set_index(np.random.permutation(info.index)) # broken, supposedly
            info = info.iloc[np.random.permutation(np.arange(len(info)))]
            # info = info.sample(frac=1)
            count = 0

        # load images
        for ind in range(0, len(info), const.BATCH_SIZE):
            count += const.BATCH_SIZE

            y = info['landmark_id'].values[ind:(ind+const.BATCH_SIZE)]

            if np.random.rand() < crop_prob:
                imgs = load_cropped_images(info.iloc[ind:(ind+const.BATCH_SIZE)],
                                           crop_p=crop_p*np.random.rand() + 0.01,
                                           crop='random')
                if image_aug:
                    cflow = datagen_crop.flow(imgs, y, batch_size=imgs.shape[0], shuffle=True)
                    imgs, y = next(cflow)
            else:
                imgs = load_images(info.iloc[ind:(ind+const.BATCH_SIZE)])
                if image_aug:
                    cflow = datagen.flow(imgs, y, batch_size=imgs.shape[0], shuffle=True)
                    imgs, y = next(cflow)

            imgs = preprocess_input(imgs)

            y_l = label_encoder.transform(y[y>=0.])
            y_oh = np.zeros((len(y), const.N_CAT))
            y_oh[y >= 0., :] = one_hot_encoder.transform(y_l.reshape(-1,1)).todense()

            yield imgs, y_oh

    if const.BASIC:
        return pd.concat([train_info])
    else:
        return pd.concat([train_info, nlm_df])

if __name__ == '__main__':
    train_info, encoders = load_data(type='train')
    gen = get_image_gen(pd.concat([train_info]), encoders,
                                      eq_dist=False,
                                      n_ref_imgs=256,
                                      crop_prob=0.5,
                                      crop_p=0.5)
    for imgs, y_oh in gen:
        print(imgs[0], y_oh)
        break

