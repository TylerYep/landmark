import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
from baseline import load_data

class LandmarkDataset(data.Dataset):

    def __init__(self, data_path):
        super().__init__()
        dataset = load_data()
        self.X = torch.from_numpy(dataset['filename']).long()
        self.y = torch.from_numpy(dataset['landmark_id']).long()

    def __getitem__(self, index):
        load_images(info, input_shape)
        return self.X[index],

    def __len__(self):
        return self.X.shape[0]

    def load_images(info, input_shape):
        input_shape = tuple(input_shape)
        imgs = torch.zeros((len(info), input_shape[0], input_shape[1], 3))

        for i in range(len(info)):
            fname = info.iloc[i]['filename']
            try:
                img = cv2.cvtColor(
                      cv2.resize(cv2.imread(fname),input_shape),
                      cv2.COLOR_BGR2RGB)
            except:
                warnings.warn('Warning: could not read image: '+ fname +
                              '. Use black img instead.')
                img = torch.zeros((input_shape[0], input_shape[1], 3))
            imgs[i,:,:,:] = img

        return imgs


    def load_cropped_images(info, crop_p=0.2, crop='random'):
        new_res = torch.tensor([int(input_shape[0]*(1+crop_p)), int(input_shape[1]*(1+crop_p))])
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

        cropped_imgs = torch.zeros((len(info), input_shape[0], input_shape[1], 3))
        for ind in range(len(info)):
            cropped_imgs[ind,:,:,:] = raw_imgs[ind,
                                               cy0[ind]:cy1[ind],
                                               cx0[ind]:cx1[ind], :]
        return cropped_imgs
