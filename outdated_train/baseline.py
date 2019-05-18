import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
import torch.optim as optim

from dataset import LandmarkDataset
from cnn_finetune import make_model
from train import train_model
from torchsummary import summary

def main():
    train_info = LandmarkDataset()
    dataloaders = {
        'train': data.DataLoader(train_info, batch_size=16, shuffle=True, num_workers=4), # TODO 32 on gpu
        'validation': data.DataLoader(train_info, batch_size=16, shuffle=False, num_workers=4)
    }

    model = make_model('xception', num_classes=n_cat)
    x = 0
    for param in model.parameters(): # 156
        if x > 86: break
        x += 1
        param.requires_grad = False

    # summary(model, input_size=(3, 299, 299))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

    model_trained = train_model(dataloaders, model, criterion, optimizer, num_epochs=3)

    # Save
    # torch.save(model_trained.state_dict(), 'models/pytorch/weights.h5')
    print(model_trained)


main()


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
