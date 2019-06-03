#!/usr/bin/env python
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim
import torchvision

from sklearn.preprocessing import LabelEncoder
from tensorboardX import SummaryWriter
from cnn_finetune import make_model
from PIL import Image
from tqdm import tqdm

import const
from dataset import ImageDataset, load_data
from util import AverageMeter, GAP

def train(train_loader, model):
    global_start_time = time.time()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=const.LR_STEP, gamma=const.LR_FACTOR)
    tbx = SummaryWriter('save/')

    for epoch in range(1, const.NUM_EPOCHS + 1):
        print('-' * 50)
        print(f'Epoch {epoch}')
        batch_time, losses, avg_score = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        num_steps = min(len(train_loader), const.MAX_STEPS_PER_EPOCH)
        print(f'total batches: {num_steps}')
        end = time.time()
        lr_str = ''

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i, (input_, target) in enumerate(tqdm(train_loader)):
            if i >= num_steps:
                break
            input_ = input_.to(device)
            target = target.to(device)

            output = model(input_)
            loss = criterion(output, target)

            confs, predicts = torch.max(output.detach(), dim=1)
            avg_score.update(GAP(predicts, confs, target))

            losses.update(loss.data.item(), input_.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % const.PLT_FREQ == 0:
                tbx.add_scalar('train/loss', losses.val, (epoch-1)*num_steps+i)
                tbx.add_scalar('train/GAP', avg_score.val, (epoch-1)*num_steps+i)

            if i % const.LOG_FREQ == 0:
                print(f'{epoch} [{i}/{num_steps}]\t'
                            f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                            f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'
                            + lr_str)
                torch.save(model.state_dict(), 'save/weights_' + str((epoch-1)*num_steps+i) + '.pth')

        print(f' * average GAP on train {avg_score.avg:.4f}')
        lr_scheduler.step()


if __name__ == '__main__':
    train_loader, label_encoder, num_classes = load_data()
    np.save('label_encoder.npy', label_encoder.classes_)

    if const.CURR_MODEL == 'xception':
        model = make_model('xception', num_classes=num_classes)

    elif const.CURR_MODEL == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if const.RUN_ON_GPU:
        if const.CONTINUE_FROM is not None:
            model.load_state_dict(torch.load(const.CONTINUE_FROM))
        model.cuda()

    train(train_loader, model)
