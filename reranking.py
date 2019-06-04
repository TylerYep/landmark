import torch
import torch.nn as nn

import const
from dataset import ImageDataset, load_data
from util import GAP
from test import inference, generate_submission

def rerank(predicts, confs, targets):
    pass

# def eval_reranking():
if __name__=="__main__":
    train_loader, dev_loader, label_encoder, nclasses = load_data()
    #initialize model
    if const.CURR_MODEL == 'xception':
        model = make_model('xception', num_classes=const.NUM_CLASSES)

    elif const.CURR_MODEL == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, const.NUM_CLASSES)

    if const.RUN_ON_GPU:
        model.load_state_dict(torch.load(const.CONTINUE_FROM))
        model.cuda()

    predicts_gpu, confs_gpu, targets_gpu = inference(dev_loader, model)

    gap_basic = GAP(predicts_gpu, confs_gpu, targets_gpu)
    print ("Unranked GAP: {}".format(gap))


    # predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()
