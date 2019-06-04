import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model
import const
from layers import CompactBilinearPooling, SpatialAttn

class Xception(nn.Module):
    def __init__(self, n_classes=const.NUM_CLASSES):
        super().__init__()
        self.xception = make_model('xception', num_classes=n_classes, pretrained=True,
                                   pool=nn.AdaptiveMaxPool2d(1))
        print(self.xception)
        self.xception._classifier = None
        self.bilinearpool = CompactBilinearPooling(10, 10, 8192)
        self.spatial = SpatialAttn()

    def forward(self, input):
        b, h, w, c = input.shape    # in = (b, 3, 299, 299)
        x = self.xception(input)    # out=(b, 10, 10, 2048)
        print(x.shape)
        x = self.bilinearpool(x)    # out=(b, 8192)
        x = x.reshape((b, c, h, w)) # out=(b, 2048, 10, 10)
        x = self.spatial(x)         # out=(b, 1, 10, 10)
        return x

if __name__ == '__main__':
    x = torch.ones((32, 3, 299, 299))
    model = Xception()
    result = model(x)
    print(result)