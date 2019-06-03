import torch
import torch.nn as nn

class SpatialAttn(nn.Module):
    """ Spatial Attention Layer """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # global cross-channel averaging, e.g. (32,2048,24,8)
        batch_size, _, h, w = x.shape
        x = x.mean(dim=1, keepdim=True)  # e.g. (32,1,24,8)
        x = x.view(batch_size, -1)     # e.g. (32,192)
        x /= torch.sum(x, dim=0)
        x = x.view(batch_size, 1, h, w)
        return x


if __name__ == '__main__':
    x = torch.ones((32, 2048, 24, 8))
    model = SpatialAttn()
    result = model(x)
    print(result)