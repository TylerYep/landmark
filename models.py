import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model
import const
from layers import CompactBilinearPooling, SpatialAttn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)

class CustomClassifier(nn.Module):
    def __init__(self, in_features=2048, num_classes=const.NUM_CLASSES):
        super(CustomClassifier, self).__init__()
    
        self.attn_dim = 16
        self.bilinearpool = CompactBilinearPooling(in_features, in_features, 8192) 
        #self.spatial = SpatialAttn()
        #self.linear = nn.Linear(self.attn_dim **2, num_classes)
        self.linear = nn.Linear(8192, num_classes)
    def forward(self, x):
        b, f = x.shape #(batch_size, 2048) from xception
        x = self.bilinearpool(x) #in: (b, 2048), out: (b, 8192)
        #x = x.view(b, -1, self.attn_dim, self.attn_dim) #out: (b, 8, 32, 32) 
        #x = self.spatial(x) #out: (b, 32*32) = (b, 1024)
        #x = x.view(b, -1) #out: (b*1024)
        x = self.linear(x) #out: (b)
        return x   
 
class Xception(nn.Module):
    def __init__(self, num_classes=const.NUM_CLASSES):
        super().__init__()
         
        def make_classifier(in_features, num_classes):
            return CustomClassifier(in_features, num_classes)
        
        self.xception = make_model('xception', num_classes=num_classes, pretrained=True,
                                   pool=nn.AdaptiveMaxPool2d(1), classifier_factory=make_classifier)
        #print(self.xception)
        #self.xception._classifier = None
        #self.bilinearpool = CompactBilinearPooling(10, 10, 8192)
        #self.spatial = SpatialAttn()
        #self.classifier = nn.Linear(100, n_classes)

    def forward(self, input):
        b, h, w, c = input.shape    # in = (b, 3, 299, 299)
        x = self.xception(input)    # out=(b, 10, 10, 2048)
        #print(x.shape)
        #x = self.bilinearpool(x)    # out=(b, 8192)
        #x = x.reshape((b, c, h, w)) # out=(b, 2048, 10, 10)
        #x = self.spatial(x)         # out=(b, 1, 10, 10)
        #x = x.reshape((b, -1)) #out=(b, 100) #flatten for fc
        #x = self.classifier(x)
        return x

if __name__ == '__main__':
    x = torch.ones((32, 3, 299, 299))
    model = Xception()
    result = model(x)
    print(result)
