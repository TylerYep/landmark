import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Layer, Lambda

class SpatialAttn(Layer):
    ''' Spatial Attention Layer '''
    def __init__(self):
        super().__init__()

    def call(self, x):
        # global cross-channel averaging, e.g. (32,2048,24,8)
        batch_size, _, h, w = x.shape
        x = K.mean(x, axis=1, keepdims=True)  # e.g. (32,1,24,8)
        x = K.reshape(x, (batch_size, -1))    # e.g. (32,192)
        row_sum = K.sum(x, axis=0, keepdims=True)
        normalize = K.repeat_elements(row_sum, rep=batch_size, axis=0)
        z = Lambda(lambda k: k[0] / k[1])([x, normalize])
        z = K.reshape(z, (batch_size, 1, h, w))
        return z

if __name__ == '__main__':
    x = K.ones((32, 2048, 24, 8))
    model = SpatialAttn()
    result = model(x)
    print(K.get_value(result))
