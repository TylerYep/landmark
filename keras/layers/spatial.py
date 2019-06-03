import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Layer, Lambda, Reshape
import const

class SpatialAttn(Layer):
    ''' Spatial Attention Layer '''
    def __init__(self):
        super().__init__()

    def call(self, x):
        # global cross-channel averaging, e.g. (32,2048,24,8)
        # NOTE: batch_size = None
        batch_size, _, h, w = x.shape
        x = K.mean(x, axis=1, keepdims=True)  # e.g. (32,1,24,8) # TODO TRY K.softmax(x, axis=1).expand_dims(dim=1)
        x = Reshape((-1,))(x) # e.g. (32,192)
        row_sum = K.sum(x, axis=0, keepdims=True)
        normalize = K.repeat_elements(row_sum, rep=const.BATCH_SIZE, axis=0)
        z = Lambda(lambda k: k[0] / k[1])([x, normalize])
        z = Reshape((1, h, w))(z)
        return z

if __name__ == '__main__':
    x = K.ones((32, 2048, 24, 8))
    model = SpatialAttn()
    result = model(x)
    print(K.get_value(result))
