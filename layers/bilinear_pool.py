import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer

class CompactBilinearPooling(Layer):
    '''Compact Bilinear Pooling by MarcBS
    # Arguments:
        d: dimension of the compact bilinear feature
    # References:
        - [Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding](http://arxiv.org/pdf/1606.01847v2.pdf)
    '''

    def __init__(self, d, return_extra=False, **kwargs):
        self.h = [None, None]
        self.s = [None, None]
        self.return_extra = return_extra
        self.d = d

        # layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.supports_masking = True
        self.trainable = False
        self.uses_learning_phase = False
        self.input_spec = None  # compatible with whatever
        super(CompactBilinearPooling, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.trainable_weights = []
        self.nmodes = len(input_shapes)
        for i in range(self.nmodes):
            if self.h[i] is None:
                self.h[i] = np.random.random_integers(0, self.d-1, size=(input_shapes[i][1],))
                self.h[i] = K.variable(self.h[i], dtype='int64', name='h'+str(i))
            if self.s[i] is None:
                self.s[i] =  (np.floor(np.random.uniform(0, 2, size=(input_shapes[i][1],)))*2-1).astype('int64')
                self.s[i] = K.variable(self.s[i], dtype='int64', name='s'+str(i))
        self.non_trainable_weights = [self.h, self.s]

        self.built = True

    def compute_mask(self, input, input_mask=None):
        to_return = []
        if input_mask is None or not any([m is not None for m in input_mask]):
            to_return.append(None)
        else:
            to_return = input_mask[0]
        if self.return_extra:
            for i in range(self.nmodes):
                to_return += [None, None, None, None]
        return to_return + [None]

    def multimodal_compact_bilinear(self, x):
        v = [[]] * self.nmodes
        fft_v = [[]] * self.nmodes
        acum_fft = 1.0
        for i in range(self.nmodes):
            v[i] = K.count_sketch(self.h[i], self.s[i], x[i], self.d)
            fft_v[i] = K.fft(v[i])
            acum_fft *= fft_v[i]
        #acum_fft = K.concatenate((acum_fft[:, 1:, 0], acum_fft[:,1:,0][::-1]))
        out = K.cast(K.ifft(acum_fft), dtype='float32')
        if self.return_extra:
            return [out]+v+fft_v+[acum_fft]
        else:
            return out

    def call(self, x, mask=None):
        if type(x) is not list or len(x) <= 1:
            raise Exception('CompactBilinearPooling must be called on a list of tensors '
                            '(at least 2). Got: ' + str(x))
        y = self.multimodal_compact_bilinear(x)
        if self.return_extra:
            return y+self.h+self.s
        return y

    def get_config(self):
        config = {'d': self.d,
                  'h': self.h,
                  'return_extra': self.return_extra,
                  's': self.s}
        config = {'d': self.d}
        base_config = super(CompactBilinearPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_output_shape_for(self, input_shape):
        assert type(input_shape) is list  # must have mutiple input shape tuples
        shapes = []
        shapes.append(tuple([input_shape[0][0], self.d]))
        if self.return_extra:
            for s in input_shape: # v
                shapes.append(tuple([s[0], self.d]))
            for s in input_shape: # fft_v
                shapes.append(tuple([s[0], self.d]))
            shapes.append(tuple([s[0], self.d])) # acum_fft
            for i in range(self.nmodes): # h
                shapes.append(tuple([input_shape[i][1],1]))
            for i in range(self.nmodes): # v
                shapes.append(tuple([input_shape[i][1],1]))
        return shapes
