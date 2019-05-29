import tensorflow as tf
import keras
from keras import Model
from keras.layers import Dense, Dropout, Input, Activation, Lambda
from keras.applications.xception import Xception
import const
import layers
from bilinearpool import compact_bilinear_pooling_layer

class Baseline():
    def __init__(self):
        x_model = build_xception_model()
        top_model = build_top_model(x_model.output_shape[1:], const.N_CAT)

        X_image = Input(list(const.INPUT_SHAPE) + [3])
        X_f = x_model(X_image)
        X_f = top_model(X_f)
        self.model = Model(inputs=X_image, outputs=X_f, name='Baseline')

class Sirius():
    def __init__(self):
        # same as baseline: Xception + top layer of xception
        x_model = build_xception_model()
        top_model = build_top_model(x_model.output_shape[1:], const.N_CAT)

        X_image = Input(list(const.INPUT_SHAPE) + [3])
        X_f = x_model(X_image)

        # bilinear pooling layer
        X_f = compact_bilinear_pooling(output_dim=8192)([X_f, X_f])

        # top layers for classification
        X_f = top_model(X_f)

        self.model = Model(inputs=X_image, outputs=X_pool, name='Sirius')



''' Model helper methods '''

def build_xception_model(freeze_layers=85):
    x_model = Xception(input_shape=list(const.INPUT_SHAPE)+[3],
                       weights='imagenet',
                       include_top=False)
    for layer in x_model.layers:
        layer.trainable = True
    for layer in x_model.layers[:freeze_layers]:
        layer.trainable = False
    return x_model

def build_top_model(input_shape, output_shape, drop_prob=0.05):
    #### Generalized mean pool
    gm_exp = tf.Variable(3., dtype=tf.float32)
    def generalized_mean_pool_2d(X):
        return (tf.reduce_mean(tf.abs(X**(gm_exp)), axis=[1,2], keepdims=False)+1.e-8)**(1./gm_exp)

    X_feat = Input(input_shape)
    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    X = lambda_layer(X_feat)
    X = Dropout(drop_prob)(X)
    X = Activation('relu')(X)
    X = Dense(output_shape, activation='softmax')(X)

    top_model = Model(inputs=X_feat, outputs=X)
    return top_model
