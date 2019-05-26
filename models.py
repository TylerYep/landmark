import keras
from keras import backend as K
from keras.layers import Model, Dense, Dropout, Input, Activation, Lambda
from keras.applications.xception import Xception

class Baseline(keras.Model):
     def __init__(self, freeze_layers=85):
        super().__init__()
        x_model = Xception(input_shape=list(const.INPUT_SHAPE)+[3],
                           weights='imagenet',
                           include_top=False)
        for layer in x_model.layers:
            layer.trainable = True
        for layer in x_model.layers[:freeze_layers]:
            layer.trainable = False

        self.xception = x_model
        self.top_model = build_top_model(input_shape, output_shape, drop_prob=0.05)


    def call(self, inputs):
        x = self.xception(inputs)
        x = self.top_model(x)
        return x


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