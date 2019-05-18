import datetime
import keras
import keras.backend as K

from keras import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU, BatchNormalization, \
                         Activation, Conv2D, GlobalAveragePooling2D, Lambda
from keras.optimizers import Adam, RMSprop

from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

import dataset
import const

# TRAIN
def train():
    K.clear_session()
    x_model = Xception(input_shape=list(const.INPUT_SHAPE)+[3],
                       weights='imagenet',
                       include_top=False)

    for layer in x_model.layers:
        layer.trainable = True

    for layer in x_model.layers[:85]:
        layer.trainable = False

    # #### Generalized mean pool
    gm_exp = tf.Variable(3., dtype=tf.float32)
    def generalized_mean_pool_2d(X):
        return (tf.reduce_mean(tf.abs(X**(gm_exp)), axis=[1,2], keepdims=False)+1.e-8)**(1./gm_exp)

    X_feat = Input(x_model.output_shape[1:])
    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    X = lambda_layer(X_feat)
    X = Dropout(0.05)(X)
    X = Activation('relu')(X)
    X = Dense(const.N_CAT, activation='softmax')(X)

    top_model = Model(inputs=X_feat, outputs=X)

    X_image = Input(list(const.INPUT_SHAPE) + [3])
    X_f = x_model(X_image)
    X_f = top_model(X_f)
    model = Model(inputs=X_image, outputs=X_f)
    opt = Adam(lr=3e-4)
    loss = 'categorical_crossentropy' # get_custom_loss(1.0) or 'binary_crossentropy'
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[binary_crossentropy_n_cat, 'accuracy', batch_GAP])

    checkpoint1 = ModelCheckpoint(const.SAVE_PATH + 'checkpoint-1.h5', save_weights_only=True)
    checkpoint2 = ModelCheckpoint(const.SAVE_PATH + 'checkpoint-2.h5', save_weights_only=True)
    checkpoint3 = ModelCheckpoint(const.SAVE_PATH + 'checkpoint-3-best.h5',
                                  monitor='loss',
                                  save_best_only=True,
                                  save_weights_only=True)

    train_info, encoders = dataset.load_data(type="train")
    train_gen = dataset.get_image_gen(pd.concat([train_info]), encoders,
                              eq_dist=False,
                              n_ref_imgs=256,
                              crop_prob=0.5,
                              crop_p=0.5)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit_generator(train_gen,
                        steps_per_epoch=len(train_info) / const.BATCH_SIZE / 8,
                        epochs=const.NUM_EPOCHS,
                        callbacks=[tensorboard_callback, checkpoint1, checkpoint2, checkpoint3])
    model.save_weights(const.SAVE_PATH + 'dd_final.h5')
    K.eval(gm_exp)

'''
#### Custom loss function

Individual losses are reweighted on each batch, but each output neuron will still always
see a binary cross-entropy loss. In other words, the learning rate is simply higher for
the most confident predictions.
'''
def get_custom_loss(rank_weight=1., epsilon=1.e-9):
    def custom_loss(y_t, y_p):
        losses = tf.reduce_sum(-y_t*tf.log(y_p+epsilon) - (1.-y_t)*tf.log(1.-y_p+epsilon),
                               axis=-1)

        pred_idx = tf.argmax(y_p, axis=-1)

        mask = tf.one_hot(pred_idx,
                          depth=y_p.shape[1],
                          dtype=tf.bool,
                          on_value=True,
                          off_value=False)
        pred_cat = tf.boolean_mask(y_p, mask)
        y_t_cat = tf.boolean_mask(y_t, mask)

        n_pred = tf.shape(pred_cat)[0]
        _, ranks = tf.nn.top_k(pred_cat, k=n_pred)

        ranks = tf.cast(n_pred-ranks, tf.float32)/tf.cast(n_pred, tf.float32)*rank_weight
        rank_losses = ranks*(-y_t_cat*tf.log(pred_cat+epsilon)-(1.-y_t_cat)*tf.log(1.-pred_cat+epsilon))

        return rank_losses + losses
    return custom_loss


# #### Additional metric
# The GAP is estimated by calculating it on each batch during training.
def batch_GAP(y_t, y_p):
    pred_cat = tf.argmax(y_p, axis=-1)
    y_t_cat = tf.argmax(y_t, axis=-1) * tf.cast(
        tf.reduce_sum(y_t, axis=-1), tf.int64)

    n_pred = tf.shape(pred_cat)[0]
    is_c = tf.cast(tf.equal(pred_cat, y_t_cat), tf.float32)

    GAP = tf.reduce_mean(
          tf.cumsum(is_c) * is_c / tf.cast(
              tf.range(1, n_pred + 1),
              dtype=tf.float32))

    return GAP


# This is just a reweighting to yield larger numbers for the loss..
def binary_crossentropy_n_cat(y_t, y_p):
    return keras.metrics.binary_crossentropy(y_t, y_p) * const.N_CAT


if __name__ == "__main__":
    train()
