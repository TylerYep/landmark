import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from models import Baseline, Sirius
from util import batch_GAP
import dataset
import const
import layers

from test import validation_set

def train(validate=False):
    K.clear_session()

    model = Baseline().model
    # model = Sirius().model

    if const.CONTINUE_TRAIN:
        model.load_weights(const.SAVE_PATH + 'dd_final.h5')
        print(model.summary())

    opt = Adam(lr=3e-4)
    loss = 'categorical_crossentropy' # get_custom_loss(1.0) or 'binary_crossentropy'

    def binary_crossentropy_n_cat(y_t, y_p):
        # This is just a reweighting to yield larger numbers for the loss.
        return keras.metrics.binary_crossentropy(y_t, y_p) * const.N_CAT

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[binary_crossentropy_n_cat, 'accuracy', batch_GAP])

    checkpoint1 = ModelCheckpoint(const.SAVE_PATH + 'checkpoint-1.h5', save_weights_only=True)
    checkpoint2 = ModelCheckpoint(const.SAVE_PATH + 'checkpoint-2.h5', save_weights_only=True)
    checkpoint3 = ModelCheckpoint(const.SAVE_PATH + 'checkpoint-3-best.h5',
                                  monitor='loss',
                                  save_best_only=True,
                                  save_weights_only=True)

    train_info, encoders = dataset.load_data(type='train')
    train_gen = dataset.get_image_gen(pd.concat([train_info]), encoders,
                                      eq_dist=False,
                                      n_ref_imgs=256,
                                      crop_prob=0.5,
                                      crop_p=0.5)
    if validate:
    	dev_set, encoders = dataset.load_data(type='dev')
    	dev_gen = dataset.get_image_gen(pd.concat([dev_set]), encoders,
                                    eq_dist=False,
                                    n_ref_imgs=256,
                                    crop_prob=0.5,
                                    crop_p=0.5)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=const.LOG_DIR)

    if validate:
        model.fit_generator(train_gen,
                        steps_per_epoch=len(train_info) / const.BATCH_SIZE / 8,
                        epochs=const.NUM_EPOCHS,
                        callbacks=[tensorboard_callback, checkpoint1, checkpoint2, checkpoint3],
                        validation_data=dev_gen, validation_steps=1)
    else:
        model.fit_generator(train_gen,
                        steps_per_epoch=len(train_info) / const.BATCH_SIZE / 8,
                        epochs=const.NUM_EPOCHS,
                        callbacks=[tensorboard_callback, checkpoint1, checkpoint2, checkpoint3])
    model.save_weights(const.SAVE_PATH + 'dd_final.h5')
    validation_set()
    # K.eval(gm_exp)

'''
#### Custom loss function

Individual losses are reweighted on each batch, but each output neuron will still always
see a binary cross-entropy loss. In other words, the learning rate is simply higher for
the most confident predictions.
'''
def get_custom_loss(rank_weight=1., epsilon=1.e-9):
    def custom_loss(y_t, y_p):
        losses = tf.reduce_sum(-y_t*tf.log(y_p+epsilon) - (1.-y_t)*tf.log(1.-y_p+epsilon), axis=-1)
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

if __name__ == '__main__':
    train(validate=True)
