import os
import numpy as np
import pandas as pd
import tensorflow as tf

import const

#### Additional metric
# The GAP is estimated by calculating it on each batch during training.
def batch_GAP(y_t, y_p):
    pred_cat = tf.argmax(y_p, axis=-1)
    y_t_cat = tf.argmax(y_t, axis=-1) * tf.cast(
        tf.reduce_sum(y_t, axis=-1), tf.int64)

    n_pred = tf.shape(pred_cat)[0]
    is_c = tf.cast(tf.equal(pred_cat, y_t_cat), tf.float32)

    GAP = tf.reduce_mean(tf.cumsum(is_c) * is_c / tf.cast(tf.range(1, n_pred + 1),
                         dtype=tf.float32))
    return GAP


def show_image(image_id):
    import cv2
    import matplotlib.pyplot as plt
    info = pd.read_csv(const.TRAIN_CSV, index_col='id')
    print("Landmark_id of image {} : {}".format(image_id, info.loc[image_id]['landmark_id']))
    img = cv2.cvtColor(cv2.imread(const.TRAIN_PATH + str(image_id) + '.jpg'), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def GAP_vector(pred, conf, true):
    '''
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition.
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    '''
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')

    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct

    gap = x.term.sum() / x.true.count()
    return gap
