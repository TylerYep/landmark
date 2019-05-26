import os
import numpy as np
import pandas as pd
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


def random_guess():
    np.random.seed(2019)
    train_df = pd.read_csv(os.path.join(const.DATA_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(const.DATA_PATH, 'test.csv'))
    submit_df = pd.read_csv(os.path.join(const.DATA_PATH, 'sample_submission.csv'))

    # take the most frequent label
    freq_label = train_df['landmark_id'].value_counts() / train_df['landmark_id'].value_counts().sum()

    submit_df['landmarks'] = '%d %2.2f' % (int(freq_label.index[0]), freq_label.values[0])
    # submit_df.to_csv('submission.csv', index=False)

    r_idx = lambda : np.random.choice(freq_label.index, p=freq_label.values)
    r_score = lambda idx: '%d %2.4f' % (freq_label.index[idx], freq_label.values[idx])

    submit_df['landmarks'] = submit_df.id.map(lambda _: r_score(r_idx()))
    submit_df.to_csv('train/rand_submission.csv', index=False)


def print_image(train_info, train_image_files, train_image_ids):
    import cv2
    import matplotlib.pyplot as plt
    print("Landmark_id of image", train_image_files[0], ":",
          train_info.loc[train_image_ids[0]]['landmark_id'])
    testimg = cv2.cvtColor(cv2.imread(np.random.choice(train_image_files)), cv2.COLOR_BGR2RGB)
    plt.imshow(testimg)


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
