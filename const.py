RUN_ON_GPU = False
import datetime
import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
np.random.seed(0)
tf.set_random_seed(0)
warnings.simplefilter('ignore')

''' File Paths '''
DATA_PATH = 'data/'
SAVE_PATH = 'save/'
TRAIN_PATH = DATA_PATH + 'images/train/'
DEV_PATH = TRAIN_PATH
TEST_PATH =  DATA_PATH + 'images/test/'
NON_LANDMARK_TRAIN_PATH = DATA_PATH + 'images/train-distractors/'
NON_LANDMARK_DEV_PATH = DATA_PATH + 'images/dev-distractors/'

TRAIN_CSV = DATA_PATH + 'train-subset.csv'
DEV_CSV = DATA_PATH + 'dev.csv'
TEST_CSV = DATA_PATH + 'test.csv'

BEST_SAVE_MODEL = SAVE_PATH + 'checkpoint-3-best.h5'

### CHANGE THESE SETTINGS ON LOCAL ###
if not RUN_ON_GPU:
    CONTINUE_TRAIN = False
    BATCH_SIZE = 16
    BATCH_SIZE_PREDICT = 16
    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(DEV_CSV)
    df = pd.concat([train_df, dev_df])
    N_CAT = df['landmark_id'].nunique()

##### GPU SETTINGS #####
else:
    CONTINUE_TRAIN = False
    BATCH_SIZE = 32
    BATCH_SIZE_PREDICT = 32
    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(DEV_CSV)
    test_df = pd.read_csv(TEST_CSV)
    df = pd.concat([train_df, dev_df, test_df])
    N_CAT = df['landmark_id'].nunique()


NUM_EPOCHS = 200
INPUT_SHAPE = (299, 299)

BASIC = True
LOG_DIR = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
