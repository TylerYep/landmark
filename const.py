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

BEST_SAVE_MODEL = SAVE_PATH + 'dd_final.h5'

### CHANGE THESE SETTINGS ON LOCAL ###
if not RUN_ON_GPU:
    BATCH_SIZE = 16
    BATCH_SIZE_PREDICT = 16
    N_CAT = 100 # classes examining

##### GPU SETTINGS #####
else:
    BATCH_SIZE = 16
    BATCH_SIZE_PREDICT = 16
    df = pd.read_csv(TRAIN_CSV)
    N_CAT = df['landmark_id'].nunique()


NUM_EPOCHS = 100
INPUT_SHAPE = (299, 299)

BASIC = True
LOG_DIR = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


#####################
