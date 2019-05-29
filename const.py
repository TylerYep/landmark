RUN_ON_GPU = False
import datetime
import warnings
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)
warnings.simplefilter('ignore')

### CHANGE THESE SETTINGS ON LOCAL ###
if not RUN_ON_GPU:
    N_CAT = 100 # classes examining

    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    BATCH_SIZE_PREDICT = 16
    INPUT_SHAPE = (299, 299)

    BASIC = True
    LOG_DIR = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    ''' File Paths '''
    DATA_PATH = 'data/'
    SAVE_PATH = 'save/'
    TRAIN_PATH = DATA_PATH + 'images/train-subset/'
    NON_LANDMARK_TRAIN_PATH = DATA_PATH + 'images/train-distractors/'
    DEV_PATH =  DATA_PATH + 'images/dev/'
    NON_LANDMARK_DEV_PATH = DATA_PATH + 'images/dev-distractors/'
    TEST_PATH =  DATA_PATH + 'images/test/'

    BEST_SAVE_MODEL = SAVE_PATH + 'dd_final.h5'


##### GPU SETTINGS #####
else:
    N_CAT = 203094 # classes examining

    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    BATCH_SIZE_PREDICT = 16
    INPUT_SHAPE = (299, 299)

    BASIC = True
    LOG_DIR = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    ''' File Paths '''
    DATA_PATH = 'data/'
    SAVE_PATH = 'save/'
    TRAIN_PATH = DATA_PATH + 'train/'
    TRAIN_CSV = DATA_PATH + 'train-subset.csv'
    NON_LANDMARK_TRAIN_PATH = DATA_PATH + 'train-distractors/'
    DEV_PATH =  DATA_PATH + 'dev/'
    NON_LANDMARK_DEV_PATH = DATA_PATH + 'dev-distractors/'
    TEST_PATH =  DATA_PATH + 'test/'

    BEST_SAVE_MODEL = SAVE_PATH + 'dd_final.h5'
