import warnings
warnings.simplefilter('ignore') # Tyler: Get rid of deprecation warnings

DATA_PATH = 'data/'

TRAIN_PATH = DATA_PATH + 'images/train-subset/'
NON_LANDMARK_TRAIN_PATH =  DATA_PATH + 'images/train-distractors/'

DEV_PATH =  DATA_PATH +  'images/dev/'
NON_LANDMARK_DEV_PATH =  DATA_PATH + 'images/dev-distractors/'

TEST_PATH =  DATA_PATH + 'images/test/'

N_CAT = 5 # classes examining

batch_size = 16
batch_size_predict = 16
INPUT_SHAPE = (299,299)

BASIC = True

# n_cat = 14942
# batch_size = 48
# batch_size_predict = 128
# input_shape = (299,299)