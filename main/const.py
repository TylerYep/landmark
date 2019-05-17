
DATA_PATH = '../data/images/'

TRAIN_PATH = DATA_PATH + 'train-subset/'
NON_LANDMARK_TRAIN_PATH =  DATA_PATH + 'train-distractors/'

DEV_PATH =  DATA_PATH +  'dev/'
NON_LANDMARK_DEV_PATH =  DATA_PATH + 'dev-distractors/'

TEST_PATH =  DATA_PATH + 'test/'

N_CAT = 5 # classes examining

batch_size = 16
batch_size_predict = 16
input_shape = (299,299)



# n_cat = 14942
# batch_size = 48
# batch_size_predict = 128
# input_shape = (299,299)