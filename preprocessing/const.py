import numpy as np
import tensorflow as tf
np.random.seed(0)

# TODO combine this with the upper level const :/ Imports are hard...

# download-images.py
NUM_RAND_EXAMPLES = 10   # Don't use anymore (meant to take arbitrary numbers of images)
CURRENT_FILE_SET = 'train-subset'    # Change after with dev

# subset-data.py
N_MOST_FREQUENT_ELEMS = 100   # Changing this number requires deleting the data/images folder!
TAKE_N_OF_EACH = 50

# Check subset-data to see if this constant is being used
TRAIN_SIZE = 100
DEV_SIZE = 10

TRAIN_CSV = 'data/train-subset.csv'
DEV_CSV = 'data/dev.csv'
