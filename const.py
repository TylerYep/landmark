RUN_ON_GPU = False
CONTINUE_FROM = 'save/weights_5.pth'
NUM_CLASSES = 6512
if RUN_ON_GPU:
    MIN_SAMPLES_PER_CLASS = 100
    BATCH_SIZE = 64
else:
    MIN_SAMPLES_PER_CLASS = 0
    BATCH_SIZE = 4

LEARNING_RATE = 1e-3
LR_STEP = 3
LR_FACTOR = 0.5
MAX_STEPS_PER_EPOCH = 15000
NUM_EPOCHS = 500
LOG_FREQ = 500
PLT_FREQ = 100
NUM_TOP_PREDICTS = 20

INPUT_SHAPE = (299, 299)
DATA_PATH = 'data/'
TRAIN_CSV = DATA_PATH + 'train-subset.csv'
DEV_CSV = DATA_PATH + 'dev.csv'
TEST_CSV = DATA_PATH + 'test.csv'