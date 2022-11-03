import torch

# Hyperparameters etc.
# True if you have only images in the data folder
IS_SEARCHING_FILES = False
LEARNING_RATE = 1e-2
ADAM_EPSILON = 1e-2
WEIGHT_DECAY = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 56
NUM_EPOCHS = 300
NUM_WORKERS = 8
NUM_CLASSES = 3
PATCH_SIZE = 256
PIN_MEMORY = True
LOAD_MODEL = False
VALID_EVAL_STEP = 2
SAVING_CHECKPOINT = True
USING_AMP = True