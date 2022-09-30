import torch

# Hyperparameters etc.
# True if you have only images in the data folder
IS_SEARCHING_FILES = False
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_EPOCHS = 1
NUM_WORKERS = 4
NUM_CLASSES = 3
PATCH_SIZE = 512
PIN_MEMORY = True
LOAD_MODEL = False
VALID_EVAL_STEP = 2
SAVING_CHECKPOINT = True
USING_AMP = True