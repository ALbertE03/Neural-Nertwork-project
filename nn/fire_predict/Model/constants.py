# Default Paths
DEFAULT_DATA_X_PATH = './data_x.npy'
DEFAULT_DATA_Y_PATH = './data_y.npy'
CHECKPOINT_DIR = './checkpoints'

# Data Dimensions
IMG_HEIGHT = 64
IMG_WIDTH = 64
INPUT_CHANNELS = 12 # 12 spectral bands

# Model Architecture Defaults
HIDDEN_CHANNELS = 64
KERNEL_SIZE = (3, 3)

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
INPUT_SEQ_LEN = 5  # Past frames
PRED_SEQ_LEN = 1   # Future frames to predict
