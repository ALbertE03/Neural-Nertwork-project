# Default Paths
DATA_PATHS = [] 

# Data Dimensions
SHAPES = (256,256)
INPUT_CHANNELS = 28 

# Model Architecture Defaults
HIDDEN_CHANNELS = 32
DROPOUT = 0.3

# Training Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 100
PRED_SEQ_LEN = 1 
ACCUM_STEPS = 4
MAX_INPUT_SEQ_LEN = 15  # Usar solo los últimos 15 pasos históricos    