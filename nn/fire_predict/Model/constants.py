# Default Paths
DATA_PATHS = [] 

# Data Dimensions
SHAPES = (256,256)
INPUT_CHANNELS = 28 

# Model Architecture Defaults
HIDDEN_CHANNELS = 64  # Aumentado de 32 a 64 para mejor capacidad
DROPOUT = 0.3

# Training Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 100
PRED_SEQ_LEN = 1 
ACCUM_STEPS = 4
MAX_INPUT_SEQ_LEN = 5  # Reducido de 10 a 5 para agilizar entrenamiento CPU 

# Classification Thresholds
# Binary Classification: Only one threshold to distinguish Fire vs No Fire
FIRE_THRESHOLDS = [0] 
NUM_CLASSES = 1

# Custom Loss Penalties (Not used in standard Binary Cross Entropy, but useful for reference)
OVERESTIMATION_COST = 1.0 
UNDERESTIMATION_COST = 2.0 

# Binary Class Weights
# We will calculate a single pos_weight for BCEWithLogitsLoss
# CLASS_WEIGHTS can be set to None and calculated dynamically or set manually
CLASS_WEIGHTS = None