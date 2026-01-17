
#Dataset
CROP_SIZE = 256         
INPUT_CHANNELS = 28 
INPUT_SEQ_LEN = 3  
STRIDE_T=1

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 100 
DROPOUT = 0.3
INPUT_SHAPE = (BATCH_SIZE,INPUT_SEQ_LEN,INPUT_CHANNELS,CROP_SIZE,CROP_SIZE)
CLIPNORM =1.0
 
#Atention 
REDUCTION=4

# LOSS
GAMMA=2.0
ALPHA = 0.5
SMOOTH = 1e-5

#DIR
BASE_CACHE_DIR = 'saved/data_cache'
last_checkpoint_path = 'saved/checkpoints/last_model_convlstm7.weights.h5'
best_checkpoint_path = 'saved/checkpoints/best_fire_model7_convlstm.weights.h5'
history_path = 'training_history7.json'