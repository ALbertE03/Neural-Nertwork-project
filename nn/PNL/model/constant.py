
import torch
import time
import os

# =================================================
# DATA / VOCABULARY
# =================================================
# Rutas relativas al directorio del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VOCAB_NAME = "Vocabulary.json"
CHECKPOINT_VOCABULARY_DIR = os.path.join(BASE_DIR, "saved", "working")

MAX_VOCAB_SIZE = 50000
MAX_LEN_SRC = 400
MAX_LEN_TGT = 50
BATCH_SIZE = 32

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
START_DECODING = "[START]"
END_DECODING = "[END]"

CREATE_VOCABULARY = not os.path.exists(
    os.path.join(CHECKPOINT_VOCABULARY_DIR, VOCAB_NAME)
)

# =================================================
# MODEL ARCHITECTURE
# =================================================
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256

NUM_ENC_LAYERS = 1
NUM_DEC_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT_RATIO = 0.2

IS_ATTENTION = True
IS_PGEN = True
IS_COVERAGE = True
COV_LOSS_LAMBDA = 1.0

# =================================================
# DECODING
# =================================================
DECODING_STRATEGY = "beam_search"
BEAM_SIZE = 5


EPOCHS = 30
WARMUP_EPOCHS = 20
ITERS_PER_EPOCH = None  # None = procesar todo el dataset

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

LEARNER = "adam"
LEARNING_RATE = 1e-3

GRAD_CLIP = 1.0

SAVE_HISTORY = True
SAVE_MODEL_EPOCH = True

# =================================================
# GPU / REPRODUCIBILITY
# =================================================
USE_GPU = True
GPU_ID = 0
DEVICE = torch.device(f"cuda:{GPU_ID}" if USE_GPU and torch.cuda.is_available() else "cpu")

SEED = 42
REPRODUCIBILITY = True

# =================================================
# PATHS / LOGGING
# =================================================
CHECKPOINT_DIR = os.path.join(BASE_DIR, "saved")
GENERATED_TEXT_DIR = os.path.join(BASE_DIR, "generated")
PLOT = False
