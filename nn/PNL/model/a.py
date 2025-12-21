from vocabulary import Vocabulary
from constant import *
from config import Config
from dataset import PGNDataset, pgn_collate_fn
import torch
from torch.utils.data import DataLoader
from typing import List, Dict
import numpy as np

vocab = Vocabulary(
    CREATE_VOCABULARY=CREATE_VOCABULARY,
    PAD_TOKEN=PAD_TOKEN,
    UNK_TOKEN=UNK_TOKEN,
    START_DECODING=START_DECODING,
    END_DECODING=END_DECODING,
    MAX_VOCAB_SIZE=MAX_VOCAB_SIZE,
    CHECKPOINT_VOCABULARY_DIR=CHECKPOINT_VOCABULARY_DIR,
    DATA_DIR=DATA_DIR,
    VOCAB_NAME=VOCAB_NAME
)
vocab.build_vocabulary()

config = Config(
    max_vocab_size=vocab.total_size(),
    src_len=MAX_LEN_SRC,
    tgt_len=MAX_LEN_TGT,
    embedding_size=EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    use_gpu=USE_GPU,
    is_pgen=IS_PGEN,
    is_coverage=IS_COVERAGE,
    grad_clip=GRAD_CLIP,
    epochs=EPOCHS,
    data_path =DATA_DIR,
    generated_text_dir=GENERATED_TEXT_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    reproducibility=REPRODUCIBILITY,
    plot=PLOT,
    dropout_ratio=DROPOUT_RATIO,
    bidirectional=BIDIRECTIONAL,
    save_history=SAVE_HISTORY,
    save_model_epoch=SAVE_MODEL_EPOCH,
    seed =SEED,
    device =DEVICE,
    decoding_strategy=DECODING_STRATEGY,
    beam_size=BEAM_SIZE,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    learner=LEARNER,
    learning_rate=LEARNING_RATE,
    iters_per_epoch=ITERS_PER_EPOCH,
    gpu_id=GPU_ID,
)

print(config)  

train_dataset = PGNDataset(
    vocab=vocab,
    MAX_LEN_SRC=config['src_len'],
    MAX_LEN_TGT=config['tgt_len'],
    data_dir=config['data_path'],
    split='train' 
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=pgn_collate_fn
)

# Tamaño del vocabulario base
V_BASE = len(vocab.word_to_id)
def create_oov_id_to_word_map(oov_words_tensor: np.ndarray, V_base: int) -> Dict[int, str]:
    """Crea el mapeo ID a Palabra OOV a partir del tensor de palabras OOV."""
    oov_id_to_word: Dict[int, str] = {}
    
    # El primer ID OOV es V_base (igual que en dataset.py)
    oov_id = V_base
    
    # Decodificar y mapear
    for word_byte in oov_words_tensor:
        word = word_byte
        if word == '': # Ignorar el padding
            continue
        oov_id_to_word[oov_id] = word
        oov_id += 1
        
    return oov_id_to_word

def decode_sequence_to_text(id_sequence: np.ndarray, vocab: 'Vocabulary', oov_id_to_word: Dict[int, str]) -> List[str]:
    """Decodifica una secuencia de IDs (Base o Extendida) a texto."""
    V_base = len(vocab.word_to_id)
    decoded_words = []
    
    for id in id_sequence:
        id = int(id)
        # 0. Tokens especiales
        if id == vocab.word2id(vocab.pad_token):
             decoded_words.append(vocab.pad_token)
        elif id == vocab.word2id(vocab.start_decoding):
             decoded_words.append(vocab.start_decoding)
        elif id == vocab.word2id(vocab.end_decoding):
             decoded_words.append(vocab.end_decoding)
        # 1. IDs base del vocabulario
        elif id < V_base:
            decoded_words.append(vocab.id2word(id))
        # 2. IDs extendidos (OOV copiado)
        elif id in oov_id_to_word:
            decoded_words.append(f"*{oov_id_to_word[id]}*")
        else:
             # Debería ser un ID UNK si no es OOV mapeado
             decoded_words.append(vocab.unk_token)
             
    return decoded_words
# Tomar un batch del DataLoader
batch = next(iter(train_loader))

# ----------------------------------------
# Unpack
# ----------------------------------------
x_batch = batch
y_batch = batch["decoder_target"]

# --- 1. Mapeo OOV del primer ejemplo ---
oov_words_example = x_batch["oov_words"][0]  # List[str]
oov_id_to_word = create_oov_id_to_word_map(oov_words_example, V_BASE)

print(f"✅ Mapeo OOV generado del Batch (V_base={V_BASE}): {oov_id_to_word}")
print("\n--- Decodificación del primer ejemplo del batch ---")

# ----------------------------------------
# INPUTS
# ----------------------------------------
print("INPUTS (x_batch):")

# a. Encoder input (vocab base)
enc_input_ids = x_batch["encoder_input"][0].cpu().tolist()
print("a. encoder_input (IDs):", enc_input_ids[:30], "...")
print(f"   Texto (OOV → {vocab.unk_token}):")
print(
    "   ",
    " ".join(
        decode_sequence_to_text(
            enc_input_ids,
            vocab,
            {}
        )
    )
)

# b. Extended encoder input
ext_enc_input_ids = x_batch["extended_encoder_input"][0].cpu().tolist()
print("\nb. extended_encoder_input (IDs):", ext_enc_input_ids[:30], "...")
print("   Texto (OOV copiadas):")
print(
    "   ",
    " ".join(
        decode_sequence_to_text(
            ext_enc_input_ids,
            vocab,
            oov_id_to_word
        )
    )
)

# c. max_oov_len
print(f"\nc. max_oov_len: {x_batch['max_oov_len'][0].item()}")

# d. Decoder input
dec_input_ids = x_batch["decoder_input"][0].cpu().tolist()
print(f"d. decoder_input (IDs):", dec_input_ids[:30], "...")
print(f"   Texto (empieza con {vocab.start_decoding}):")
print(
    "   ",
    " ".join(
        decode_sequence_to_text(
            dec_input_ids,
            vocab,
            oov_id_to_word
        )
    )
)

print("-" * 50)

# ----------------------------------------
# TARGET
# ----------------------------------------
print("TARGETS (y_batch):")

dec_output_ids = y_batch[0].cpu().tolist()
print("decoder_output_pgn (IDs):", dec_output_ids[:30], "...")
print(f"Texto (termina con {vocab.end_decoding}):")
print(
    "   ",
    " ".join(
        decode_sequence_to_text(
            dec_output_ids,
            vocab,
            oov_id_to_word
        )
    )
)
