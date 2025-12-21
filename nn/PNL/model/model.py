from vocabulary import Vocabulary
from constant import *
from config import Config
from dataset import PGNDataset, pgn_collate_fn
import torch
from torch.utils.data import DataLoader


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

train_dataset = PGNDataset(
    vocab=vocab,
    MAX_LEN_SRC=config['src_len'],
    MAX_LEN_TGT=config['tgt_len'],
    data_dir=config['data_path'],
    split='train' 
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['train_batch_size'],
    shuffle=False,
    collate_fn=pgn_collate_fn
)

val_dataset = PGNDataset(
    vocab=vocab,
    MAX_LEN_SRC=config['src_len'],
    MAX_LEN_TGT=config['tgt_len'],
    data_dir=config['data_path'],
    split='val' 
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['eval_batch_size'],
    shuffle=False,
    collate_fn=pgn_collate_fn
)
