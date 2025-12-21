import os
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
class PGNDataset(Dataset):
    """
    Dataset para PGN con OOVs dinámicos y head truncation
    """
    
    def __init__(self, vocab, MAX_LEN_SRC: int, MAX_LEN_TGT: int, data_dir: str, split: str):
        self.vocab = vocab
        self.MAX_LEN_SRC = MAX_LEN_SRC
        self.MAX_LEN_TGT = MAX_LEN_TGT
        self.data_dir = data_dir
        self.split = split

        self.PAD_ID = self.vocab.word2id(self.vocab.pad_token)
        self.SOS_ID = self.vocab.word2id(self.vocab.start_decoding)
        self.EOS_ID = self.vocab.word2id(self.vocab.end_decoding)
        self.UNK_ID = self.vocab.word2id(self.vocab.unk_token)

        src_path = os.path.join(data_dir, f"{split}.txt.src")
        tgt_path = os.path.join(data_dir, f"{split}.txt.tgt")
        
        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            raise FileNotFoundError(f"Split '{split}' no encontrado")

        with open(src_path, encoding="utf-8") as f:
            self.src_lines = f.readlines()

        with open(tgt_path, encoding="utf-8") as f:
            self.tgt_lines = f.readlines()

        assert len(self.src_lines) == len(self.tgt_lines)
    
    def _get_extended_src_ids(
        self, src_tokens_raw: List[str]
    ) -> Tuple[List[int], int, Dict[str, int], List[str]]:
        """Obtener IDs extendidos para fuente con OOVs"""
        extended_src_ids = []
        temp_oov_map = {}
        oov_words = []
        
        vocab_size = len(self.vocab.word_to_id)
        oov_id_counter = vocab_size  # Empezar después del vocabulario base
        
        for token in src_tokens_raw:
            base_id = self.vocab.word2id(token)
            
            if base_id == self.UNK_ID:
                if token not in temp_oov_map:
                    temp_oov_map[token] = oov_id_counter
                    oov_words.append(token)
                    oov_id_counter += 1
                extended_src_ids.append(temp_oov_map[token])
            else:
                extended_src_ids.append(base_id)
        
        extended_vocab_size = oov_id_counter
        return extended_src_ids, extended_vocab_size, temp_oov_map, oov_words
    
    def _map_target_to_extended_ids(self, tgt_tokens, oov_map):
        """Mapear tokens objetivo a IDs extendidos"""
        mapped_ids = []
        for token in tgt_tokens:
            base_id = self.vocab.word2id(token)
            if base_id == self.UNK_ID and token in oov_map:
                mapped_ids.append(oov_map[token])
            else:
                mapped_ids.append(base_id)
        return mapped_ids
    
    def _pad_sequence(self, ids, max_len):
        """Rellenar secuencia con PAD_ID"""
        if len(ids) < max_len:
            ids.extend([self.PAD_ID] * (max_len - len(ids)))
        return ids[:max_len]
    
    def __len__(self):
        return len(self.src_lines)
    
    def __getitem__(self, idx):
        src_line = self.src_lines[idx].strip()
        tgt_line = self.tgt_lines[idx].strip()
        
        # --- 1. Head truncation por oraciones ---
        raw_sentences = src_line.split(" . ")
        trimmed_src_tokens = []
        
        for sentence in raw_sentences:
            sentence_tokens = self.vocab.preprocess_text(sentence.strip())
            if not sentence_tokens:
                continue
            
            tokens_to_add = sentence_tokens + ['.']
            
            if len(trimmed_src_tokens) + len(tokens_to_add) > self.MAX_LEN_SRC:
                break
            
            trimmed_src_tokens.extend(tokens_to_add)
        
        if trimmed_src_tokens and trimmed_src_tokens[-1] == '.':
            trimmed_src_tokens = trimmed_src_tokens[:-1]
        
        # --- 2. Encoder con OOVs ---
        ext_src_ids, ext_vocab_size, oov_map, oov_words = \
            self._get_extended_src_ids(trimmed_src_tokens)
        
        max_oov_len = ext_vocab_size - len(self.vocab.word_to_id)
        
        # Extended encoder input (con IDs extendidos para pointer-generator)
        extended_encoder_input = self._pad_sequence(ext_src_ids.copy(), self.MAX_LEN_SRC)
        
        # Encoder input regular (convertir OOVs a UNK para embeddings)
        encoder_input = [
            i if i < len(self.vocab.word_to_id) else self.UNK_ID
            for i in extended_encoder_input
        ]
        
        # --- 3. Decoder ---
        tgt_tokens = self.vocab.preprocess_text(tgt_line)
        tgt_ext_ids = self._map_target_to_extended_ids(tgt_tokens, oov_map)
        
        MAX_RAW_TGT_LEN = self.MAX_LEN_TGT - 1
        tgt_ext_ids = tgt_ext_ids[:MAX_RAW_TGT_LEN]
        
        # Decoder input (convertir OOVs a UNK para embeddings)
        decoder_input_ids = [self.SOS_ID]
        for token_id in tgt_ext_ids:
            if token_id < len(self.vocab.word_to_id):
                decoder_input_ids.append(token_id)
            else:
                decoder_input_ids.append(self.UNK_ID)
        
        # Decoder target (mantener extended IDs para loss)
        decoder_output_ids = tgt_ext_ids + [self.EOS_ID]
        
        decoder_input = self._pad_sequence(decoder_input_ids, self.MAX_LEN_TGT)
        decoder_output = self._pad_sequence(decoder_output_ids, self.MAX_LEN_TGT)
        
        # --- 4. Información adicional ---
        encoder_length = len(trimmed_src_tokens)
        encoder_mask = [1] * encoder_length + [0] * (self.MAX_LEN_SRC - encoder_length)
        
        return {
            "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
            "extended_encoder_input": torch.tensor(extended_encoder_input, dtype=torch.long),
            "encoder_length": torch.tensor(encoder_length, dtype=torch.long),
            "encoder_mask": torch.tensor(encoder_mask, dtype=torch.bool),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "decoder_target": torch.tensor(decoder_output, dtype=torch.long),
            "max_oov_len": max_oov_len,
            "oov_words": oov_words
        }

def pgn_collate_fn(batch):
    """Función para combinar muestras en batches"""
    max_oov = max(x["max_oov_len"] for x in batch)
    
    def pad_oov(words):
        """Rellenar lista de OOVs con strings vacíos"""
        return words + [""] * (max_oov - len(words))
    
    return {
        "encoder_input": torch.stack([x["encoder_input"] for x in batch]),
        "extended_encoder_input": torch.stack([x["extended_encoder_input"] for x in batch]),
        "encoder_length": torch.stack([x["encoder_length"] for x in batch]),
        "encoder_mask": torch.stack([x["encoder_mask"] for x in batch]),
        "decoder_input": torch.stack([x["decoder_input"] for x in batch]),
        "decoder_target": torch.stack([x["decoder_target"] for x in batch]),
        "max_oov_len": torch.tensor([x["max_oov_len"] for x in batch], dtype=torch.long),
        "oov_words": [pad_oov(x["oov_words"]) for x in batch]
    }