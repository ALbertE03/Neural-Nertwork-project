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
        
        # Verificar si existen versiones tokenizadas
        src_tokenized_path = f"{split}.txt.src" + ".tokenized"
        tgt_tokenized_path = f"{split}.txt.tgt" + ".tokenized"
        
        self.is_tokenized = False
        
        if os.path.exists(src_tokenized_path) and os.path.exists(tgt_tokenized_path):
            print(f"✓ Usando archivos TOKENIZADOS para {split} (Carga optimizada en Kaggle)")
            src_path = src_tokenized_path
            tgt_path = tgt_tokenized_path
            self.is_tokenized = True
        else:
            print(f"⚠ Usando archivos RAW para {split} (Tokenización en tiempo real -> LENTO)")

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
            
        # --- 1. Head truncation por oraciones ---
        if self.is_tokenized:
            # Si ya está tokenizado, recuperamos las oraciones separando por "[.]"
            raw_sentences = src_line.split(" [.] ")
             
        trimmed_src_tokens = []
        
        for sentence in raw_sentences:
            if self.is_tokenized:
                # Fast path: Ya está tokenizado por palabras
                sentence_tokens = sentence.split()
                tokens_to_add = sentence_tokens
            else:
                tokens_to_add = sentence.strip().split()
            
            if len(trimmed_src_tokens) + len(tokens_to_add) > self.MAX_LEN_SRC:
                break
            
            trimmed_src_tokens.extend(tokens_to_add)
        
        # --- 2. Encoder con OOVs ---
        ext_src_ids, ext_vocab_size, oov_map, oov_words = \
            self._get_extended_src_ids(trimmed_src_tokens)
        
        max_oov_len = ext_vocab_size - len(self.vocab.word_to_id)
        
        # Extended encoder input (con IDs extendidos para pointer-generator)
        # Dynamic Batching
        # extended_encoder_input = self._pad_sequence(ext_src_ids.copy(), self.MAX_LEN_SRC)
        extended_encoder_input = ext_src_ids[:self.MAX_LEN_SRC]
        
        # Encoder input regular (convertir OOVs a UNK para embeddings)
        encoder_input = [
            i if i < len(self.vocab.word_to_id) else self.UNK_ID
            for i in extended_encoder_input
        ]
        
        # --- 3. Decoder ---
        if self.is_tokenized:
            # Asumimos que el target tokenizado está separado por espacios
            # Si hay separadores de oraciones [.] los tratamos como tokens o los ignoramos según el caso
            # Aquí simplemente hacemos split por espacios
            tgt_tokens = tgt_line.strip().split()
        else:
            # Fallback para texto crudo
            tgt_tokens = tgt_line.strip().split()
        
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
        
        # Dynamic Batching
        # decoder_input = self._pad_sequence(decoder_input_ids, self.MAX_LEN_TGT)
        # decoder_output = self._pad_sequence(decoder_output_ids, self.MAX_LEN_TGT)
        decoder_input = decoder_input_ids[:self.MAX_LEN_TGT]
        decoder_output = decoder_output_ids[:self.MAX_LEN_TGT]
        
        # --- 4. Información adicional ---
        encoder_length = len(trimmed_src_tokens)
        # Mask será dinámica en collate, aquí solo devolvemos el largo real
        # encoder_mask = [1] * encoder_length + [0] * (self.MAX_LEN_SRC - encoder_length)
        encoder_mask = [1] * encoder_length
        
        return {
            "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
            "extended_encoder_input": torch.tensor(extended_encoder_input, dtype=torch.long),
            "encoder_length": torch.tensor(encoder_length, dtype=torch.long),
            "encoder_mask": torch.tensor(encoder_mask, dtype=torch.bool),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "decoder_target": torch.tensor(decoder_output, dtype=torch.long),
            "max_oov_len": max_oov_len,
            "oov_words": oov_words,
            "pad_id": self.PAD_ID  
        }

def pgn_collate_fn(batch):
    """Función para combinar muestras en batches"""
    # Filtrar ejemplos con encoder_length <= 0
    filter_batch = []
    for x in batch:
        if x['encoder_length'].item() > 0:
            filter_batch.append(x)
    
    # Si todos los ejemplos fueron filtrados, retornar None
    if len(filter_batch) == 0:
        return None
    
    batch = filter_batch
    max_oov = max(x["max_oov_len"] for x in batch)
    
    # 1. Obtener lengths máximos del batch actual (Dynamic Batching)
    max_enc_len = max(x["encoder_length"].item() for x in batch)
    max_dec_len = max(len(x["decoder_input"]) for x in batch)
    
    pad_id = batch[0]["pad_id"]
    
    def pad_tensor(t, length, val):
        """Pad tensor to length with val"""
        return torch.cat([t, torch.full((length - len(t),), val, dtype=t.dtype)])

    def pad_oov(words):
        """Rellenar lista de OOVs con strings vacíos"""
        return words + [""] * (max_oov - len(words))
    
    return {
        # Encoders: Pad a max_enc_len
        "encoder_input": torch.stack([pad_tensor(x["encoder_input"], max_enc_len, pad_id) for x in batch]),
        "extended_encoder_input": torch.stack([pad_tensor(x["extended_encoder_input"], max_enc_len, pad_id) for x in batch]),
        "encoder_length": torch.stack([x["encoder_length"] for x in batch]),
        "encoder_mask": torch.stack([pad_tensor(x["encoder_mask"], max_enc_len, 0) for x in batch]), # 0 es False/Pad
        
        # Decoders: Pad a max_dec_len
        "decoder_input": torch.stack([pad_tensor(x["decoder_input"], max_dec_len, pad_id) for x in batch]),
        "decoder_target": torch.stack([pad_tensor(x["decoder_target"], max_dec_len, pad_id) for x in batch]),
        
        "max_oov_len": torch.tensor([x["max_oov_len"] for x in batch], dtype=torch.long),
        "oov_words": [pad_oov(x["oov_words"]) for x in batch]
    }