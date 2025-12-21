import os 
from collections import Counter
import json
import re

class Vocabulary:
    def __init__(self, CREATE_VOCABULARY,
                 PAD_TOKEN, UNK_TOKEN,
                  END_DECODING, START_DECODING,
                 MAX_VOCAB_SIZE, CHECKPOINT_VOCABULARY_DIR, DATA_DIR,VOCAB_NAME):
        
        self.vocab_name = VOCAB_NAME
        self.create_vocabulary = CREATE_VOCABULARY
        self.checkpoint_vocab_dir = CHECKPOINT_VOCABULARY_DIR
        self.data_dir = DATA_DIR
        self.max_vocab_size = MAX_VOCAB_SIZE
        self._c = 0

        # Token de Relleno (Padding) - Usado para igualar longitudes de secuencias.
        self.pad_token = PAD_TOKEN
        # Token Desconocido (Unknown) - Usado para palabras no vistas en el vocabulario.
        self.unk_token = UNK_TOKEN
        
        # Tokens para Delimitación de Sentencias/Secuencias
        self.start_decoding = START_DECODING
        self.end_decoding = END_DECODING
        
   
        # Diccionario para mapear palabras a sus IDs (índices)     
        self.word_to_id = {}
        # Lista para mapear IDs a sus palabras
        self.id_to_word = []
        # Contador de frecuencia de palabras
        self.word_count = {}
        
        self._add_special_tokens()
        
    def total_size(self):
        return len(self.word_to_id)

    def word2id(self, word):
        """Retorna el id de la palabra o [UNK] id si es OOV."""
        if word not in self.word_to_id:
          return self.word_to_id[self.unk_token]
        return self.word_to_id[word]

    def id2word(self, word_id):
        """Retorna la palabra dado el id si existe en el vocabulario"""
        if 0 <= word_id < len(self.id_to_word):
            return self.id_to_word[word_id]
        
        raise ValueError('Id no esta en el vocab: %d' % word_id)
        
    def _add_special_tokens(self):
        """Añade los tokens especiales al vocabulario."""
        # Se añaden en un orden específico para que sus IDs sean fijos.
        special_tokens = [
            self.pad_token, self.unk_token, 
            self.start_decoding, self.end_decoding
        ]
        
        for token in special_tokens: #{'[PAD]':0,'[UNK]':1,'[START]':2,'[END]':3}
            if token not in self.word_to_id:
                self.word_to_id[token] = len(self.id_to_word)
                self.id_to_word.append(token)
                self.word_count[token] = 0 # Frecuencia inicial 0
        
        self.num_special_tokens = len(self.id_to_word)

    def _load_vocabulary(self):
        """
        Carga el vocabulario completo desde disco y restaura el estado interno.
        """
        try:
            vocab_path = os.path.join(self.checkpoint_vocab_dir, self.vocab_name)
    
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Vocabulario no encontrado en {vocab_path}")
            print(vocab_path)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            # -------------------------
            # Restaurar vocabulario base
            # -------------------------
                self.word_to_id = saved_data['word_to_id']
                self.id_to_word = saved_data['id_to_word']
                self.word_count = saved_data['word_count']
                self._c = saved_data['size']
        
                # -------------------------
                # Restaurar tokens especiales
                # -------------------------
                special_tokens = saved_data['special_tokens']
    
                self.pad_token = special_tokens['PAD']
                self.unk_token = special_tokens['UNK']
                self.start_decoding = special_tokens.get('START')
                self.end_decoding = special_tokens.get('END_DECODING')
        
                self.num_special_tokens = saved_data['metadata']['num_special_tokens']
        
                # -------------------------
                # Restaurar metadata
                # -------------------------
                metadata = saved_data['metadata']
        
                self.max_vocab_size = metadata['max_vocab_size']
                self.data_dir = metadata['data_dir']
                self.create_vocabulary = metadata['create_vocabulary']
                self.vocab_name = metadata['vocab_name']
                self.checkpoint_vocab_dir = metadata['checkpoint_dir']
    
            
            if len(self.word_to_id) != len(self.id_to_word):
                raise ValueError("Inconsistencia: word_to_id e id_to_word tienen tamaños distintos")
    
            if self.pad_token not in self.word_to_id:
                raise ValueError("Token PAD no encontrado en el vocabulario")
    
            print(f" Vocabulario cargado desde: {vocab_path}")
            print(f" Tamaño total: {len(self.word_to_id)}")
            print(f" Tokens especiales: {self.num_special_tokens}")
            print(f" Tokens regulares: {self._c}")
    
            return True
    
        except Exception as e:
            print(f"✗ Error cargando vocabulario: {e}")
            return False

            
    def size(self):
        """Retorna el tamaño real de el vocabulario"""
        return self._c    
        
    def preprocess_text(self,text):
        text = re.sub(r'\s+', ' ', text)
        return text.split()

    
    def _save_vocabulary(self):
        """Guarda el vocabulario completo en el disco."""
        try:
            # Crear directorio si no existe
            os.makedirs(self.checkpoint_vocab_dir, exist_ok=True)
            
            path = os.path.join(self.checkpoint_vocab_dir, self.vocab_name)
            
            # Preparar datos para guardar
            save_data = {
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'word_count': self.word_count,
                'size': self._c,
                'special_tokens': {
                    'PAD': self.pad_token,
                    'UNK': self.unk_token,
                    'START': self.start_decoding,
                    'END_DECODING': self.end_decoding
                },
                'metadata': {
                    'max_vocab_size': self.max_vocab_size,
                    'data_dir': self.data_dir,
                    'create_vocabulary': self.create_vocabulary,
                    'vocab_name': self.vocab_name,
                    'checkpoint_dir': self.checkpoint_vocab_dir,
                    'num_special_tokens': self.num_special_tokens,
                    'total_size': len(self.word_to_id)
                }
            }
            
            # Guardar como JSON
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=4)
            
            print(f"  Vocabulario guardado en: {path}")
            print(f"  Tamaño total: {len(self.word_to_id)} palabras")
            print(f"  Tokens especiales: {self.num_special_tokens}")
            print(f"  Tokens regulares: {self._c}")
                       
            return True
            
        except Exception as e:
            print(f"✗ Error al guardar el vocabulario: {e}")
            raise
            
    def _create_vocabulary(self):
        print(f"Construyendo vocabulario a partir de los datos en: {self.data_dir}")
        src_files = [os.path.join(self.data_dir, f"{split}.txt.src") for split in ["train"]]
        tgt_files = [os.path.join(self.data_dir, f"{split}.txt.tgt") for split in ["train"]]
        all_files = src_files + tgt_files
        all_words = []
       
        word_counts = Counter()
        for file_path in all_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = self.preprocess_text(line)
                    word_counts.update(tokens)
      
        # Calcular cuántas palabras regulares podemos añadir:
        if self.max_vocab_size <= self.num_special_tokens:
            raise ValueError(
                f"ERROR: MAX_VOCAB_SIZE ({self.max_vocab_size}) debe ser mayor que "
                f"el número de tokens especiales ({self.num_special_tokens}). "
                "Vocabulario muy pequeño."
            )
        limit = self.max_vocab_size - self.num_special_tokens
        # Seleccionar las 'limit' palabras más comunes, excluyendo las que ya son tokens especiales
        for word, count in word_counts.most_common(limit):
            if word not in self.word_to_id and len(self.word_to_id) < self.max_vocab_size:
                self.word_to_id[word] = len(self.id_to_word)
                self.id_to_word.append(word)
                self.word_count[word] = count
                self._c+=1
                
        # Guardar el vocabulario 
        self._save_vocabulary()

        print(f"Vocabulario construido. Tamaño final: {len(self.word_to_id)}")
        return True
        
    def build_vocabulary(self):
        if not self.create_vocabulary:
            return self._load_vocabulary()
        return self._create_vocabulary()
        