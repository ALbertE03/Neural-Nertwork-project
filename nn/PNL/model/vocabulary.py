import os 
from collections import Counter
import json
import re
import spacy
from tqdm import tqdm
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
        try:
            spacy.prefer_gpu()
            self.nlp = spacy.load("es_core_news_sm", disable=['ner','lemmatizer','morphologizer','attribute_ruler'])
            self.nlp.add_pipe('sentencizer')
        except OSError:
            print("Descargando modelo de spaCy español (Large)...")
            spacy.prefer_gpu()
            spacy.cli.download("es_core_news_sm")
            self.nlp = spacy.load("es_core_news_sm", disable=['ner','lemmatizer','morphologizer','attribute_ruler'])
            self.nlp.add_pipe('sentencizer')
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
        
    def _clean_text(self, text,for_vocab=False):
        """Limpieza inicial de texto antes de pasar por spaCy."""
        
        # Atrapa http, https, www y los que empiezan con //
        url_pattern = r'(http[s]?://|www\.|//)[^\s/$.?#].[^\s]*'
        text = re.sub(url_pattern, ' ', text)
        # 3. Limpieza de caracteres especiales y ruido
        text = text.replace('\xa0', ' ')
        # Caracteres decorativos repetidos
        text = re.sub(r'[~*\-_=]{2,}', ' ', text)

        if for_vocab:
            # Quita números aislados: "1", "2025", "10.5", "50%"
            # Y también combinaciones numéricas con guión: "1-0", "24-7", "2023-2024"
            text = re.sub(r'\b\d+([.,-]\d+)*%?\b', ' ', text)
        
        text = text.replace('...', ' ')
        
        # 4. Normalizar espacios 
        return re.sub(r'\s+', ' ', text).strip()

    def _tokens_from_doc(self, doc, for_vocab=False):
        """Extrae y filtra tokens de un doc de spaCy."""
        tokens = []
        for token in doc:
            # Si estamos filtrando para el VOCABULARIO
            if for_vocab:
                # Omitimos Números y Fechas en el vocabulario fijo
                if token.like_num or token.pos_ == "NUM":
                    continue
                # Intento de detectar fechas por forma básica
                if re.match(r'\d+[/-]\d+', token.text):
                    continue
            
            # Filtros comunes (Puntuación ruidosa, brackets, quotes)
            if token.is_punct and token.text not in ['.', ',', '!', '?','¿']:
                continue
            if token.is_bracket or token.is_quote:
                continue
            
            t = token.text
            t = t.replace('``', '"').replace("''", '"')
            if t:
                tokens.append(t)
        return tokens

    def process_text(self, text):
        """Procesa un único texto para el modelo (mantiene fechas/números)."""
        text = self._clean_text(text,for_vocab=False)
        doc = self.nlp(text)
        return self._tokens_from_doc(doc, for_vocab=False)

    
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
        import multiprocessing
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Construyendo vocabulario usando {num_cores} núcleos...")
        print(f"Construyendo vocabulario a partir de los datos en: {self.data_dir}")
        src_files = [os.path.join(self.data_dir, f"{split}.txt.src") for split in ["train"]]
        tgt_files = [os.path.join(self.data_dir, f"{split}.txt.tgt") for split in ["train"]]
        all_files = src_files + tgt_files
        all_words = []
       
        word_counts = Counter()
       
        for file_path in all_files:
            
            def line_generator(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        # Aplicamos la limpieza básica de strings antes de spaCy
                        yield self._clean_text(line, for_vocab=True)
                        
            doc_stream = self.nlp.pipe(
                line_generator(file_path), 
                batch_size=1000, 
                n_process=num_cores
            )
            for doc in tqdm(doc_stream, desc=f"Procesando {os.path.basename(file_path)}"):
                tokens = self._tokens_from_doc(doc, for_vocab=True)
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

    def load_pretrained_embeddings(self, embedding_path, embedding_dim):
        """
        Carga embeddings pre-entrenados y los alinea con el vocabulario actual.
        Solo realiza coincidencias EXACTAS. Las palabras no encontradas (incluyendo
        variaciones de mayúsculas/minúsculas no presentes en el archivo) serán
        aprendidas por el modelo durante el entrenamiento.
        """
        import torch
        import numpy as np
        from tqdm import tqdm

        # 1. Verificar si el archivo existe, si no, descargar SBW (News) por defecto
        if embedding_path is not None and not os.path.exists(embedding_path):
            print(f"⚠ Archivo {embedding_path} no encontrado.")
            if "sbw_news.vec" in embedding_path:
                print("Iniciando descarga automática de SBW News Embeddings (Noticias en español)...")
                self.download_spanish_embeddings(os.path.dirname(embedding_path), type='sbw')
            elif "wiki.es.vec" in embedding_path:
                print("Iniciando descarga automática de FastText Spanish...")
                self.download_spanish_embeddings(os.path.dirname(embedding_path), type='fasttext')
            else:
                print("Usando inicialización aleatoria.")
                return torch.randn(len(self.id_to_word), embedding_dim) * 0.1

        vocab_size = len(self.id_to_word)
        # Inicialización aleatoria para que el modelo "aprenda" lo que no esté en los embeddings
        weights = torch.randn(vocab_size, embedding_dim) * 0.1
        
        if embedding_path is None:
            return weights

        print(f"Cargando embeddings (Solo Coincidencias Exactas) desde {embedding_path}...")
        
        found_indices = set()
        
        try:
            with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().split()
                if len(header) != 2:
                    f.seek(0)
                
                for line in tqdm(f, desc="Alineando embeddings"):
                    parts = line.rstrip().split(' ')
                    if len(parts) < embedding_dim + 1:
                        continue
                        
                    emb_word = parts[0]
                    
                    # Búsqueda Exacta ÚNICAMENTE
                    if emb_word in self.word_to_id:
                        idx = self.word_to_id[emb_word]
                        if idx not in found_indices:
                            vec = np.array([float(x) for x in parts[1:embedding_dim+1]])
                            weights[idx] = torch.from_numpy(vec)
                            found_indices.add(idx)
                                
            print(f"✓ Cobertura Exacta: {len(found_indices)} / {vocab_size} ({len(found_indices)/vocab_size*100:.1f}%)")
            print(f"  - Las {vocab_size - len(found_indices)} palabras restantes serán aprendidas desde cero.")

            if self.pad_token in self.word_to_id:
                weights[self.word_to_id[self.pad_token]] = torch.zeros(embedding_dim)
                
        except Exception as e:
            print(f"✗ Error cargando embeddings: {e}")
            
        return weights

    def download_spanish_embeddings(self, target_dir, type='sbw'):
        """
        Descarga y extrae vectores pre-entrenados.
        type: 'fasttext' (Wikipedia/CC) o 'sbw' (Spanish Billion Words - Noticias/Libros)
        """
        import urllib.request
        import os
        import gzip
        import shutil

        os.makedirs(target_dir, exist_ok=True)
        
        if type == 'sbw':
            # Spanish Billion Words (Más orientado a Noticias/Libros)
            url = "http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i2e.300d.vec.gz"
            target_name = "sbw_news.vec"
        else:
            # FastText CC Spanish
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz"
            target_name = "wiki.es.vec"

        gz_path = os.path.join(target_dir, f"{target_name}.gz")
        vec_path = os.path.join(target_dir, target_name)
        
        print(f"Descargando embeddings de tipo '{type}' desde {url}...")
        try:
            urllib.request.urlretrieve(url, gz_path)
            print(f"✓ Descarga completada. Descomprimiendo en {vec_path}...")
            
            with gzip.open(gz_path, 'rb') as f_in:
                with open(vec_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            print(f"✓ Extracción completada.")
            os.remove(gz_path) 
            return vec_path
        except Exception as e:
            print(f"✗ Error descargando: {e}")
            return None
        