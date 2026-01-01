import os
import sys
from tqdm import tqdm
import multiprocessing

# Añadir path para importar modulos del proyecto
sys.path.append(os.path.join(os.getcwd(), 'nn/PNL/model'))
import re
from vocabulary import Vocabulary
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def _clean_text( text,for_vocab=False):
        """Limpieza inicial de texto antes de pasar por spaCy."""
        # 1. Quitar HTML
        text = re.sub(r'<[^>]+>', ' ', text)
        
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
def _tokens_from_doc(doc, for_vocab=False):
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
def preprocess_files(files, is_source=True):
    """
    Preprocesa una lista de archivos y guarda el resultado en .tokenized
    """
    import spacy
    num_cores = 3
    nlp = spacy.load("es_core_news_sm", disable=['ner','lemmatizer','morphologizer','attribute_ruler'])
    nlp.add_pipe('sentencizer')
    for file_path, original_name in files:
        output_path = original_name + ".tokenized"
        if not os.path.isabs(output_path):
            output_path = os.path.join(DATA_DIR, output_path)
            
        print(f"Procesando {os.path.basename(file_path)} -> {os.path.basename(output_path)}")
        
        def line_generator(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield _clean_text(line, for_vocab=False)
        

        doc_stream = nlp.pipe(
            line_generator(file_path), 
            batch_size=200, 
            n_process=num_cores
        )

        with open(output_path, "w", encoding="utf-8") as f_out:
            for doc in tqdm(doc_stream):
                # Procesamos por oraciones para mantener la estructura
                sentences_tokens = []
                for sent in doc.sents:
                    tokens = _tokens_from_doc(sent, for_vocab=False)
                    if tokens:
                        sentences_tokens.append(" ".join(tokens))
                
                # Unimos las oraciones con "[.]" para que dataset.py las recupere de forma segura
                f_out.write(" [.] ".join(sentences_tokens) + "\n")

def main():
    print("=== INICIANDO PREPROCESAMIENTO ===")
    

    # 2. Definir archivos
    splits = ['train', 'val', 'test']
    src_files = []
    tgt_files = []
    
    for split in splits:
        s_name = f"{split}.txt.src"
        t_name = f"{split}.txt.tgt"
        s_path = os.path.join(DATA_DIR, s_name)
        t_path = os.path.join(DATA_DIR, t_name)
        
        if os.path.exists(s_path):
            src_files.append((s_path, s_name))
        if os.path.exists(t_path):
            tgt_files.append((t_path, t_name))
            
    # 3. Procesar
    if src_files:
        print(f"\nArchivos Source encontrados: {len(src_files)}")
        preprocess_files(src_files, is_source=True)
    
    if tgt_files:
        print(f"\nArchivos Target encontrados: {len(tgt_files)}")
        preprocess_files(tgt_files, is_source=False)
    
    print("\n=== PREPROCESAMIENTO COMPLETADO ===")

if __name__ == "__main__":
    main()
