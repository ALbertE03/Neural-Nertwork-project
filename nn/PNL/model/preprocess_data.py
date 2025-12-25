import os
import sys
from tqdm import tqdm
import multiprocessing

# Añadir path para importar modulos del proyecto
sys.path.append(os.path.join(os.getcwd(), 'nn/PNL/model'))

from vocabulary import Vocabulary
from constant import *

def preprocess_files(vocab, files, is_source=True):
    """
    Preprocesa una lista de archivos y guarda el resultado en .tokenized
    """
    # Determinar número de núcleos
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    for file_path, original_name in files:
        output_path = original_name + ".tokenized"
        # Si no es un path absoluto, le ponemos el DATA_DIR
        if not os.path.isabs(output_path):
            output_path = os.path.join(DATA_DIR, output_path)
            
        print(f"Procesando {os.path.basename(file_path)} -> {os.path.basename(output_path)}")
        
        # 1. Definir generador de líneas
        def line_generator(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    yield vocab._clean_text(line, for_vocab=False)
        
        # 2. Pipeline de spaCy
        doc_stream = vocab.nlp.pipe(
            line_generator(file_path), 
            batch_size=500, 
            n_process=num_cores
        )
        
        # 3. Guardar resultados
        with open(output_path, "w", encoding="utf-8") as f_out:
            for doc in tqdm(doc_stream, desc=f"  Tokenizando..."):
                # Procesamos por oraciones para mantener la estructura
                sentences_tokens = []
                for sent in doc.sents:
                    tokens = vocab._tokens_from_doc(sent, for_vocab=False)
                    if tokens:
                        sentences_tokens.append(" ".join(tokens))
                
                # Unimos las oraciones con "[.]" para que dataset.py las recupere de forma segura
                f_out.write(" [.] ".join(sentences_tokens) + "\n")

def main():
    print("=== INICIANDO PREPROCESAMIENTO ===")
    
    # 1. Cargar Vocabulario
    print("Cargando instancia de Vocabulary...")
    vocab = Vocabulary(
        CREATE_VOCABULARY=False,
        PAD_TOKEN=PAD_TOKEN,
        UNK_TOKEN=UNK_TOKEN,
        START_DECODING=START_DECODING,
        END_DECODING=END_DECODING,
        MAX_VOCAB_SIZE=MAX_VOCAB_SIZE,
        CHECKPOINT_VOCABULARY_DIR=CHECKPOINT_VOCABULARY_DIR,
        DATA_DIR=DATA_DIR,
        VOCAB_NAME=VOCAB_NAME
    )

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
        preprocess_files(vocab, src_files, is_source=True)
    
    if tgt_files:
        print(f"\nArchivos Target encontrados: {len(tgt_files)}")
        preprocess_files(vocab, tgt_files, is_source=False)
    
    print("\n=== PREPROCESAMIENTO COMPLETADO ===")

if __name__ == "__main__":
    main()
