import os
import json
import sys

# Añadir path para importar modulos del proyecto
sys.path.append(os.path.join(os.getcwd(), 'nn/PNL/model'))

from constant import EMBEDDING_PATH, CHECKPOINT_VOCABULARY_DIR, VOCAB_NAME

def analyze_missing_words():
    vocab_path = os.path.join(CHECKPOINT_VOCABULARY_DIR, VOCAB_NAME)
    
    if not os.path.exists(vocab_path):
        print(f"✗ No se encontró el vocabulario en {vocab_path}. Asegúrate de haberlo construido antes.")
        return

    print(f"Cargando vocabulario desde {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    word_to_id = data['word_to_id']
    id_to_word = data['id_to_word']
    word_count = data['word_count']
    
    if not os.path.exists(EMBEDDING_PATH):
        print(f"✗ No se encontró el archivo de embeddings en {EMBEDDING_PATH}")
        return

    print(f"Leyendo archivo de embeddings para identificar palabras presentes...")
    words_in_embeddings = set()
    try:
        with open(EMBEDDING_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            # Saltar header si existe
            header = f.readline().split()
            if len(header) != 2:
                f.seek(0)
            
            for line in f:
                parts = line.rstrip().split(' ')
                words_in_embeddings.add(parts[0])
    except Exception as e:
        print(f"✗ Error leyendo embeddings: {e}")
        return

    missing_words = []
    found_count = 0
    
    for word in id_to_word:
        # Ignorar tokens especiales
        if word.startswith('[') and word.endswith(']'):
            continue
            
        if word in words_in_embeddings:
            found_count += 1
        else:
            count = word_count.get(word, 0)
            missing_words.append((word, count))

    # Ordenar palabras faltantes por frecuencia
    missing_words.sort(key=lambda x: x[1], reverse=True)

    total_vocab = len(id_to_word) - 4 # Descontar tokens especiales
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE COBERTURA")
    print(f"{'='*60}")
    print(f"Total palabras (sin especiales): {total_vocab}")
    print(f"Palabras encontradas:            {found_count} ({found_count/total_vocab*100:.2f}%)")
    print(f"Palabras FALTANTES:              {len(missing_words)} ({len(missing_words)/total_vocab*100:.2f}%)")
    print(f"{'='*60}")
    
    print("\nTOP 50 PALABRAS MÁS FRECUENTES FALTANTES EN EMBEDDINGS:")
    print(f"{'Palabra':<30} | {'Frecuencia':<10}")
    print("-" * 45)
    for word, count in missing_words[:50]:
        print(f"{word:<30} | {count:<10}")

    # Análisis de causas comunes
    casing_issues = sum(1 for w, c in missing_words if w.lower() in words_in_embeddings and w != w.lower())
    print(f"\nPosibles mejoras:")
    print(f"- {casing_issues} palabras podrían encontrarse si pasamos todo a minúsculas.")
    
if __name__ == "__main__":
    analyze_missing_words()
