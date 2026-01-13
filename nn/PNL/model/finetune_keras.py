import os
import tensorflow as tf
import numpy as np
from new_model import build_model, codificar, limpiar_texto, tokenizer, VOCAB_SIZE, EMB_DIM, LATENT_DIM, MAX_LEN_TEXTO, MAX_LEN_RESUMEN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

# Directorios de datos
DATA_DIR = "/Users/alberto/Desktop/Neural-Nertwork-project/nn/PNL/data"
SRC_DIR = os.path.join(DATA_DIR, "outputs_src")
TGT_DIR = os.path.join(DATA_DIR, "outputs_tgt")
WEIGHTS_PATH = "/Users/alberto/Desktop/Neural-Nertwork-project/nn/PNL/saved/model_weights5.weights.h5"

def load_data_from_dirs(src_dir, tgt_dir):
    src_lines = []
    tgt_lines = []
    
    src_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.src.txt')])
    print(f"Escaneando {len(src_files)} archivos en {src_dir}...")
    
    for src_file in src_files:
        try:
            # Extraer ID: data_XXX.src.txt -> XXX
            file_id = src_file.split('_')[1].split('.')[0]
            tgt_file = f"target_{file_id}.tgt.txt"
            
            src_path = os.path.join(src_dir, src_file)
            tgt_path = os.path.join(tgt_dir, tgt_file)
            
            if os.path.exists(tgt_path):
                with open(src_path, 'r', encoding='utf-8') as f:
                    s_lines = [limpiar_texto(l) for l in f if l.strip()]
                with open(tgt_path, 'r', encoding='utf-8') as f:
                    t_lines = [limpiar_texto(l) for l in f if l.strip()]
                
                if len(s_lines) == len(t_lines):
                    src_lines.extend(s_lines)
                    tgt_lines.extend(t_lines)
                else:
                    print(f"⚠ Saltando {src_file}: Desajuste de líneas ({len(s_lines)} vs {len(t_lines)})")
        except Exception as e:
            print(f"Error procesando {src_file}: {e}")
            
    print(f"✓ Cargados {len(src_lines)} pares de datos.")
    return src_lines, tgt_lines

def prepare_sequences(src_lines, tgt_lines):
    print("Codificando secuencias...")
    X_enc = codificar(src_lines, MAX_LEN_TEXTO)
    
    # Preparar datos para el decoder (Teacher Forcing)
    # El decoder recibe [START] + resumen hasta n-1
    # El target es resumen desde token 1 hasta [END]
    
    tgt_with_tokens = ["[START] " + t + " [END]" for t in tgt_lines]
    
    # Codificamos el resumen completo (con START y END)
    full_tgt_ids = [tokenizer.encode(t).ids for t in tgt_with_tokens]
    
    # X_dec: [START] ... [END] (longitud MAX_LEN_RESUMEN - 1)
    # Y: ... [END] [PAD] ... (longitud MAX_LEN_RESUMEN - 1)
    
    X_dec = []
    Y = []
    
    for ids in full_tgt_ids:
        # Para X_dec usamos los tokens desde 0 hasta el penúltimo
        dec_seq = ids[:-1]
        # Para Y usamos los tokens desde 1 hasta el último
        target_seq = ids[1:]
        
        X_dec.append(dec_seq)
        Y.append(target_seq)
    
    X_dec = tf.keras.preprocessing.sequence.pad_sequences(X_dec, maxlen=MAX_LEN_RESUMEN-1, padding='post')
    Y = tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen=MAX_LEN_RESUMEN-1, padding='post')
    
    return X_enc, X_dec, Y

def finetune():
    # 1. Cargar datos
    src_lines, tgt_lines = load_data_from_dirs(SRC_DIR, TGT_DIR)
    if not src_lines:
        print("No hay datos para entrenar.")
        return
        
    X_enc, X_dec, Y = prepare_sequences(src_lines, tgt_lines)
    
    # 2. Construir modelo
    print(f"Construyendo modelo con VOCAB_SIZE={VOCAB_SIZE}, EMB_DIM={EMB_DIM}, LATENT_DIM={LATENT_DIM}")
    model = build_model()
    
    # 3. Cargar pesos
    if os.path.exists(WEIGHTS_PATH):
        print(f"Cargando pesos desde {WEIGHTS_PATH}...")
        import h5py
        
        try:
            with h5py.File(WEIGHTS_PATH, 'r') as f:
                layers_data = f['layers']
                
                # Mapeo de capas y sus pesos
                for layer in model.layers:
                    name = layer.name
                    if name in layers_data:
                        print(f"  - Cargando pesos para {name}...")
                        weights = []
                        
                        if 'embedding' in name:
                            weights.append(layers_data[name]['vars/0'][:])
                        elif name == 'dense':
                            weights.append(layers_data[name]['vars/0'][:])
                            weights.append(layers_data[name]['vars/1'][:])
                        elif name == 'lstm':
                            weights.append(layers_data[name]['cell/vars/0'][:])
                            weights.append(layers_data[name]['cell/vars/1'][:])
                            weights.append(layers_data[name]['cell/vars/2'][:])
                        elif name == 'bidirectional':
                            # Forward
                            weights.append(layers_data[name]['forward_layer/cell/vars/0'][:])
                            weights.append(layers_data[name]['forward_layer/cell/vars/1'][:])
                            weights.append(layers_data[name]['forward_layer/cell/vars/2'][:])
                            # Backward
                            weights.append(layers_data[name]['backward_layer/cell/vars/0'][:])
                            weights.append(layers_data[name]['backward_layer/cell/vars/1'][:])
                            weights.append(layers_data[name]['backward_layer/cell/vars/2'][:])
                        
                        if weights:
                            layer.set_weights(weights)
                            print(f"    ✓ {name} OK")
            print("✓ Todos los pesos cargados manualmente.")
        except Exception as e:
            print(f"❌ Error al cargar pesos manualmente: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"⚠ Archivo de pesos no encontrado en {WEIGHTS_PATH}")
        return

    # 4. Congelar capas (Encoder y Embeddings)
    print("Congelando Encoder y Embeddings...")
    for layer in model.layers:
        # Congelamos embeddings
        if 'embedding' in layer.name.lower():
            layer.trainable = False
            print(f"  - Capa {layer.name} congelada.")
        # Congelamos el encoder (Bidirectional LSTM)
        if 'bidirectional' in layer.name.lower():
            layer.trainable = False
            print(f"  - Capa {layer.name} congelada.")
            
    # Re-compilar para aplicar cambios en trainable
    # Usamos un learning rate super bajo para fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
                  loss='sparse_categorical_crossentropy')
    
    model.summary()
    
    # 5. Entrenamiento
    checkpoint_path = os.path.join(DATA_DIR, "..", "saved", "finetune_best_keras.weights.h5")
    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='loss', save_best_only=True, save_weights_only=True),
        CSVLogger(filename='finetune_log.csv')
    ]
    
    print("Iniciando fine-tuning...")
    model.fit([X_enc, X_dec], Y, batch_size=64, epochs=10, validation_split=0.2, callbacks=callbacks)
    
    # Guardar modelo final
    model.save_weights(os.path.join(DATA_DIR, "..", "saved", "finetune_final.weights.h5"))
    print("✓ Fine-tuning completado.")

if __name__ == "__main__":
    finetune()
