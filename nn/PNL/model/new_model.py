import re
import unicodedata
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention,Dropout,Concatenate,Bidirectional
from tokenizers import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger
from tensorflow.keras import mixed_precision

def limpiar_texto(texto):
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'/', '', texto)
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ.?, /-]+", '', texto)
    url_pattern = r'(http[s]?://|www\.|//)[^\s/$.?#].[^\s]*'
    texto = re.sub(url_pattern, ' ', texto)
    texto = texto.replace('\xa0', ' ')
    texto = re.sub(r'[~*\-_=]{2,}', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Intentar cargar el tokenizer desde rutas comunes (PNL/saved, model/, cwd)
candidate_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'saved', 'bpe_resumidor.json'),
    os.path.join(os.path.dirname(__file__), 'bpe_resumidor.json'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved', 'bpe_resumidor.json'),
    'bpe_resumidor.json'
]
found = None
for p in candidate_paths:
    p_norm = os.path.normpath(p)
    if os.path.exists(p_norm):
        found = p_norm
        break

if found is None:
    raise FileNotFoundError(
        'No se encontró bpe_resumidor.json. Coloca el archivo en `nn/PNL/saved/` o en el directorio del modelo.'
    )

tokenizer = Tokenizer.from_file(found)

VOCAB_SIZE = tokenizer.get_vocab_size()
EMB_DIM = 128
LATENT_DIM = 256
MAX_LEN_TEXTO = 300
MAX_LEN_RESUMEN = 50



def build_model():
    print("Configurando precisión mixta...")

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Política de precisión: {mixed_precision.global_policy()}")
    print("Precisión mixta activada - los cálculos usarán float16, reduciendo uso de memoria y acelerando entrenamiento.")

    #  ENCODER BIDIRECCIONAL 
    encoder_inputs = Input(shape=(MAX_LEN_TEXTO,), name='input_layer')
    enc_emb = Embedding(VOCAB_SIZE, EMB_DIM, name='embedding')(encoder_inputs)
    enc_emb = Dropout(0.3, name='dropout')(enc_emb)

    # salida total, h_forward, c_forward, h_backward, c_backward
    encoder_lstm = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, return_state=True), name='bidirectional')
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(enc_emb)

    state_h = Concatenate(name='concatenate')([forward_h, backward_h])
    state_c = Concatenate(name='concatenate_1')([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    #  DECODER 

    decoder_inputs = Input(shape=(MAX_LEN_RESUMEN-1,), name='input_layer_1')
    dec_emb_layer = Embedding(VOCAB_SIZE, EMB_DIM, name='embedding_1')
    dec_emb = dec_emb_layer(decoder_inputs)
    dec_emb = Dropout(0.3, name='dropout_1')(dec_emb)
    decoder_lstm = LSTM(LATENT_DIM * 2, return_sequences=True, return_state=True, name='lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    #  ATENCIÓN 
    attn_layer = Attention(name='attention')
    attn_out = attn_layer([decoder_outputs, encoder_outputs])


    decoder_concat_input = Concatenate(axis=-1, name='concatenate_2')([decoder_outputs, attn_out])

    decoder_dense = Dense(VOCAB_SIZE, activation='softmax', dtype='float32', name='dense')
    output = decoder_dense(decoder_concat_input)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    model.summary()
    return model

def build_tokenizer(corpus, vocab_size=15000, save_path="bpe_resumidor.json"):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    with open("corpus.txt", "w") as f:
        for t in corpus:
            f.write(t + "\n")

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["[PAD]", "[START]", "[END]", "[UNK]"]
        )
        tokenizer.train([corpus], trainer)
        tokenizer.save(save_path)


def codificar(lista_textos, max_len):
    ids = [tokenizer.encode(t).ids for t in lista_textos]
    return tf.keras.preprocessing.sequence.pad_sequences(ids, maxlen=max_len, padding='post')
def decodificar(text,tokenizer):

    return " ".join([tokenizer.id_to_token(x).replace("@@ ", "").strip() for x in text[0]])



def train(X_enc, X_dec, Y, X_enc_val, X_dec_val, Y_val):
   
    model = tf.keras.models.load_model(f'/kaggle/working/model_last5.keras')

    csv_filename = 'historial_entrenamiento.csv'

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        
        ModelCheckpoint(
            filepath='mejor_modelo_resumen5.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        CSVLogger(
            filename=csv_filename,
            separator=',',
            append=True  
        )
    ]
    print("Iniciando entrenamiento...")

    hist = model.fit([X_enc, X_dec], Y, batch_size=128, epochs=50, verbose=1,
                    validation_data=([X_enc_val, X_dec_val], Y_val),
                    callbacks=callbacks, initial_epoch=10)

    import matplotlib.pyplot as plt

    def graficar_historial(historial):
        plt.figure(figsize=(12, 5))

        # Gráfica de Pérdida (Loss)
        plt.plot(historial.history['loss'], label='Entrenamiento (loss)')
        plt.plot(historial.history['val_loss'], label='Validación (val_loss)')
        plt.title('Progreso de la Pérdida del Modelo')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        
        plt.show()


    graficar_historial(hist)


    model.save("model_last5.keras")
    model.save_weights("model_weights5.weights.h5")

    print(f"CSV con métricas: {csv_filename}")


def resumir_beam_search(texto_entrada, enc_mod, dec_mod, tokenizer, beam_width=3):
    texto_p = codificar([limpiar_texto(texto_entrada)], MAX_LEN_TEXTO)
    print(decodificar(texto_p,tokenizer))
    enc_out, h, c = enc_mod.predict(texto_p, verbose=0)
    
    start_token = tokenizer.token_to_id("[START]")
    end_token = tokenizer.token_to_id("[END]")

 
    beams = [(0.0, [start_token], h, c)]
    
    for _ in range(MAX_LEN_RESUMEN):
        candidates = []
        
        for log_prob, seq, state_h, state_c in beams:
            # Si el último token es [END], este beam ya terminó
            if seq[-1] == end_token:
                candidates.append((log_prob, seq, state_h, state_c))
                continue
            
            # Predecir siguiente token
            target_seq = np.array([[seq[-1]]])
            tokens_pred, next_h, next_c = dec_mod.predict([target_seq, enc_out, state_h, state_c], verbose=0)
            

            log_probs = np.log(tokens_pred[0, 0, :] + 1e-10) 
            top_indices = np.argsort(log_probs)[-beam_width:]
            
            for idx in top_indices:
                candidates.append((log_prob + log_probs[idx], seq + [idx], next_h, next_c))
        

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        
        # Si todos los beams terminaron en [END], paramos
        if all(seq[-1] == end_token for _, seq, _, _ in beams):
            break


    mejor_seq = beams[0][1]
    
    resumen_final = []
    for idx in mejor_seq:
        token = tokenizer.id_to_token(idx)
        if token in ["[START]", "[END]", "[PAD]", None]:
            continue
        resumen_final.append(token)
    
    resumen_texto = " ".join(resumen_final).replace("@@ ", "").strip()
    return resumen_texto


def build_inference_models(trained_model_path):
    
    full_model = load_model(trained_model_path)

   
    encoder_inputs = full_model.input[0]


    for layer in full_model.layers:
        if "bidirectional" in layer.name.lower():
            encoder_bidirectional = layer
            break


    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bidirectional.output


    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])

    # Modelo de inferencia del Encoder
    encoder_model = Model(
        inputs=encoder_inputs, 
        outputs=[encoder_outputs, state_h, state_c]
    )

    
    decoder_state_input_h = Input(shape=(state_h.shape[1],))
    decoder_state_input_c = Input(shape=(state_c.shape[1],))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


    encoder_outputs_input = Input(shape=(None, encoder_outputs.shape[-1]))


    embedding_layers = [l for l in full_model.layers if isinstance(l, tf.keras.layers.Embedding)]
    decoder_embedding = embedding_layers[1] if len(embedding_layers) > 1 else embedding_layers[0]

    
    lstm_layers = [l for l in full_model.layers if "lstm" in l.name.lower() and "bidirectional" not in l.name.lower()]
    decoder_lstm = lstm_layers[0]


    attention_layer = None
    for layer in full_model.layers:
        if "attention" in layer.name.lower():
            attention_layer = layer
            break

    
    dense_layer = None
    for layer in full_model.layers:
        if layer.name == 'output_layer':
            dense_layer = layer
            break
    if dense_layer is None:

        dense_layers = [l for l in full_model.layers if isinstance(l, tf.keras.layers.Dense)]
        dense_layer = dense_layers[-1]


    decoder_inputs_single = Input(shape=(1,))  # Una palabra a la vez

    # Embedding
    decoder_embeddings_single = decoder_embedding(decoder_inputs_single)

    # LSTM con estados
    decoder_outputs_single, state_h_single, state_c_single = decoder_lstm(
        decoder_embeddings_single, 
        initial_state=decoder_states_inputs
    )

    # Atención u
    attention_result_single = attention_layer(
        [decoder_outputs_single, encoder_outputs_input]
    )

    decoder_concat_input_single = Concatenate(axis=-1)(
        [decoder_outputs_single, attention_result_single]
    )

    # Predicción
    decoder_outputs_single = dense_layer(decoder_concat_input_single)

    decoder_model = Model(
        inputs=[
            decoder_inputs_single,
            encoder_outputs_input,
            decoder_state_input_h,
            decoder_state_input_c
        ],
        outputs=[
            decoder_outputs_single,
            state_h_single,
            state_c_single
        ]
    )
    return encoder_model, decoder_model

