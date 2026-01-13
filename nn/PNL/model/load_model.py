import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from new_model import (
    build_model, 
    tokenizer, 
    MAX_LEN_TEXTO, 
    MAX_LEN_RESUMEN, 
    limpiar_texto, 
    codificar
)

class TextSummarizer:
    """
    Clase para cargar un modelo de redes neuronales (Seq2Seq con Atención)
    y generar resúmenes de texto utilizando Beam Search.
    """
    def __init__(self, weights_path=None):
        """
        Inicializa el modelo, carga los pesos y construye los modelos de inferencia.
        """
        if weights_path is None:
            # Ruta por defecto basada en finetune_keras.py
            weights_path = os.path.join(os.path.dirname(__file__), "..", "saved", "finetune_final.weights.h5")
            # Si no existe, intentar con el modelo base
            if not os.path.exists(weights_path):
                weights_path = os.path.join(os.path.dirname(__file__), "..", "saved", "model_weights5.weights.h5")
        
        self.weights_path = weights_path
        self.tokenizer = tokenizer
        
        print(f"Instanciando TextSummarizer...")
        self.full_model = build_model()
        
        if os.path.exists(self.weights_path):
            print(f"Cargando pesos desde {self.weights_path}...")
            # Si el archivo termina en .weights.h5, usamos load_weights
            if self.weights_path.endswith('.weights.h5'):
                # En Keras 3 es necesario que el modelo tenga la misma arquitectura
                self.full_model.load_weights(self.weights_path)
            else:
                # Si es un modelo completo (.keras o .h5)
                self.full_model = tf.keras.models.load_model(self.weights_path)
            print("✓ Pesos cargados correctamente.")
        else:
            print(f"⚠ Advertencia: No se encontró el archivo de pesos en {self.weights_path}")
        
        self._build_inference_models()

    def _build_inference_models(self):
        """
        Extrae y construye los modelos de encoder y decoder para inferencia
        a partir del modelo completo cargado.
        """
        print("Construyendo modelos de inferencia (Encoder/Decoder)...")
        
        # 1. ENCODER
        encoder_inputs = self.full_model.get_layer('input_layer').input
        encoder_lstm = self.full_model.get_layer('bidirectional')
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm.output
        
        state_h = tf.keras.layers.Concatenate(name='inf_concat_h')([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate(name='inf_concat_c')([forward_c, backward_c])
        
        self.encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
        
        # 2. DECODER
        # Estados iniciales para el decoder (vienen del encoder)
        lat_dim_total = state_h.shape[-1]
        inf_state_h = tf.keras.layers.Input(shape=(lat_dim_total,), name='inf_state_h')
        inf_state_c = tf.keras.layers.Input(shape=(lat_dim_total,), name='inf_state_c')
        
        # Salidas del encoder para la atención
        inf_encoder_out = tf.keras.layers.Input(shape=(MAX_LEN_TEXTO, encoder_outputs.shape[-1]), name='inf_encoder_out')
        
        # Capa de entrada del decoder (un token a la vez)
        inf_decoder_inputs = tf.keras.layers.Input(shape=(1,), name='inf_decoder_input')
        
        # Reutilizar capas del modelo completo
        dec_emb_layer = self.full_model.get_layer('embedding_1')
        dec_lstm_layer = self.full_model.get_layer('lstm')
        attn_layer = self.full_model.get_layer('attention')
        concat_layer = self.full_model.get_layer('concatenate_2')
        dense_layer = self.full_model.get_layer('dense')
        
        # Flujo del decoder
        dec_emb = dec_emb_layer(inf_decoder_inputs)
        dec_out, dec_h, dec_c = dec_lstm_layer(dec_emb, initial_state=[inf_state_h, inf_state_c])
        
        # Atención
        attn_out = attn_layer([dec_out, inf_encoder_out])
        dec_concat = concat_layer([dec_out, attn_out])
        
        # Predicción
        outputs = dense_layer(dec_concat)
        
        self.decoder_model = Model(
            inputs=[inf_decoder_inputs, inf_encoder_out, inf_state_h, inf_state_c],
            outputs=[outputs, dec_h, dec_c]
        )
        print("✓ Modelos de inferencia listos.")

    def summarize(self, text, beam_width=3):
        """
        Genera un resumen para el texto dado utilizando Beam Search.
        """
        if not text.strip():
            return ""
            
        # Preprocesamiento
        texto_limpio = limpiar_texto(text)
        texto_enc = codificar([texto_limpio], MAX_LEN_TEXTO)
        
        # Obtener estados iniciales del encoder
        enc_out, h, c = self.encoder_model.predict(texto_enc, verbose=0)
        
        start_token = self.tokenizer.token_to_id("[START]")
        end_token = self.tokenizer.token_to_id("[END]")
        
        # Estructura del Beam: (log_prob, secuencia_tokens, estado_h, estado_c)
        beams = [(0.0, [start_token], h, c)]
        
        for _ in range(MAX_LEN_RESUMEN):
            candidates = []
            
            for log_prob, seq, state_h, state_c in beams:
                # Si el último token es [END], este beam ya terminó
                if seq[-1] == end_token:
                    candidates.append((log_prob, seq, state_h, state_c))
                    continue
                
                # Predecir siguiente token usando el último de la secuencia
                target_seq = np.array([[seq[-1]]])
                preds, next_h, next_c = self.decoder_model.predict([target_seq, enc_out, state_h, state_c], verbose=0)
                
                # Obtener probabilidades (usamos log para evitar underflow)
                # preds shape: (1, 1, VOCAB_SIZE)
                probs = preds[0, 0, :]
                log_probs = np.log(probs + 1e-10)
                
                # Seleccionar los mejores beam_width candidatos
                top_indices = np.argsort(log_probs)[-beam_width:]
                
                for idx in top_indices:
                    candidates.append((log_prob + log_probs[idx], seq + [idx.item()], next_h, next_c))
            
            # Seleccionar los mejores N candidatos globales
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # Si todos los beams terminaron, paramos
            if all(b[1][-1] == end_token for b in beams):
                break
        
        # Decodificar el mejor beam
        mejor_seq = beams[0][1]
        resumen_tokens = []
        for idx in mejor_seq:
            token = self.tokenizer.id_to_token(idx)
            if token in ["[START]", "[END]", "[PAD]", None]:
                continue
            resumen_tokens.append(token)
            
        resumen_texto = " ".join(resumen_tokens).replace("@@ ", "").strip()
        return resumen_texto

if __name__ == "__main__":
    # Ejemplo de uso
    summarizer = TextSummarizer()
    
    texto_ejemplo = """
    El telescopio espacial James Webb ha capturado imágenes sin precedentes de las galaxias más antiguas del universo. 
    Estas observaciones permiten a los científicos entender mejor cómo se formaron las primeras estrellas después del Big Bang.
    Los datos revelan detalles sobre la composición química y la estructura de estas estructuras masivas.
    """
    
    print("\n--- Generando Resumen ---")
    resumen = summarizer.summarize(texto_ejemplo)
    print(f"Original: {texto_ejemplo.strip()}")
    print(f"Resumen: {resumen}")
