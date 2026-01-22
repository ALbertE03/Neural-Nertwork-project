import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from pgn_model import PointerGeneratorNetwork
from vocabulary import Vocabulary
from beam_search import BeamSearch
from config import Config
from constant import *
from preprocess_data import _clean_text, _tokens_from_doc

def decode_sequence_to_text(id_sequence, vocab, oov_id_to_word):
    """
    Decodifica una secuencia de IDs a texto.
    
    Args:
        id_sequence: List o tensor de IDs
        vocab: Vocabulary object
        oov_id_to_word: Dict[int, str] - Mapeo de IDs extendidos a palabras OOV
    
    Returns:
        List[str] - Palabras decodificadas
    """
    if torch.is_tensor(id_sequence):
        id_sequence = id_sequence.cpu().tolist()
    
    V_base = len(vocab.word_to_id)
    decoded_words = []
    
    for id in id_sequence:
        id = int(id)
        
        # Tokens especiales
        if id == vocab.word2id(vocab.pad_token):
            continue  # Ignorar padding
        elif id == vocab.word2id(vocab.start_decoding):
            continue  # Ignorar START
        elif id == vocab.word2id(vocab.end_decoding):
            break  # Terminar en END
        # IDs base del vocabulario
        elif id < V_base:
            decoded_words.append(vocab.id2word(id))
        # IDs extendidos (OOV copiado)
        elif id in oov_id_to_word:
            decoded_words.append(oov_id_to_word[id])
        else:
            decoded_words.append(vocab.unk_token)
    
    return decoded_words


def create_oov_id_to_word_map(oov_words, V_base):
    """
    Crea el mapeo ID a Palabra OOV.
    
    Args:
        oov_words: List[str] - Palabras OOV del ejemplo
        V_base: int - Tamaño del vocabulario base
    
    Returns:
        Dict[int, str] - Mapeo de ID extendido a palabra OOV
    """
    oov_id_to_word = {}
    oov_id = V_base
    
    for word in oov_words:
        if word == '':  # Ignorar padding
            continue
        oov_id_to_word[oov_id] = word
        oov_id += 1
    
    return oov_id_to_word

    return oov_id_to_word


def plot_attention(attention, source_tokens, generated_tokens, filename_prefix):
    """
    Genera visualizaciones de la atención.
    
    Args:
        attention: numpy array (tgt_len, src_len)
        source_tokens: List[str]
        generated_tokens: List[str]
        filename_prefix: str - Prefijo para los archivos
    """
    tgt_len, src_len = attention.shape
    
    # 1. HEATMAP
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    cax = ax.matshow(attention, cmap='viridis', aspect='auto')
    fig.colorbar(cax)
    
    # Ejes
    ax.set_yticks(np.arange(len(generated_tokens)))
    ax.set_yticklabels(generated_tokens)
    
    # X-axis: si hay muchos tokens, mostrar solo algunos
    if len(source_tokens) > 50:
        step = len(source_tokens) // 20
        ax.set_xticks(np.arange(0, len(source_tokens), step))
        ax.set_xticklabels([source_tokens[i] for i in np.arange(0, len(source_tokens), step)], rotation=90)
    else:
        ax.set_xticks(np.arange(len(source_tokens)))
        ax.set_xticklabels(source_tokens, rotation=90)
        
    plt.xlabel('Source Text')
    plt.ylabel('Generated Summary')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_heatmap.png")
    plt.close()
    
    # 2. ATTENTION PROFILE (Start vs End focus)
    # Sumar atención recibida por cada token del source a lo largo de toda la generación
    total_attention = np.sum(attention, axis=0) # (src_len,)
    # Normalizar para que sume 1 (opcional, para ver distribución relativa)
    total_attention_norm = total_attention / (np.sum(total_attention) + 1e-10)
    
    plt.figure(figsize=(12, 6))
    plt.plot(total_attention_norm, label='Attention Mass')
    
    # Añadir línea de tendencia suavizada
    if len(total_attention_norm) > 10:
        window = max(len(total_attention_norm) // 20, 5)
        smoothed = np.convolve(total_attention_norm, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(len(smoothed)) + window//2, smoothed, 'r-', linewidth=2, label='Smoothed')

    plt.title('Global Attention Distribution over Source Text')
    plt.xlabel('Source Token Position')
    plt.ylabel('Accumulated Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Marcar cuartiles
    plt.axvline(x=len(source_tokens)*0.25, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=len(source_tokens)*0.50, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=len(source_tokens)*0.75, color='gray', linestyle='--', alpha=0.5)
    plt.text(len(source_tokens)*0.1, max(total_attention_norm), 'Beginning', ha='center')
    plt.text(len(source_tokens)*0.9, max(total_attention_norm), 'End', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_profile.png")
    plt.close()

    print(f"✓ Visualizaciones guardadas con prefijo {filename_prefix}")
class Generator:
    """
    Clase para generar resúmenes usando el modelo entrenado.
    """
    
    def __init__(self, config, vocab, model_path):
        """
        Args:
            config: Config object
            vocab: Vocabulary object
            model_path: Ruta al checkpoint del modelo
        """
        self.config = config
        self.vocab = vocab
        self.device = config['device']
        
        print(f"Cargando modelo desde {model_path}")
        self.model = PointerGeneratorNetwork(config, vocab).to(self.device)

        print(f"✓ Modelo creado")
        self.model.to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
      
        # Beam search
        self.beam_search = BeamSearch(
            self.model,
            vocab,
            beam_size=config['beam_size'],
            max_len=config['tgt_len']
        )
        print(f"✓ Modelo cargado y en modo evaluación")

    def _get_extended_src_ids(self, src_tokens_raw):
        """Obtener IDs extendidos para fuente con OOVs (Helper para inferencia)"""
        extended_src_ids = []
        temp_oov_map = {}
        oov_words = []
        
        vocab_size = len(self.vocab.word_to_id)
        oov_id_counter = vocab_size
        
        for token in src_tokens_raw:
            base_id = self.vocab.word2id(token)
            
            if base_id == self.vocab.word2id(self.vocab.unk_token):
                if token not in temp_oov_map:
                    temp_oov_map[token] = oov_id_counter
                    oov_words.append(token)
                    oov_id_counter += 1
                extended_src_ids.append(temp_oov_map[token])
            else:
                extended_src_ids.append(base_id)
        
        return extended_src_ids, temp_oov_map, oov_words

    def predict_single_text(self, raw_text, strategy=None):
        """
        Genera resumen para un texto crudo.
        """
        # 1. Preprocesar
        cleaned_text = _clean_text(raw_text)
        doc = self.vocab.nlp(cleaned_text)
        tokens = _tokens_from_doc(doc)
        if not tokens:
            return ""

        raw_sentences = " ".join(tokens).split("[.]")
        trimmed_src_tokens = []
        for sentence in raw_sentences:
            sentence_tokens = sentence.split()
            tokens_to_add = sentence_tokens
            
            if len(trimmed_src_tokens) + len(tokens_to_add) > self.config['src_len']:
                break
            
            trimmed_src_tokens.extend(tokens_to_add)
        
    
        
        #  Preparar tensores
        extended_src_ids, oov_map, oov_words = self._get_extended_src_ids(trimmed_src_tokens)
        
       

        # Encoder input (UNKs)
        encoder_input = [
            i if i < len(self.vocab.word_to_id) else self.vocab.word2id(self.vocab.unk_token)
            for i in extended_src_ids
        ]
        
        # Crear batch (añadir dimensión batch=1)
        batch = {
            'encoder_input': torch.tensor([encoder_input], dtype=torch.long).to(self.device),
            'extended_encoder_input': torch.tensor([extended_src_ids], dtype=torch.long).to(self.device),
            'encoder_length': torch.tensor([len(trimmed_src_tokens)], dtype=torch.long).to(self.device),
            'encoder_mask': torch.ones((1, len(trimmed_src_tokens)), dtype=torch.bool).to(self.device),
            'oov_words': [oov_words], # Lista de listas
            'max_oov_len': torch.tensor([len(oov_words)], dtype=torch.long).to(self.device),
            'decoder_target': torch.zeros((1, 1), dtype=torch.long).to(self.device)
        }
        
        # 3. Generar
        candidates = []
        current_strategy = strategy if strategy else self.config['decoding_strategy']
        
        with torch.no_grad():
            if current_strategy == 'beam_search':
                hypotheses = self.beam_search.search(batch)
                # Tomar solo la mejor hipótesis
                best_hyp = hypotheses[0]
                
                # Stack attention dists
                attention_dists = torch.stack(best_hyp.attention_dists, dim=0) # (tgt_len, 1, src_len)
                attention_dists = attention_dists.squeeze(1) # (tgt_len, src_len)
                
                candidates.append({
                    'ids': best_hyp.tokens[1:],
                    'log_probs': best_hyp.log_probs[1:],
                    'p_gens': best_hyp.p_gens,
                    'attention_dists': attention_dists
                })
            else:
                generated_ids, p_gens, log_probs_tensor, attention_dists = self.model.decode_greedy(batch, max_len=self.config['tgt_len'])
                generated_ids = generated_ids[0].cpu().tolist()
                p_gens = [pg[0] for pg in p_gens[0]] # Unpack batch
                log_probs = log_probs_tensor[0].cpu().tolist()
                attention_dists = attention_dists[0] # (tgt_len, src_len)
                
                candidates.append({
                    'ids': generated_ids,
                    'log_probs': log_probs,
                    'p_gens': p_gens,
                    'attention_dists': attention_dists
                })
        
        # 4. Decodificar
        vocab_size = len(self.vocab.word_to_id)
        oov_id_to_word = {vocab_size + i: w for i, w in enumerate(oov_words)}
        
        results = []
        
        for cand in candidates:
            generated_ids = cand['ids']
            log_probs = cand['log_probs']
            generated_ids = cand['ids']
            log_probs = cand['log_probs']
            p_gens = cand['p_gens']
            attention_dists = cand['attention_dists']
            
            decoded_words = []
            details = []
            
            for i, token_id in enumerate(generated_ids):
                token_id = int(token_id)
                
                if token_id == self.vocab.word2id(self.vocab.pad_token):
                    continue
                if token_id == self.vocab.word2id(self.vocab.start_decoding):
                    continue
                if token_id == self.vocab.word2id(self.vocab.end_decoding):
                    break
                
                if token_id < vocab_size:
                    word = self.vocab.id2word(token_id)
                elif token_id in oov_id_to_word:
                    word = oov_id_to_word[token_id]
                else:
                    word = self.vocab.unk_token
                    
                decoded_words.append(word)
                
                # Get metrics
                prob = torch.exp(torch.tensor(log_probs[i])).item()
                
                p_gen_val = 0.0
                if i < len(p_gens):
                    if torch.is_tensor(p_gens[i]):
                        p_gen_val = p_gens[i].item()
                    else:
                        p_gen_val = p_gens[i]
                
                details.append({
                    'token': word,
                    'prob': prob,
                    'p_gen': p_gen_val,
                    'p_copy': 1.0 - p_gen_val
                })
            
            results.append((' '.join(decoded_words), details, attention_dists, decoded_words, trimmed_src_tokens))
            
        return results[0]
    
    def generate(self, test_loader, output_file=None, num_examples=None):
        """
        Genera resúmenes para el dataset de test.
        
        Args:
            test_loader: DataLoader de test
            output_file: Archivo donde guardar los resúmenes generados
            num_examples: Número de ejemplos a generar (None = todos)
        
        Returns:
            List[Dict] - Lista con resultados (source, target, generated)
        """
        results = []
        V_base = len(self.vocab.word_to_id)
        
        print(f"\n{'='*60}")
        print(f"Generando resúmenes")
        print(f"{'='*60}")
        print(f"Estrategia: {self.config['decoding_strategy']}")
        print(f"Beam size: {self.config['beam_size']}")
        print(f"{'='*60}\n")
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Generando", total=len(test_loader))
            
            for batch_idx, batch in enumerate(pbar):
                batch_size = batch['encoder_input'].size(0)
                
                for b in range(batch_size):
                    # Extraer ejemplo individual
                    single_batch = {
                        'encoder_input': batch['encoder_input'][b:b+1],
                        'extended_encoder_input': batch['extended_encoder_input'][b:b+1],
                        'encoder_length': batch['encoder_length'][b:b+1],
                        'encoder_mask': batch['encoder_mask'][b:b+1],
                        'decoder_target': batch['decoder_target'][b:b+1]
                    }
                    
                    oov_words = batch['oov_words'][b]
                    oov_map = create_oov_id_to_word_map(oov_words, V_base)
                    
                    
                    # Generar resumen
                    if self.config['decoding_strategy'] == 'beam_search':
                        hypotheses = self.beam_search.search(single_batch)
                        hypothesis = hypotheses[0]
                        generated_ids = hypothesis.tokens[1:]  # Quitar START
                    else:  # greedy
                        generated_ids, _, _ = self.model.decode_greedy(single_batch, max_len=self.config['tgt_len'])
                        generated_ids = generated_ids[0].cpu().tolist()
                    source_ids = single_batch['extended_encoder_input'][0].cpu().tolist()
                    target_ids = single_batch['decoder_target'][0].cpu().tolist()
                    
                    source_text = decode_sequence_to_text(source_ids, self.vocab, oov_map)
                    target_text = decode_sequence_to_text(target_ids, self.vocab, oov_map)
                    generated_text = decode_sequence_to_text(generated_ids, self.vocab, oov_map)
                    
                    result = {
                        'source': ' '.join(source_text),
                        'target': ' '.join(target_text),
                        'generated': ' '.join(generated_text)
                    }
                    
                    results.append(result)
                    
                    if num_examples and len(results) >= num_examples:
                        break
                
                if num_examples and len(results) >= num_examples:
                    break
            
            pbar.close()
        
        print(f"\n✓ Generados {len(results)} resúmenes")
        
        # Guardar resultados
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Resultados guardados en {output_file}")
            
            # También guardar en formato legible
            txt_file = output_file.replace('.json', '.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    f.write(f"{'='*60}\n")
                    f.write(f"Ejemplo {i+1}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"SOURCE:\n{result['source']}\n\n")
                    f.write(f"TARGET:\n{result['target']}\n\n")
                    f.write(f"GENERATED:\n{result['generated']}\n\n")
            
            print(f"✓ Resultados legibles en {txt_file}")
        
        return results

def interactive_mode(generator):
    print("\n" + "="*60)
    print("MODO VISUALIZACIÓN")
    print("="*60)
    
    # Texto hardcoded para prueba rápida
    text ='Actores políticos e integrantes de la sociedad civil panameña reafirmaron en una declaratoria firmada este miércoles que el memorándum de entendimiento firmado por el Gobierno de Panamá y EE.UU. es lesivo a la soberanía nacional.\nLa declaración en rechazo al memorándum de entendimiento pactado entre el Gobierno de Panamá y Estados Unidos el pasado 10 de abril, durante la visita del secretario de Defensa de EE.UU., Pete Hegseth refirió que todos los documentos firmados son lesivos a la soberanía nacional, contrarios a la Constitución Política y a las leyes de nuestro país.\nLos políticos panameños e integrantes de la sociedad civil precisaron que “conscientes de la gravedad de los acontecimientos en los que actualmente se encuentra sumida nuestra República y ante las intenciones expansionistas y hegemónicas de los Estados Unidos de América, subrayamos que el Tratado de Neutralidad del Canal de Panamá garantiza el trato igualitario y el tránsito pacífico a las naves de todas las naciones, sin discriminación ni privilegios”.\nEn este sentido “reafirman que, después del cumplimiento de los Tratados Torrijos-Carter, solo la República de Panamá, a través de la Autoridad del Canal de Panamá, administra y maneja, eficientemente, el Canal, sin presencia de ninguna fuerza militar extranjera ni bases militares”.\nAl insistir que deben deponerse intereses particulares, la declaratoria denuncia “las inminentes violaciones al régimen de neutralidad del Canal de Panamá, ante el evidente trato discriminatorio y no igualitario otorgado a las naves de guerra estadounidenses, así como el envío de fuerzas militares estadounidenses a sitios conjuntos en el territorio nacional”.\nExigen además “el cumplimiento de lo dispuesto en el artículo 325 de la Constitución Política, que dispone que cualquier acuerdo que modifique o transgreda el régimen de neutralidad y otros acuerdos será objeto de aprobación por la Asamblea Nacional y sometido a referéndum nacional”.\nAl declarar que es necesario hacer frente a cualquier amenaza presente o futura a nuestro país y al Canal de Panamá, los firmantes insisten en los recursos de la diplomacia, el multilateralismo y el derecho internacional, incluyendo la Asamblea General de las Naciones Unidas, el Consejo de Seguridad y la Organización de Estados Americanos, activando a todas las representaciones diplomáticas para recabar el apoyo a la causa panameña.\nEntre tanto, hacen un llamado a todas “las fuerzas vivas del país a fin de defender nuestra soberanía e integridad territorial, coadyuvar a forjar la unidad patriótica que requieren estas circunstancias y la formación de una agrupación amplia de panameños en defensa de la soberanía e integridad de nuestro país”.\nLos documentos firmados en la reciente visita a Panamá del Secretario de Defensa de los Estados Unidos, Pete Hegseth, fueron la Declaración Conjunta Mulino-Hegseth, el Memorando de Entendimiento Ábrego-Hegseth y la Declaración Conjunta Icaza-Hegseth.\nEntre los firmantes destacan que el abogado en derecho internacional Alonso Illueca, el expresidente Martín Torrijos, el presidente de la coalición Vamos Juan Diego Vásquez, el excandidato presidencial Ricardo Lombana, la abogada Suky Yard, el diputado Crispiano Adames, el vicealcalde de la ciudad de Panamá Roberto Ruiz Díaz, la exvicealcaldesa Raisa Banfield, entre otros.' 
    
    print(f"Procesando texto de longitud: {len(text.split())} palabras...")
    
    try:
        # Generar con ambas estrategias
        summary_beam, details_beam, attn_beam, tokens_beam, src_tokens = generator.predict_single_text(text, strategy='beam_search')
        summary_greedy, details_greedy, attn_greedy, tokens_greedy, _ = generator.predict_single_text(text, strategy='greedy')
        
        # Plot attention
        if len(attn_beam) > 0:
            plot_attention(
                attn_beam.cpu().numpy(),
                src_tokens,
                tokens_beam,
                "attention_beam"
            )
            
        if len(attn_greedy) > 0:
            plot_attention(
                attn_greedy.cpu().numpy(),
                src_tokens,
                tokens_greedy,
                "attention_greedy"
            )
        

        print(text)
        
        # Mostrar Beam Search
        print(f"\nRESUMEN GENERADO (Beam Search):")
        print(f"{summary_beam}")
        print("-" * 60)
        
        print(f"\nRESUMEN GENERADO (Greedy):")
        print(f"{summary_greedy}")
        print("-" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Función principal para generar resúmenes.
    """
    # 1. Cargar vocabulario
    print("Cargando vocabulario...")
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
    vocab.build_vocabulary()
    print(f"✓ Vocabulario cargado: {vocab.total_size()} palabras")
    
    # 2. Configurar
    config = Config(
        max_vocab_size=vocab.total_size(),
        src_len=MAX_LEN_SRC,
        tgt_len=MAX_LEN_TGT,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_enc_layers=NUM_ENC_LAYERS,
        num_dec_layers=NUM_DEC_LAYERS,
        use_gpu=USE_GPU,
        is_pgen=IS_PGEN,
        is_coverage=IS_COVERAGE,
        dropout_ratio=DROPOUT_RATIO,
        bidirectional=BIDIRECTIONAL,
        device=DEVICE,
        decoding_strategy=DECODING_STRATEGY,
        beam_size=BEAM_SIZE,
        gpu_id=GPU_ID
    )
    
    model_path = os.path.join('/Users/alberto/Desktop/ARI/Report_Generator/src/summarization/finetune_last-v2.pt')
        
    generator = Generator(config, vocab, model_path)

    interactive_mode(generator)
 

