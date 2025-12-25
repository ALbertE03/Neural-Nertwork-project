import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk

from pgn_model import PointerGeneratorNetwork
from beam_search import BeamSearch
from dataset import PGNDataset, pgn_collate_fn
from vocabulary import Vocabulary
from config import Config
from constant import *

# Descargar recursos de NLTK necesarios para METEOR
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def decode_sequence_to_text(id_sequence, vocab, oov_id_to_word):
    """Decodifica una secuencia de IDs a texto."""
    if torch.is_tensor(id_sequence):
        id_sequence = id_sequence.cpu().tolist()
    
    V_base = len(vocab.word_to_id)
    decoded_words = []
    
    for id in id_sequence:
        id = int(id)
        
        if id == vocab.word2id(vocab.pad_token):
            continue
        elif id == vocab.word2id(vocab.start_decoding):
            continue
        elif id == vocab.word2id(vocab.end_decoding):
            break
        elif id < V_base:
            decoded_words.append(vocab.id2word(id))
        elif id in oov_id_to_word:
            decoded_words.append(oov_id_to_word[id])
        else:
            decoded_words.append(vocab.unk_token)
    
    return decoded_words


def create_oov_id_to_word_map(oov_words, V_base):
    """Crea el mapeo ID a Palabra OOV."""
    oov_id_to_word = {}
    oov_id = V_base
    
    for word in oov_words:
        if word == '':
            continue
        oov_id_to_word[oov_id] = word
        oov_id += 1
    
    return oov_id_to_word


def calculate_rouge_scores(reference, candidate):
    """
    Calcula ROUGE scores usando la librería rouge.
    
    Args:
        reference: str - Resumen de referencia
        candidate: str - Resumen generado
        
    Returns:
        Dict con ROUGE-1, ROUGE-2 y ROUGE-L scores
    """
    if not reference.strip() or not candidate.strip():
        return {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }
    
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference)[0]
        return scores
    except Exception as e:
        print(f"⚠ Error calculando ROUGE: {e}")
        return {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }


def calculate_meteor(reference, candidate):
    """
    Calcula METEOR score usando NLTK.
    
    Args:
        reference: str - Resumen de referencia
        candidate: str - Resumen generado
        
    Returns:
        float - METEOR score
    """
    if not reference.strip() or not candidate.strip():
        return 0.0
    
    try:
        # METEOR requiere lista de tokens
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        
        # METEOR espera una lista de referencias
        score = meteor_score([reference_tokens], candidate_tokens)
        return score
    except Exception as e:
        print(f"⚠ Error calculando METEOR: {e}")
        return 0.0


class Evaluator:
        
    """
    Clase para evaluar el modelo en el dataset de test.
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
        
        # Cargar modelo
        print(f"Cargando modelo desde {model_path}")
        self.model = PointerGeneratorNetwork(config, vocab).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Modelo cargado (Epoch {checkpoint['epoch']})")
        print(f"✓ Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # Beam search
        self.beam_search = BeamSearch(
            self.model,
            vocab,
            beam_size=config['beam_size'],
            max_len=config['tgt_len']
        )
    def _copy_rate(self, candidate_tokens, source_tokens):
            """Porcentaje de palabras del resumen generado que aparecen en el source."""
            if not candidate_tokens or not source_tokens:
                return 0.0
            source_set = set(source_tokens)
            copy_count = sum(1 for w in candidate_tokens if w in source_set)
            return copy_count / len(candidate_tokens)

    def _ngram_overlap(self, candidate_tokens, source_tokens, n=2):
            """Porcentaje de n-gramas del resumen generado que aparecen en el source."""
            if len(candidate_tokens) < n or len(source_tokens) < n:
                return 0.0
            def ngrams(tokens, n):
                return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
            cand_ngrams = ngrams(candidate_tokens, n)
            src_ngrams = ngrams(source_tokens, n)
            if not cand_ngrams:
                return 0.0
            overlap = len(cand_ngrams & src_ngrams)
            return overlap / len(cand_ngrams)
    def evaluate(self, test_loader, num_examples=None):
        """
        Evalúa el modelo en el dataset de test.
        
        Args:
            test_loader: DataLoader de test
            num_examples: Número de ejemplos a evaluar (None = todos)
            
        Returns:
            Dict con métricas y resultados
        """
        V_base = len(self.vocab.word_to_id)
        
        # Métricas acumuladas
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        meteor_scores = []
        copy_rates = []
        bigram_overlaps = []
        p_gen_avgs = [] # Promedio de p_gen por ejemplo
        
        test_loss = 0.0
        num_batches = 0
        
        results = []
        
        print(f"\n{'='*60}")
        print(f"Evaluando modelo en Test Set")
        print(f"{'='*60}")
        print(f"Estrategia: {self.config['decoding_strategy']}")
        print(f"Beam size: {self.config['beam_size']}")
        print(f"{'='*60}\n")
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluando", total=len(test_loader))
            
            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    continue
                
                # Calcular loss
                outputs = self.model(batch, is_training=True)
                test_loss += outputs['loss'].item()
                num_batches += 1
                
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
                        hypothesis = self.beam_search.search(single_batch)
                        generated_ids = hypothesis.tokens[1:]  # Quitar START
                        # Extraer p_gens de la hipótesis (lista de tensores (1,1))
                        p_gens_list = hypothesis.p_gens
                        if p_gens_list:
                            p_gens_tensor = torch.cat(p_gens_list).squeeze() # (tgt_len,)
                        else:
                            p_gens_tensor = torch.tensor([])
                            
                    else:  # greedy
                        generated_ids, p_gens_tensor = self.model.decode_greedy(single_batch, max_len=self.config['tgt_len'])
                        generated_ids = generated_ids[0].cpu().tolist()
                        p_gens_tensor = p_gens_tensor[0].squeeze(-1).cpu() # (max_len,)
                    
                    # Calcular promedio de p_gen para este ejemplo
                    if p_gens_tensor.numel() > 0:
                        avg_p_gen_example = p_gens_tensor.mean().item()
                    else:
                        avg_p_gen_example = 0.0
                        
                    p_gen_avgs.append(avg_p_gen_example)
                    
                    # Decodificar a texto
                    target_ids = single_batch['decoder_target'][0].cpu().tolist()
                    
                    reference = decode_sequence_to_text(target_ids, self.vocab, oov_map)
                    candidate = decode_sequence_to_text(generated_ids, self.vocab, oov_map)
                    
                    # Convertir a string
                    reference_text = ' '.join(reference)
                    candidate_text = ' '.join(candidate)
                    
                    # Calcular ROUGE usando la librería
                    rouge_scores = calculate_rouge_scores(reference_text, candidate_text)
                    # Calcular METEOR
                    meteor = calculate_meteor(reference_text, candidate_text)
                    rouge1_scores.append(rouge_scores['rouge-1']['f'])
                    rouge2_scores.append(rouge_scores['rouge-2']['f'])
                    rougeL_scores.append(rouge_scores['rouge-l']['f'])
                    meteor_scores.append(meteor)
                    # Calcular tasa de copia y bigram overlap
                    src_ids = single_batch['encoder_input'][0].cpu().tolist()
                    source_tokens = decode_sequence_to_text(src_ids, self.vocab, oov_map)
                    copy_rate = self._copy_rate(candidate, source_tokens)
                    bigram_overlap = self._ngram_overlap(candidate, source_tokens, n=2)

                    copy_rates.append(copy_rate)
                    bigram_overlaps.append(bigram_overlap)

                    # Guardar resultado
                    result = {
                        'reference': reference_text,
                        'candidate': candidate_text,
                        'rouge1_f1': rouge_scores['rouge-1']['f'],
                        'rouge2_f1': rouge_scores['rouge-2']['f'],
                        'rougeL_f1': rouge_scores['rouge-l']['f'],
                        'meteor': meteor,
                        'copy_rate': copy_rate,
                        'bigram_overlap': bigram_overlap,
                        'avg_p_gen': avg_p_gen_example,
                        'avg_p_copy': 1.0 - avg_p_gen_example
                    }
                    results.append(result)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{outputs['loss'].item():.4f}",
                        'R1': f"{np.mean(rouge1_scores):.4f}",
                        'R2': f"{np.mean(rouge2_scores):.4f}",
                        'RL': f"{np.mean(rougeL_scores):.4f}",
                        'M': f"{np.mean(meteor_scores):.4f}",
                        'Copy': f"{np.mean(copy_rates):.2f}",
                        'BiOv': f"{np.mean(bigram_overlaps):.2f}",
                        'P_Gen': f"{np.mean(p_gen_avgs):.2f}"
                    })
                    
                    if num_examples and len(results) >= num_examples:
                        break
                
                if num_examples and len(results) >= num_examples:
                    break
            
            pbar.close()
        
        # Calcular promedios
        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
        avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
        avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0.0
        avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0.0
        avg_meteor = np.mean(meteor_scores) if meteor_scores else 0.0
        
        avg_copy_rate = np.mean(copy_rates) if copy_rates else 0.0
        avg_bigram_overlap = np.mean(bigram_overlaps) if bigram_overlaps else 0.0
        avg_p_gen = np.mean(p_gen_avgs) if p_gen_avgs else 0.0
        
        metrics = {
            'test_loss': avg_test_loss,
            'rouge1_f1': avg_rouge1,
            'rouge2_f1': avg_rouge2,
            'rougeL_f1': avg_rougeL,
            'meteor': avg_meteor,
            'copy_rate': avg_copy_rate,
            'bigram_overlap': avg_bigram_overlap,
            'avg_p_gen': avg_p_gen,
            'avg_p_copy': 1.0 - avg_p_gen,
            'num_examples': len(results)
        }
        
        return metrics, results
    
    def save_results(self, metrics, results, output_dir):
        """
        Guarda los resultados de la evaluación.
        
        Args:
            metrics: Dict con métricas
            results: Lista de resultados por ejemplo
            output_dir: Directorio donde guardar
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar métricas
        metrics_path = os.path.join(output_dir, 'test_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Métricas guardadas en {metrics_path}")
        
        # Guardar resultados completos
        results_path = os.path.join(output_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Resultados guardados en {results_path}")
        
        # Guardar formato legible
        txt_path = os.path.join(output_dir, 'test_results.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"MÉTRICAS DE EVALUACIÓN\n")
            f.write(f"{'='*60}\n")
            f.write(f"Test Loss: {metrics['test_loss']:.4f}\n")
            f.write(f"ROUGE-1 F1: {metrics['rouge1_f1']:.4f}\n")
            f.write(f"ROUGE-2 F1: {metrics['rouge2_f1']:.4f}\n")
            f.write(f"ROUGE-L F1: {metrics['rougeL_f1']:.4f}\n")
            f.write(f"METEOR: {metrics['meteor']:.4f}\n")
            f.write(f"Copy Rate: {metrics['copy_rate']:.4f}\n")
            f.write(f"Bigram Overlap: {metrics['bigram_overlap']:.4f}\n")
            f.write(f"Avg P_Gen: {metrics['avg_p_gen']:.4f}\n")
            f.write(f"Avg P_Copy: {metrics['avg_p_copy']:.4f}\n")
            f.write(f"Ejemplos evaluados: {metrics['num_examples']}\n")
            f.write(f"{'='*60}\n\n")
            
            for i, result in enumerate(results[:10]):  # Primeros 10 ejemplos
                f.write(f"{'='*60}\n")
                f.write(f"Ejemplo {i+1}\n")
                f.write(f"{'='*60}\n")
                f.write(f"REFERENCE:\n{result['reference']}\n\n")
                f.write(f"GENERATED:\n{result['candidate']}\n\n")
                f.write(f"ROUGE-1: {result['rouge1_f1']:.4f} | ")
                f.write(f"ROUGE-2: {result['rouge2_f1']:.4f} | ")
                f.write(f"ROUGE-L: {result['rougeL_f1']:.4f} | ")
                f.write(f"METEOR: {result['meteor']:.4f} | ")
                f.write(f"Copy Rate: {result['copy_rate']:.4f} | ")
                f.write(f"Bigram Overlap: {result['bigram_overlap']:.4f} | ")
                f.write(f"P_Gen: {result['avg_p_gen']:.4f} | ")
                f.write(f"P_Copy: {result['avg_p_copy']:.4f}\n\n")
        
        print(f"✓ Resultados legibles en {txt_path}")


def plot_training_history(output_dir, checkpoint_path):
    """
    Grafica la historia de entrenamiento desde el checkpoint del modelo.
    
    Args:
        output_dir: Directorio donde guardar las gráficas
        checkpoint_path: Ruta al checkpoint del modelo
    """
    if not os.path.exists(checkpoint_path):
        print(f"⚠ No se encontró el checkpoint {checkpoint_path}")
        return
    
    print(f"\nCargando historial de entrenamiento desde {checkpoint_path}...")
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extraer historial del checkpoint
    history = checkpoint.get('train_history', None)
    
    if not history or not history.get('epoch'):
        print("⚠ El historial está vacío o no existe en el checkpoint")
        return
    
    # Extraer datos
    epochs = history['epoch']
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    
    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Train y Val Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Vocab Loss y Coverage Loss
    vocab_losses = history.get('vocab_loss', [])
    coverage_losses = history.get('coverage_loss', [])
    
    if vocab_losses and coverage_losses:
        ax2.plot(epochs, vocab_losses, 'g-', label='Vocab Loss', linewidth=2)
        ax2.plot(epochs, coverage_losses, 'r-', label='Coverage Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Vocab Loss vs Coverage Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada en {plot_path}")
    
    # Mostrar estadísticas
    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS DE ENTRENAMIENTO")
    print(f"{'='*60}")
    print(f"Épocas completadas: {len(epochs)}")
    print(f"Mejor Train Loss: {min(train_losses):.4f} (Epoch {epochs[train_losses.index(min(train_losses))]})")
    print(f"Mejor Val Loss: {min(val_losses):.4f} (Epoch {epochs[val_losses.index(min(val_losses))]})")
    print(f"Última Train Loss: {train_losses[-1]:.4f}")
    print(f"Última Val Loss: {val_losses[-1]:.4f}")
    
    if vocab_losses and coverage_losses:
        print(f"Última Vocab Loss: {vocab_losses[-1]:.4f}")
        print(f"Última Coverage Loss: {coverage_losses[-1]:.4f}")
    
    print(f"{'='*60}\n")
    
    plt.close()


def main():
    """
    Función principal para evaluar el modelo.
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
        coverage_lambda=COV_LOSS_LAMBDA,
        dropout_ratio=DROPOUT_RATIO,
        bidirectional=BIDIRECTIONAL,
        device=DEVICE,
        decoding_strategy=DECODING_STRATEGY,
        beam_size=BEAM_SIZE,
        gpu_id=GPU_ID
    )
    
    # 3. Dataset de test
    print("\nCargando dataset de test...")
    test_dataset = PGNDataset(
        vocab=vocab,
        MAX_LEN_SRC=config['src_len'],
        MAX_LEN_TGT=config['tgt_len'],
        data_dir=DATA_DIR,
        split='test'
    )
    
    print(f"✓ Test: {len(test_dataset)} ejemplos")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=pgn_collate_fn,
        num_workers=0
    )
    
    # 4. Ruta del modelo
    model_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pt')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_last.pt')
    
    if not os.path.exists(model_path):
        print(f"⚠ No se encontró ningún checkpoint en {CHECKPOINT_DIR}")
        return
    
    output_dir = GENERATED_TEXT_DIR
    
    # 5. Graficar pérdida de entrenamiento (desde el checkpoint)
    plot_training_history(output_dir, model_path)
    
    # 6. Evaluar
    evaluator = Evaluator(config, vocab, model_path)
    
    metrics, results = evaluator.evaluate(
        test_loader,
        num_examples=None  # None = todos
    )
    
    # 7. Mostrar resultados
    print(f"\n{'='*60}")
    print(f"RESULTADOS DE EVALUACIÓN")
    print(f"{'='*60}")
    print(f"Test Loss:   {metrics['test_loss']:.4f}")
    print(f"ROUGE-1 F1:  {metrics['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1:  {metrics['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1:  {metrics['rougeL_f1']:.4f}")
    print(f"METEOR:      {metrics['meteor']:.4f}")
    print(f"Ejemplos:    {metrics['num_examples']}")
    print(f"{'='*60}\n")
    
    # 8. Guardar resultados
    evaluator.save_results(metrics, results, output_dir)
    
    # 9. Mostrar algunos ejemplos
    print(f"\n{'='*60}")
    print("EJEMPLOS DE RESÚMENES GENERADOS")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results[:3]):
        print(f"Ejemplo {i+1}:")
        print(f"REFERENCE: {result['reference'][:150]}...")
        print(f"GENERATED: {result['candidate'][:150]}...")
        print(f"ROUGE-1: {result['rouge1_f1']:.4f} | "
              f"ROUGE-2: {result['rouge2_f1']:.4f} | "
              f"ROUGE-L: {result['rougeL_f1']:.4f} | "
              f"METEOR: {result['meteor']:.4f} | "
              f"P_Gen: {result['avg_p_gen']:.4f}")
        print()


if __name__ == "__main__":
    main()
