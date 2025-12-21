import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json

from pgn_model import PointerGeneratorNetwork
from beam_search import BeamSearch
from dataset import PGNDataset, pgn_collate_fn
from vocabulary import Vocabulary
from config import Config
from constant import *

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
        
        # Cargar modelo
        print(f"Cargando modelo desde {model_path}")
        self.model = PointerGeneratorNetwork(config, vocab).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Modelo cargado (Epoch {checkpoint['epoch']})")
        
        # Beam search
        self.beam_search = BeamSearch(
            self.model,
            vocab,
            beam_size=config['beam_size'],
            max_len=config['tgt_len']
        )
    
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
                        hypothesis = self.beam_search.search(single_batch)
                        generated_ids = hypothesis.tokens[1:]  # Quitar START
                    else:  # greedy
                        generated_ids = self.model.decode_greedy(single_batch, max_len=self.config['tgt_len'])
                        generated_ids = generated_ids[0].cpu().tolist()
                    
                    # Decodificar a texto
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


def main():
    """
    Función principal para generar resúmenes.
    """
    # 1. Cargar vocabulario
    print("Cargando vocabulario...")
    vocab = Vocabulary(
        CREATE_VOCABULARY=False,  # No crear, solo cargar
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
        batch_size=1,  # Procesar de uno en uno para beam search
        shuffle=False,
        collate_fn=pgn_collate_fn,
        num_workers=0
    )
    
    # 4. Ruta del modelo
    # Usar el mejor modelo por defecto
    model_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pt')
    
    # Si no existe, buscar el último
    if not os.path.exists(model_path):
        model_path = os.path.join(CHECKPOINT_DIR, 'checkpoint_last.pt')
    
    if not os.path.exists(model_path):
        print(f"⚠ No se encontró ningún checkpoint en {CHECKPOINT_DIR}")
        return
    
    # 5. Generar
    generator = Generator(config, vocab, model_path)
    
    # Generar todos los ejemplos (o especificar un número)
    output_file = os.path.join(GENERATED_TEXT_DIR, 'test_results.json')
    
    results = generator.generate(
        test_loader,
        output_file=output_file,
        num_examples=None  # None = todos, o especificar un número
    )
    
    # Mostrar algunos ejemplos
    print(f"\n{'='*60}")
    print("Ejemplos generados:")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results[:3]):  # Mostrar primeros 3
        print(f"Ejemplo {i+1}:")
        print(f"TARGET: {result['target'][:100]}...")
        print(f"GENERATED: {result['generated'][:100]}...")
        print()


if __name__ == "__main__":
    main()
