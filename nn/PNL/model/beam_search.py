import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

class Hypothesis:
    """
    Representa una hipótesis durante beam search.
    """
    def __init__(self, tokens, log_probs, decoder_state, context_vector, coverage, attention_dists=None, p_gens=None):
        """
        Args:
            tokens: List[int] - Secuencia de tokens generados
            log_probs: List[float] - Log probabilities de cada token
            decoder_state: Tuple (h, c) - Estado del decoder
            context_vector: Tensor - Context vector actual
            coverage: Tensor - Coverage acumulado
            attention_dists: List[Tensor] - Lista de attention dists para cada paso
            p_gens: List[Tensor] - Lista de p_gen para cada paso
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_state = decoder_state
        self.context_vector = context_vector
        self.coverage = coverage
        self.attention_dists = attention_dists if attention_dists is not None else []
        self.p_gens = p_gens if p_gens is not None else []
    
    def extend(self, token, log_prob, decoder_state, context_vector, coverage, attention_dist, p_gen):
        """
        Extiende la hipótesis con un nuevo token.
        
        Returns:
            Nueva Hypothesis
        """
        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            decoder_state=decoder_state,
            context_vector=context_vector,
            coverage=coverage,
            attention_dists=self.attention_dists + [attention_dist],
            p_gens=self.p_gens + [p_gen]
        )
    
    @property
    def avg_log_prob(self):
        """Promedio de log probabilities (para ranking)."""
        return sum(self.log_probs) / len(self.tokens)
    
    @property
    def latest_token(self):
        """Último token generado."""
        return self.tokens[-1]


class BeamSearch:
    """
    Implementa Beam Search para decodificación.
    """
    
    def __init__(self, model, vocab, beam_size=4, max_len=50, min_len=10):
        """
        Args:
            model: PointerGeneratorNetwork
            vocab: Vocabulary object
            beam_size: Tamaño del beam
            max_len: Longitud máxima de generación
            min_len: Longitud mínima (penaliza secuencias cortas)
        """
        self.model = model
        self.vocab = vocab
        self.beam_size = beam_size
        self.max_len = max_len
        self.min_len = min_len
        
        self.start_id = vocab.word2id(vocab.start_decoding)
        self.end_id = vocab.word2id(vocab.end_decoding)
        self.unk_id = vocab.word2id(vocab.unk_token)
        self.vocab_size = len(vocab.word_to_id)
        
        self.device = model.device
    
    def search(self, batch):
        """
        Realiza beam search para un batch (asume batch_size=1).
        
        Args:
            batch: Dict con encoder inputs
            
        Returns:
            best_hypothesis: Hypothesis con la mejor secuencia
        """
        # Asegurar batch_size = 1
        encoder_input = batch['encoder_input'].to(self.device)  # (1, src_len)
        extended_encoder_input = batch['extended_encoder_input'].to(self.device)
        encoder_length = batch['encoder_length'].to(self.device)
        encoder_mask = batch['encoder_mask'].to(self.device)
        
        batch_size = encoder_input.size(0)
        assert batch_size == 1, "Beam search solo soporta batch_size=1"
        
        src_len = encoder_input.size(1)
        
        # 1. Encoder
        encoder_outputs, decoder_state = self.model.encoder(encoder_input, encoder_length)
        # encoder_outputs: (1, src_len, hidden_size * 2)
        
        # 2. Inicializar beam
        initial_context = torch.zeros(1, self.model.config['hidden_size'] * 2, device=self.device)
        initial_coverage = torch.zeros(1, src_len, device=self.device) if self.model.is_coverage else None
        
        # Hipótesis inicial con START token
        initial_hyp = Hypothesis(
            tokens=[self.start_id],
            log_probs=[0.0],
            decoder_state=decoder_state,
            context_vector=initial_context,
            coverage=initial_coverage,
            attention_dists=[],
            p_gens=[]
        )
        
        hypotheses = [initial_hyp]  # Beam actual
        completed = []  # Hipótesis completas (con END)
        
        # 3. Beam search loop
        for step in range(self.max_len):
            if len(completed) >= self.beam_size:
                break
            
            all_candidates = []
            
            # Expandir cada hipótesis en el beam
            for hyp in hypotheses:
                # Si ya terminó, mover a completed
                if hyp.latest_token == self.end_id:
                    completed.append(hyp)
                    continue
                
                # Preparar input para el decoder
                decoder_input = torch.tensor(
                    [hyp.latest_token],
                    dtype=torch.long,
                    device=self.device
                )
                
                # Convertir OOV a UNK para embeddings
                if decoder_input.item() >= self.vocab_size:
                    decoder_input = torch.tensor(
                        [self.unk_id],
                        dtype=torch.long,
                        device=self.device
                    )
                
                # Decoder step
                final_dist, new_decoder_state, new_context, attention_dist, p_gen, new_coverage = self.model.decoder(
                    decoder_input=decoder_input,
                    decoder_state=hyp.decoder_state,
                    encoder_outputs=encoder_outputs,
                    encoder_mask=encoder_mask,
                    extended_encoder_input=extended_encoder_input,
                    context_vector=hyp.context_vector,
                    coverage=hyp.coverage
                )
                
                # Log probabilities
                log_probs = torch.log(final_dist + 1e-12)  # (1, extended_vocab_size)
                log_probs = log_probs.squeeze(0)  # (extended_vocab_size,)
                
                # Penalizar UNK 
                log_probs[self.unk_id] -= 100.0
                
                # Penalizar END si estamos antes de min_len
                if step < self.min_len:
                    log_probs[self.end_id] = -1e20
                
                # --- Trigram Blocking ---
                # Si generar este token crearía un trigrama repetido, penalizarlo
                if len(hyp.tokens) >= 2:
                    current_trigram_prefix = tuple(hyp.tokens[-2:])
                    for i in range(len(hyp.tokens) - 2):
                        if tuple(hyp.tokens[i:i+2]) == current_trigram_prefix:
                            forbidden_token = hyp.tokens[i+2]
                            log_probs[forbidden_token] = -1e20
                # ------------------------

                # Top-k candidatos
                top_k_log_probs, top_k_ids = torch.topk(log_probs, self.beam_size * 2)
                
                # Crear nuevas hipótesis
                for i in range(self.beam_size * 2):
                    token_id = top_k_ids[i].item()
                    token_log_prob = top_k_log_probs[i].item()
                    
                    new_hyp = hyp.extend(
                        token=token_id,
                        log_prob=token_log_prob,
                        decoder_state=new_decoder_state,
                        context_vector=new_context,
                        coverage=new_coverage,
                        attention_dist=attention_dist,
                        p_gen=p_gen if p_gen is not None else torch.tensor([[1.0]], device=self.device)
                    )
                    
                    all_candidates.append(new_hyp)
            
            # Ordenar candidatos por avg_log_prob
            all_candidates.sort(key=lambda h: h.avg_log_prob, reverse=True)
            
            # Seleccionar top beam_size hipótesis
            hypotheses = all_candidates[:self.beam_size]
        
        # 4. Si no hay hipótesis completas, tomar las mejores actuales
        if len(completed) == 0:
            completed = hypotheses
        
        # Ordenar por avg_log_prob and return the sorted list
        completed.sort(key=lambda h: h.avg_log_prob, reverse=True)
        return completed
    
    def decode_batch(self, data_loader, num_examples=None):
        """
        Decodifica múltiples ejemplos usando beam search.
        
        Args:
            data_loader: DataLoader
            num_examples: Número de ejemplos a decodificar (None = todos)
            
        Returns:
            List[List[int]]: Secuencias generadas
        """
        self.model.eval()
        generated_sequences = []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if num_examples is not None and i >= num_examples:
                    break
                
                # Beam search espera batch_size=1
                # Si el batch tiene más de 1, procesar uno por uno
                batch_size = batch['encoder_input'].size(0)
                
                for b in range(batch_size):
                    # Extraer ejemplo individual
                    single_batch = {
                        'encoder_input': batch['encoder_input'][b:b+1],
                        'extended_encoder_input': batch['extended_encoder_input'][b:b+1],
                        'encoder_length': batch['encoder_length'][b:b+1],
                        'encoder_mask': batch['encoder_mask'][b:b+1]
                    }
                    
                    # Beam search
                    hypotheses = self.search(single_batch)
                    best_hyp = hypotheses[0]
                    generated_sequences.append(best_hyp.tokens)
                    
                    if num_examples is not None and len(generated_sequences) >= num_examples:
                        break
        
        return generated_sequences
