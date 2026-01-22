import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
class PointerGeneratorNetwork(nn.Module):
    """
    Pointer-Generator Network con Coverage Mechanism para text summarization.
    
    Referencia: "Get To The Point: Summarization with Pointer-Generator Networks"
    (See et al., 2017) - https://arxiv.org/abs/1704.04368
    """
    
    def __init__(self, config, vocab, pretrained_weights=None):
        """
        Args:
            config: Objeto Config con hiperparámetros
            vocab: Objeto Vocabulary
            pretrained_weights: Tensor con pesos pre-entrenados (opcional)
        """
        super(PointerGeneratorNetwork, self).__init__()
        
        self.config = config
        self.vocab = vocab
        self.vocab_size = config['max_vocab_size']
        
        # Encoder
        self.encoder = Encoder(
            vocab_size=self.vocab_size,
            embedding_size=config['embedding_size'],
            hidden_size=config['hidden_size'],
            num_enc_layers=config['num_enc_layers'],  
            dropout_ratio=config['dropout_ratio'],
            bidirectional=config['bidirectional'],
            pretrained_weights=pretrained_weights
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            embedding_size=config['embedding_size'],
            hidden_size=config['hidden_size'],
            num_dec_layers=config['num_dec_layers'],  
            dropout_ratio=config['dropout_ratio'],
            is_attention=True,
            is_pgen=config['is_pgen'],
            is_coverage=config['is_coverage']
        )
        
        self.is_coverage = config['is_coverage']
        self.coverage_lambda = config['coverage_lambda'] if config['coverage_lambda'] is not None else 1.0
        self.device = config['device']
        
        # Compartir embeddings entre encoder y decoder
        self.decoder.embedding.weight = self.encoder.embedding.weight
    
    def forward(self, batch, is_training=True):
        """
        Forward pass completo para training.
        
        Args:
            batch: Dict con:
                - encoder_input: (batch_size, src_len) - IDs base
                - extended_encoder_input: (batch_size, src_len) - IDs extendidos
                - encoder_length: (batch_size,)
                - encoder_mask: (batch_size, src_len)
                - decoder_input: (batch_size, tgt_len) - IDs base
                - decoder_target: (batch_size, tgt_len) - IDs extendidos
            is_training: Si es modo training (teacher forcing)
            
        Returns:
            Dict con:
                - loss: Scalar tensor
                - vocab_loss: Scalar tensor
                - coverage_loss: Scalar tensor (si is_coverage=True)
                - final_dists: (batch_size, tgt_len, extended_vocab_size)
        """
        # Unpack batch
        encoder_input = batch['encoder_input'].to(self.device)
        extended_encoder_input = batch['extended_encoder_input'].to(self.device)
        encoder_length = batch['encoder_length'].to(self.device)
        encoder_mask = batch['encoder_mask'].to(self.device)
        decoder_input = batch['decoder_input'].to(self.device)
        decoder_target = batch['decoder_target'].to(self.device)
        
        batch_size, tgt_len = decoder_input.size()
        src_len = encoder_input.size(1)
        
        # 1. Encoder
        encoder_outputs, decoder_state = self.encoder(encoder_input, encoder_length)
        # encoder_outputs: (batch_size, src_len, hidden_size * 2)
        # decoder_state: Tuple (h, c) - (1, batch_size, hidden_size)
        
        # 2. Inicializar
        context_vector = torch.zeros(batch_size, self.config['hidden_size'] * 2, device=self.device)
        coverage = torch.zeros(batch_size, src_len, device=self.device) if self.is_coverage else None
        
        # 3. Decoder loop (teacher forcing)
        final_dists = []
        attention_dists = []
        coverages = []
        
        for t in range(tgt_len):
            decoder_input_t = decoder_input[:, t]  # (batch_size,)
            
            final_dist, decoder_state, context_vector, attention_dist, p_gen, coverage = self.decoder(
                decoder_input=decoder_input_t,
                decoder_state=decoder_state,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                extended_encoder_input=extended_encoder_input,
                context_vector=context_vector,
                coverage=coverage
            )
            
            final_dists.append(final_dist)
            attention_dists.append(attention_dist)
            if self.is_coverage:
                coverages.append(coverage)
        
        # Stack outputs
        final_dists = torch.stack(final_dists, dim=1)  # (batch_size, tgt_len, extended_vocab_size)
        attention_dists = torch.stack(attention_dists, dim=1)  # (batch_size, tgt_len, src_len)
        
        # 4. Calcular loss
        vocab_loss = self._calculate_vocab_loss(final_dists, decoder_target)
        
        coverage_loss = torch.tensor(0.0, device=self.device)
        if self.is_coverage and len(coverages) > 0:
            coverages = torch.stack(coverages, dim=1)  # (batch_size, tgt_len, src_len)
            coverage_loss = self._calculate_coverage_loss(attention_dists, coverages, encoder_mask)
        
        # Loss total (aplicar lambda al coverage loss)
        total_loss = vocab_loss + self.coverage_lambda * coverage_loss
        
        return {
            'loss': total_loss,
            'vocab_loss': vocab_loss,
            'coverage_loss': coverage_loss,
            'final_dists': final_dists
        }
    
    def _calculate_vocab_loss(self, final_dists, targets):
        """
        Calcula negative log likelihood loss.
        
        Args:
            final_dists: (batch_size, tgt_len, extended_vocab_size)
            targets: (batch_size, tgt_len) - IDs extendidos
            
        Returns:
            loss: Scalar tensor
        """
        batch_size, tgt_len, _ = final_dists.size()
        
        # Evitar log(0)
        final_dists = final_dists + 1e-12
        
        # Gather las probabilidades de los targets
        targets_expanded = targets.unsqueeze(2)  # (batch_size, tgt_len, 1)
        
        # Clamp targets para evitar índices fuera de rango
        max_idx = final_dists.size(2) - 1
        targets_clamped = torch.clamp(targets_expanded, 0, max_idx)
        
        probs = torch.gather(final_dists, dim=2, index=targets_clamped)  # (batch_size, tgt_len, 1)
        probs = probs.squeeze(2)  # (batch_size, tgt_len)
        
        # Negative log likelihood
        losses = -torch.log(probs)
        
        # Máscara de padding (PAD_ID = 0)
        mask = (targets != 0).float()
        
        # Loss promedio sobre tokens no-padding
        loss = (losses * mask).sum() / mask.sum()
        
        return loss
    
    def _calculate_coverage_loss(self, attention_dists, coverages, encoder_mask):
        """
        Calcula coverage loss para penalizar atención repetida.
        
        Args:
            attention_dists: (batch_size, tgt_len, src_len)
            coverages: (batch_size, tgt_len, src_len) - Coverage en cada paso
            encoder_mask: (batch_size, src_len)
            
        Returns:
            coverage_loss: Scalar tensor
        """
        # Coverage loss = sum_t min(a_t, c_t)
        # Penaliza cuando atendemos posiciones ya atendidas
        
        # Shift coverage: usamos coverage del paso anterior
        coverage_prev = torch.cat([
            torch.zeros_like(coverages[:, :1, :]),  # t=0 no tiene coverage previo
            coverages[:, :-1, :]  # t>0 usa coverage de t-1
        ], dim=1)
        
        # min(attention, coverage_prev)
        min_vals = torch.min(attention_dists, coverage_prev)
        
        # Sumar sobre src_len y tgt_len, aplicar máscara
        encoder_mask_expanded = encoder_mask.unsqueeze(1)  # (batch_size, 1, src_len)
        
        coverage_loss = (min_vals * encoder_mask_expanded.float()).sum()
        
        # Normalizar por número de tokens
        num_tokens = encoder_mask.sum()
        coverage_loss = coverage_loss / num_tokens
        
        return coverage_loss
    
    def decode_greedy(self, batch, max_len=None):
        """
        Decodificación greedy (sin beam search).
        
        Args:
            batch: Dict con encoder inputs
            max_len: Longitud máxima de generación
            
        Returns:
            generated_ids: (batch_size, max_len) - Secuencia generada
        """
        if max_len is None:
            max_len = self.config['tgt_len']
        
        # Unpack
        encoder_input = batch['encoder_input'].to(self.device)
        extended_encoder_input = batch['extended_encoder_input'].to(self.device)
        encoder_length = batch['encoder_length'].to(self.device)
        encoder_mask = batch['encoder_mask'].to(self.device)
        
        batch_size = encoder_input.size(0)
        src_len = encoder_input.size(1)
        
        # Encoder
        encoder_outputs, decoder_state = self.encoder(encoder_input, encoder_length)
        
        # Inicializar
        context_vector = torch.zeros(batch_size, self.config['hidden_size'] * 2, device=self.device)
        coverage = torch.zeros(batch_size, src_len, device=self.device) if self.is_coverage else None
        
        # Start token
        decoder_input = torch.full(
            (batch_size,), 
            self.vocab.word2id(self.vocab.start_decoding),
            dtype=torch.long,
            device=self.device
        )
        
        generated_ids = []
        p_gens = []
        log_probs = []
        attention_dists = []
                
        for t in range(max_len):
            final_dist, decoder_state, context_vector, attention_dist, p_gen, coverage = self.decoder(
                decoder_input=decoder_input,
                decoder_state=decoder_state,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                extended_encoder_input=extended_encoder_input,
                context_vector=context_vector,
                coverage=coverage
            )
            
            # Greedy: seleccionar el token con mayor probabilidad
            probs, predicted_ids = torch.max(final_dist, dim=1)
            log_prob = torch.log(probs + 1e-12)
            
            generated_ids.append(predicted_ids)
            log_probs.append(log_prob)
            attention_dists.append(attention_dist)
            
            # Guardamos p_gen (si es None por no-pgen, guardamos 1.0 = generación pura)
            if p_gen is not None:
                p_gens.append(p_gen)
            else:
                p_gens.append(torch.ones(batch_size, 1, device=self.device))
            
            # Próximo input: convertir OOVs a UNK
            decoder_input = torch.where(
                predicted_ids < self.vocab_size,
                predicted_ids,
                torch.full_like(predicted_ids, self.vocab.word2id(self.vocab.unk_token))
            )
        
        generated_ids = torch.stack(generated_ids, dim=1)  # (batch_size, max_len)
        p_gens = torch.stack(p_gens, dim=1) # (batch_size, max_len, 1)
        log_probs = torch.stack(log_probs, dim=1)
        attention_dists = torch.stack(attention_dists, dim=1) # (batch_size, max_len, src_len)
        return generated_ids, p_gens, log_probs, attention_dists
