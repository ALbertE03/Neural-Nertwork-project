
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention
class Decoder(nn.Module):
    """
    Decoder LSTM con Pointer-Generator Network y Coverage Mechanism.
    """
    
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_dec_layers,
                 dropout_ratio,
                 is_attention=True,
                 is_pgen=True,
                 is_coverage=True):
        """
        Args:
            vocab_size: Tamaño del vocabulario base (sin OOVs)
            embedding_size: Dimensión de los embeddings
            hidden_size: Dimensión del hidden state del LSTM
            num_dec_layers: Número de capas del LSTM (debe ser 1 para PGN)
            dropout_ratio: Probabilidad de dropout
            is_attention: Si se usa atención
            is_pgen: Si se usa pointer-generator
            is_coverage: Si se usa coverage mechanism
        """
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.is_attention = is_attention
        self.is_pgen = is_pgen
        self.is_coverage = is_coverage
        self.dropout = nn.Dropout(dropout_ratio)
        # Embedding layer (solo para vocabulario base)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        
        # LSTM decoder
        # Input: embedding + context vector (si hay atención)
        lstm_input_size = embedding_size + (hidden_size * 2 if is_attention else 0)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_dec_layers,
            batch_first=True,
            dropout=dropout_ratio if num_dec_layers > 1 else 0
        )
        
        # Attention mechanism
        if is_attention:
            self.attention = Attention(hidden_size, is_coverage=is_coverage)
        
        # Proyección para generar distribución de vocabulario
        # Input: decoder state + context vector
        vocab_input_size = hidden_size + (hidden_size * 2 if is_attention else 0)
        self.vocab_proj = nn.Linear(vocab_input_size, vocab_size)
        
        # Pointer-Generator: calcular p_gen
        if is_pgen:
            # p_gen depende de: context vector, decoder state, decoder input
            pgen_input_size = (hidden_size * 2) + hidden_size + embedding_size
            self.p_gen_linear = nn.Linear(pgen_input_size, 1)
    
    def forward(self, decoder_input, decoder_state, encoder_outputs, encoder_mask, 
                extended_encoder_input, context_vector=None, coverage=None):
        """
        Un paso de decodificación.
        
        Args:
            decoder_input: (batch_size,) - Token actual (ID del vocabulario base)
            decoder_state: Tuple (h, c) donde cada uno es (1, batch_size, hidden_size)
            encoder_outputs: (batch_size, src_len, hidden_size * 2)
            encoder_mask: (batch_size, src_len)
            extended_encoder_input: (batch_size, src_len) - IDs extendidos del source
            context_vector: (batch_size, hidden_size * 2) - Context vector del paso anterior
            coverage: (batch_size, src_len) - Coverage acumulado
            
        Returns:
            final_dist: (batch_size, extended_vocab_size) - Distribución final sobre vocab extendido
            decoder_state: Tuple (h, c) actualizado
            context_vector: (batch_size, hidden_size * 2) - Nuevo context vector
            attention_dist: (batch_size, src_len) - Distribución de atención
            p_gen: (batch_size, 1) - Probabilidad de generar (si is_pgen=True)
            coverage: (batch_size, src_len) - Coverage actualizado
        """
        batch_size = decoder_input.size(0)
        
        # 1. Embeddings del input
        embedded = self.embedding(decoder_input)  # (batch_size, embedding_size)
        embedded = self.dropout(embedded)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, embedding_size)
        
        # 2. Concatenar con context vector si hay atención
        if self.is_attention and context_vector is not None:
            lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)
        else:
            lstm_input = embedded
        
        # 3. LSTM step
        lstm_output, decoder_state = self.lstm(lstm_input, decoder_state)
        # lstm_output: (batch_size, 1, hidden_size)
        lstm_output = lstm_output.squeeze(1)  # (batch_size, hidden_size)
        
        # 4. Calcular atención
        if self.is_attention:
            context_vector, attention_dist, coverage = self.attention(
                lstm_output, encoder_outputs, encoder_mask, coverage
            )
            # context_vector: (batch_size, hidden_size * 2)
            # attention_dist: (batch_size, src_len)
        else:
            attention_dist = None
        
        # 5. Generar distribución de vocabulario
        if self.is_attention:
            vocab_input = torch.cat([lstm_output, context_vector], dim=1)
        else:
            vocab_input = lstm_output
        
        vocab_logits = self.vocab_proj(vocab_input)  # (batch_size, vocab_size)
        vocab_dist = F.softmax(vocab_logits, dim=1)  # (batch_size, vocab_size)
        
        # 6. Pointer-Generator mechanism
        p_gen = None
        if self.is_pgen and self.is_attention:
            # Calcular p_gen
            pgen_input = torch.cat([
                context_vector,           # (batch_size, hidden_size * 2)
                lstm_output,              # (batch_size, hidden_size)
                embedded.squeeze(1)       # (batch_size, embedding_size)
            ], dim=1)
            
            p_gen = torch.sigmoid(self.p_gen_linear(pgen_input))  # (batch_size, 1)
            
            # Combinar vocab_dist y attention_dist
            # final_dist = p_gen * vocab_dist + (1 - p_gen) * copy_dist
            
            # Crear distribución extendida
            src_len = extended_encoder_input.size(1)
            extended_vocab_size = self.vocab_size + src_len  # Aproximación conservadora
            
            # Inicializar distribución extendida
            final_dist = torch.zeros(batch_size, extended_vocab_size, device=vocab_dist.device)
            
            # Añadir vocab_dist ponderado por p_gen
            final_dist[:, :self.vocab_size] = p_gen * vocab_dist
            
            # Añadir attention_dist ponderado por (1 - p_gen) usando scatter_add
            # Copiar de las posiciones del source
            attention_weighted = (1 - p_gen) * attention_dist  # (batch_size, src_len)
            
            # scatter_add para acumular probabilidades en posiciones extendidas
            final_dist.scatter_add_(
                dim=1,
                index=extended_encoder_input,
                src=attention_weighted
            )
        else:
            final_dist = vocab_dist
        
        return final_dist, decoder_state, context_vector, attention_dist, p_gen, coverage