import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Mecanismo de atención de Bahdanau para el modelo Pointer-Generator.
    Calcula la distribución de atención sobre los encoder outputs.
    """
    
    def __init__(self, hidden_size, is_coverage=False):
        """
        Args:
            hidden_size: Dimensión del hidden state
            is_coverage: Si se usa coverage mechanism
        """
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.is_coverage = is_coverage
        
        # Proyecciones para calcular attention scores
        # encoder_outputs: (batch, src_len, hidden_size * 2) 
        self.W_h = nn.Linear(hidden_size * 2, hidden_size, bias=False)  # Para encoder outputs
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=True)       # Para decoder state
        
        # Coverage feature
        if is_coverage:
            self.W_c = nn.Linear(1, hidden_size, bias=False)
        
        # Proyección final
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_state, encoder_outputs, encoder_mask, coverage=None):
        """
        Args:
            decoder_state: (batch_size, hidden_size) - Estado actual del decoder
            encoder_outputs: (batch_size, src_len, hidden_size * 2) - Outputs del encoder
            encoder_mask: (batch_size, src_len) - Máscara de padding (1 = válido, 0 = padding)
            coverage: (batch_size, src_len) - Vector de coverage acumulado (opcional)
            
        Returns:
            context_vector: (batch_size, hidden_size * 2) - Vector de contexto
            attention_dist: (batch_size, src_len) - Distribución de atención
            coverage: (batch_size, src_len) - Coverage actualizado (si is_coverage=True)
        """
        batch_size, src_len, _ = encoder_outputs.size()
        
        #  Proyectar encoder outputs
        encoder_features = self.W_h(encoder_outputs)  # (batch_size, src_len, hidden_size)
        
        #  Proyectar decoder state y expandir
        decoder_features = self.W_s(decoder_state)  # (batch_size, hidden_size)
        decoder_features = decoder_features.unsqueeze(1)  # (batch_size, 1, hidden_size)
        decoder_features = decoder_features.expand(-1, src_len, -1)  # (batch_size, src_len, hidden_size)
        
        #  Calcular attention scores
        attention_features = encoder_features + decoder_features  # (batch_size, src_len, hidden_size)
        
        #  Añadir coverage si está activado
        if self.is_coverage and coverage is not None:
            coverage_features = self.W_c(coverage.unsqueeze(2))  # (batch_size, src_len, hidden_size)
            attention_features = attention_features + coverage_features
        
        #  Calcular scores
        e = self.v(torch.tanh(attention_features))  # (batch_size, src_len, 1)
        e = e.squeeze(2)  # (batch_size, src_len)
        
        # Aplicar máscara (hacer -inf los padding para que softmax → 0)
        e = e.masked_fill(encoder_mask == 0, -1e4)
        
        # Softmax para obtener distribución de atención
        attention_dist = F.softmax(e, dim=1)  # (batch_size, src_len)
        
        # Calcular context vector
        attention_dist_expanded = attention_dist.unsqueeze(1)  # (batch_size, 1, src_len)
        context_vector = torch.bmm(attention_dist_expanded, encoder_outputs)  # (batch_size, 1, hidden_size * 2)
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size * 2)
        
        # Actualizar coverage
        if self.is_coverage:
            if coverage is None:
                coverage = attention_dist
            else:
                coverage = coverage + attention_dist
        
        return context_vector, attention_dist, coverage
