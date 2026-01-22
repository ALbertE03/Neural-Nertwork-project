import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder bidireccional LSTM para el modelo Pointer-Generator.
    """
    
    def __init__(self, vocab_size, embedding_size, hidden_size, num_enc_layers, dropout_ratio, bidirectional, pretrained_weights=None):
        """
        Args:
            vocab_size: Tamaño del vocabulario base (sin OOVs)
            embedding_size: Dimensión de los embeddings
            hidden_size: Dimensión del hidden state del LSTM
            num_enc_layers: Número de capas del LSTM
            dropout_ratio: Probabilidad de dropout
            bidirectional: Si el LSTM es bidireccional
            pretrained_weights: Tensor con pesos pre-entrenados (opcional)
        """
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_enc_layers = num_enc_layers
        self.bidirectional = bidirectional
        
        # Embedding layer (solo para vocabulario base)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = False 
            print("✓ Encoder: Pesos de embedding inicializados (no Entrenables).")
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_enc_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_ratio if num_enc_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_ratio)
        # Proyección para reducir hidden state bidireccional
        if bidirectional:
            self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
            self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, encoder_input, encoder_length):
        """
        Args:
            encoder_input: (batch_size, src_len) - IDs del vocabulario base (OOVs → UNK)
            encoder_length: (batch_size,) - Longitudes reales de cada secuencia
            
        Returns:
            encoder_outputs: (batch_size, src_len, hidden_size * 2) si bidirectional
            hidden: Tuple (h_n, c_n) reducidos a (batch_size, hidden_size)
        """
        batch_size, src_len = encoder_input.size()
        
        # 1. Embeddings
        embedded = self.embedding(encoder_input)  # (batch_size, src_len, embedding_size)
        embedded = self.dropout(embedded)
        # 2. Pack padded sequence para eficiencia
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            encoder_length.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 3. LSTM
        packed_outputs, (h_n, c_n) = self.lstm(packed)
        
        # 4. Unpack
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, 
            batch_first=True,
            total_length=src_len
        )
        # encoder_outputs: (batch_size, src_len, hidden_size * 2)
        
        # 5. Reducir hidden states si es bidireccional
        if self.bidirectional:
            # h_n: (num_layers * 2, batch_size, hidden_size)
            # Tomar última capa: (2, batch_size, hidden_size)
            h_n = h_n[-2:]  # [forward, backward] de la última capa
            c_n = c_n[-2:]
            
            # Concatenar forward y backward
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch_size, hidden_size * 2)
            c_n = torch.cat([c_n[0], c_n[1]], dim=1)
            
            # Reducir a hidden_size
            h_n = torch.relu(self.reduce_h(h_n))  # (batch_size, hidden_size)
            c_n = torch.relu(self.reduce_c(c_n))  # (batch_size, hidden_size)
            
            # Añadir dimensión de capas
            h_n = h_n.unsqueeze(0)  # (1, batch_size, hidden_size)
            c_n = c_n.unsqueeze(0)
        
        return encoder_outputs, (h_n, c_n)