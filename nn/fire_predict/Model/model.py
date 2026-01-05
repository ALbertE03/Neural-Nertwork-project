import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Bahdanau Attention (Additive)
    Score = v^T * tanh(W_q(Q) + W_k(K))
    Diseñada para ser ligera en memoria.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        # Usamos pocos canales para la atención para ahorrar memoria
        self.att_channels = max(1, hidden_channels // 4)
        
        self.query_conv = nn.Conv2d(hidden_channels, self.att_channels, 1)
        self.key_conv = nn.Conv2d(hidden_channels, self.att_channels, 1)
        self.score_conv = nn.Conv2d(self.att_channels, 1, 1)
        
        self.value_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, query, keys_values):
        """
        Args:
            query: (B, C, H, W) - estado actual
            keys_values: (B, T, C, H, W) - estados históricos
        Returns:
            attended: (B, C, H, W) - contexto atendido
        """
        B, C, H, W = query.shape
        T = keys_values.shape[1]
        
        # Proyectar Query
        # (B, C_att, H, W) -> (B, 1, C_att, H, W)
        Q_proj = self.query_conv(query).unsqueeze(1)
        
        # Proyectar Keys (procesar todo el batch*tiempo junto para eficiencia)
        # (B*T, C, H, W)
        keys_flat = keys_values.view(B*T, C, H, W)
        K_proj = self.key_conv(keys_flat) # (B*T, C_att, H, W)
        K_proj = K_proj.view(B, T, -1, H, W) # (B, T, C_att, H, W)
        
        # Calcular Scores (Additive)
        # tanh(Q + K)
        # Broadcasting Q sobre T
        energy = torch.tanh(Q_proj + K_proj) # (B, T, C_att, H, W)
        
        # v^T * energy
        # (B*T, C_att, H, W)
        energy_flat = energy.view(B*T, -1, H, W)
        scores = self.score_conv(energy_flat) # (B*T, 1, H, W)
        scores = scores.view(B, T, H, W) # (B, T, H, W)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=1) # (B, T, H, W)
        
        # Weighted Sum
        # Proyectar Values
        V_proj = self.value_conv(keys_flat).view(B, T, C, H, W)
        
        # (B, T, 1, H, W) * (B, T, C, H, W) -> Sum over T
        context = (attn_weights.unsqueeze(2) * V_proj).sum(dim=1)
        
        out = self.gamma * context + query
        return out


class ConvLSTMCell(nn.Module):
    """Celda ConvLSTM con BatchNorm"""
    def __init__(self, input_channels, hidden_channels, kernel_size=3, use_bn=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_bn = use_bn
        padding = kernel_size // 2
        
        # Gates: input, forget, cell, output (4x hidden)
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        if use_bn:
            self.bn = nn.BatchNorm2d(4 * hidden_channels)
    
    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        if self.use_bn:
            gates = self.bn(gates)
        
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
    
    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


class FirePredictModel(nn.Module):
    """
    Modelo ConvLSTM con Attention para predecir incendios.
    
    Input: Secuencia temporal (B, T, C, H, W)
    Output: pred_horizon máscaras futuras (B, pred_horizon, H, W)
    """
    def __init__(self, input_channels=10, hidden_channels=32, dropout=0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        
        # Encoder: 1 capa ConvLSTM (Reducido de 2 para ahorrar memoria)
        self.encoder1 = ConvLSTMCell(input_channels, hidden_channels, use_bn=True)
        
        self.dropout = nn.Dropout2d(dropout)
        
        # Attention para acceder al contexto histórico
        self.attention = SpatialAttention(hidden_channels)
        
        # Decoder que combina estado actual + contexto atendido
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),  # *2 por concat
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, 1),  # 1 canal = 1 día
        )
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - secuencia temporal histórica
        Returns:
            pred: (B, H, W) - predicción día+1
            
        Estrategia:
        - Predice un solo día después usando la secuencia histórica.
        """
        B, T, C, H, W = x.shape
        device = x.device
        
        # Inicializar hidden states
        state1 = self.encoder1.init_hidden(B, H, W, device)
        
        # Procesar secuencia histórica y guardar estados
        historical_states = []
        for t in range(T):
            h1, c1 = self.encoder1(x[:, t], state1)
            h1 = self.dropout(h1)
            state1 = (h1, c1)
            
            historical_states.append(h1)
        
        # Stack historical states: (B, T, C, H, W)
        historical_states = torch.stack(historical_states, dim=1)
                
        # Aplicar attention: estado actual consulta a todos los estados históricos
        # h1 es el último estado
        h1_attended = self.attention(h1, historical_states)  # (B, C, H, W)
        
        # Combinar estado actual + contexto atendido
        combined = torch.cat([h1, h1_attended], dim=1)  # (B, 2*C, H, W)
        
        # Decodificar predicción
        pred_logits = self.decoder(combined)  # (B, 1, H, W)
        
        return pred_logits.squeeze(1)  # (B, H, W)