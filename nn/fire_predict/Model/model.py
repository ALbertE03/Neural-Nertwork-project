import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Modelo ConvLSTM para predecir incendios 2 días en el futuro.
    Con BatchNorm y Dropout para mejor generalización.
    
    Input: Secuencia temporal (B, T, C, H, W)
    Output: 2 máscaras futuras (B, 2, H, W) → día+1 y día+2
    """
    def __init__(self, input_channels=10, hidden_channels=32, pred_horizon=2, dropout=0.3):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.hidden_channels = hidden_channels
        
        # Encoder: 2 capas ConvLSTM
        self.encoder1 = ConvLSTMCell(input_channels, hidden_channels, use_bn=True)
        self.encoder2 = ConvLSTMCell(hidden_channels, hidden_channels, use_bn=True)
        
        self.dropout = nn.Dropout2d(dropout)
        
        # Decoder con BatchNorm
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, pred_horizon, 1),  # 2 canales = 2 días
            nn.Sigmoid()  # Probabilidad de fuego [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - secuencia temporal
        Returns:
            pred: (B, 2, H, W) - predicción día+1 y día+2
        """
        B, T, C, H, W = x.shape
        device = x.device
        
        # Inicializar hidden states
        state1 = self.encoder1.init_hidden(B, H, W, device)
        state2 = self.encoder2.init_hidden(B, H, W, device)
        
        # Procesar secuencia completa
        for t in range(T):
            h1, c1 = self.encoder1(x[:, t], state1)
            h1 = self.dropout(h1)
            state1 = (h1, c1)
            
            h2, c2 = self.encoder2(h1, state2)
            h2 = self.dropout(h2)
            state2 = (h2, c2)
        
        # Decodificar a 2 predicciones futuras
        pred = self.decoder(h2)  # (B, 2, H, W)
        
        return pred