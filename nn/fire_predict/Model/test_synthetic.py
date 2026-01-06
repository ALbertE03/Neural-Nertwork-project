import torch
from model import UNet3D
from constants import INPUT_CHANNELS, HIDDEN_CHANNELS, DROPOUT

if __name__ == "__main__":
    # Instanciar el modelo
    model = UNet3D(in_channels=28, out_channels=1).to('cpu')
    
    
    # Contar parámetros totales
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros: {total_params}")
    print(f"Parámetros entrenables: {trainable_params}")
