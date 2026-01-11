import torch
from model import build_unet3d as UNet3D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = UNet3D(input_shape=(3, 256, 256, 28), out_channels=1, dropout=0.3)
  
    total_params = count_parameters(c)
    print(f"Total: {total_params:,}")
