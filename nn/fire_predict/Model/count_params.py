import torch
from model import FireModelROI,UNet3D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = UNet3D(in_channels=28, out_channels=1).to(device)
    model = FireModelROI(c).to(device)
    total_params = count_parameters(model)
    print(f"Total: {total_params:,}")
