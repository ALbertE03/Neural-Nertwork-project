import torch
import torch.nn as nn
import numpy as np

class UNet3D(nn.Module):
    def __init__(self, in_channels=28, out_channels=1, base_features=32):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, base_features)
        self.enc2 = conv_block(base_features, base_features * 2)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) # Reducimos H, W pero NO el tiempo

        # Bottleneck
        self.bottleneck = conv_block(base_features * 2, base_features * 4)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.dec2 = conv_block(base_features * 4 + base_features * 2, base_features * 2)
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.dec1 = conv_block(base_features * 2 + base_features, base_features)

        self.final = nn.Conv3d(base_features, out_channels, kernel_size=1)
        
        # Bias inicial para evitar que prediga basura al inicio
        prior_prob = 0.01
        nn.init.constant_(self.final.bias, -np.log((1 - prior_prob) / prior_prob))

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1) # Skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # Skip connection
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out[:, :, -1, :, :] # Salida para el tiempo T+1