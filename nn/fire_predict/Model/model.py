import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock3D(nn.Module):
    """Attention Gate for 3D UNet."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet3D(nn.Module):
    def __init__(self, in_channels=28, out_channels=1, base_features=32, dropout=0.3):
        super().__init__()
        
        def conv_block(in_c, out_c):
            layers = [
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout)
            ]
            return nn.Sequential(*layers)

        # Encoder
        self.enc1 = conv_block(in_channels, base_features)
        self.enc2 = conv_block(base_features, base_features * 2)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) # Reduce H, W but NOT time

        # Bottleneck
        self.bottleneck = conv_block(base_features * 2, base_features * 4)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.att2 = AttentionBlock3D(F_g=base_features * 4, F_l=base_features * 2, F_int=base_features * 2)
        self.dec2 = conv_block(base_features * 4 + base_features * 2, base_features * 2)

        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.att1 = AttentionBlock3D(F_g=base_features * 2, F_l=base_features, F_int=base_features)
        self.dec1 = conv_block(base_features * 2 + base_features, base_features)

        self.final = nn.Conv3d(base_features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        
        d2 = self.up2(b)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2_att], dim=1) # Attention-based skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1_att], dim=1) # Attention-based skip connection
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out[:, :, -1, :, :] # Output for time T