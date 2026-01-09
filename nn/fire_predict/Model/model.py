import torch.nn as nn
import torch
    
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
    def __init__(self, in_channels=28, out_channels=1, dropout=0.3,enc1_channels=32,enc2_channels=64,bottleneck_channels=256):
        super().__init__()

    
        self.enc1_channels = enc1_channels
        self.enc2_channels = enc2_channels
        self.bottleneck_channels = 256

        def conv_block(in_c, out_c, norm_c=None):
            # norm_c: canales para la segunda BatchNorm (opcional)
            layers = [
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(norm_c if norm_c is not None else out_c),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout)
            ]
            return nn.Sequential(*layers)

        # Encoder
        self.enc1 = conv_block(in_channels, self.enc1_channels)
        self.enc2 = conv_block(self.enc1_channels, self.enc2_channels)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) # Reduce H, W but NOT time

        # Bottleneck
        self.bottleneck = conv_block(self.enc2_channels, self.bottleneck_channels)

        # Decoder
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.att2 = AttentionBlock3D(F_g=self.bottleneck_channels, F_l=self.enc2_channels, F_int=self.enc2_channels)
        self.dec2 = conv_block(self.bottleneck_channels + self.enc2_channels, self.enc2_channels)

        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.att1 = AttentionBlock3D(F_g=self.enc2_channels, F_l=self.enc1_channels, F_int=self.enc1_channels)
        self.dec1 = conv_block(self.enc2_channels + self.enc1_channels, self.enc1_channels)
        self.final = nn.Conv3d(self.enc1_channels, out_channels, kernel_size=1)
        
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



class FireModelROI(nn.Module):
    def __init__(self, unet_3d):
        super().__init__()
        self.unet = unet_3d

    def forward(self, x):
        # (B, R, T, C, H, W) 
        b, r, t, c, h, w = x.shape
        
        # Nueva forma: (B*R, T, C, H, W) 
        x = x.view(b * r, t, c, h, w)
        

        out = self.unet(x) #(B*R, 1, H, W)
        
        # Final: (B, R, 1, H, W) 
        out = out.view(b, r, 1, h, w)
        return out