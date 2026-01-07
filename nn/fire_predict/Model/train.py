import matplotlib.pyplot as plt
import torch
import numpy as np # Para convertir a numpy para matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

def gaussian_blur(target, kernel_size=5, sigma=1.0):
    """Crea una zona de probabilidad alrededor del píxel de fuego."""
    # Forzamos a que sea 4D: [Batch, Channel, H, W]
    if target.dim() == 2: # [256, 256] -> [1, 1, 256, 256]
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 3: # [C, H, W] -> [1, C, H, W]
        target = target.unsqueeze(0)
    
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    
    # El kernel debe tener forma: [out_channels, in_channels, kH, kW]
    kernel_4d = kernel_2d.expand(target.shape[1], 1, kernel_size, kernel_size).to(target.device)
    
    return F.conv2d(target, kernel_4d, padding=kernel_size//2)
def calculate_metrics_zone(preds, targets, threshold=0.5, tolerance_px=3):
    """
    Calcula métricas considerando que acertar cerca del fuego es un acierto (TP).
    """
    preds_bin = (preds > threshold).float()
    
    # Dilatamos el target: si el fuego real está a 'tolerance_px', se considera zona de impacto
    kernel_size = 2 * tolerance_px + 1
    target_zone = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=tolerance_px)
    
    tp = (preds_bin * target_zone).sum()
    fp = (preds_bin * (1 - target_zone)).sum()
    # El recall se mide sobre el target original para ser estrictos
    fn = ((1 - preds_bin) * targets).sum() 
    
    return tp, fp, fn
class SoftZoneLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, dice_weight=0.8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight

    def forward(self, logits, soft_targets):
        probs = torch.sigmoid(logits)
        
        # Focal Loss para targets continuos
        pos_weight = torch.tensor([self.alpha / (1 - self.alpha)]).to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, soft_targets, pos_weight=pos_weight, reduction='none')
        
        # pt es la cercanía al target; si el target es 0.8, queremos que la prob sea 0.8
        pt = torch.where(soft_targets > 0.5, probs, 1 - probs)
        focal_loss = ((1 - pt) ** self.gamma * bce).mean()

        # Soft Dice Loss (comparación de áreas)
        intersection = (probs * soft_targets).sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (probs.sum() + soft_targets.sum() + 1e-6)

        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
    
def train_epoch(model, loader, optimizer, scaler, device,
                criterion, pred_horizon=1, accum_steps=1):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)
        targets = batch['label'].float().to(device)
        
        input_seq = x[:, :-pred_horizon]
        target_bin = targets[:, -1].unsqueeze(1) 
        
        target_soft = gaussian_blur(target_bin, kernel_size=15, sigma=3.0)
        # Normalizamos para que el máximo sea 1.0
        target_soft = target_soft / (target_soft.max() + 1e-6)

        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            logits = model(input_seq)
            # La loss ahora compara la predicción con la zona suave
            loss = criterion(logits, target_soft)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})

    return total_loss / len(loader)
@torch.no_grad()
def validate_zonal(model, loader, device, criterion, pred_horizon=1, tolerance_px=3, epoch=0, num_display_images=3):
    model.eval()
    total_loss = 0.0
    g_tp, g_fp, g_fn = 0, 0, 0
    
    # Para guardar ejemplos visuales
    display_count = 0 

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Validation Epoch {epoch+1}")):
        x = batch['x'].to(device)
        targets = batch['label'].float().to(device)
        input_seq = x[:, :-pred_horizon]
        target_bin = targets[:, -1].unsqueeze(1) # [B, 1, H, W]

        logits = model(input_seq)
        probs = torch.sigmoid(logits) # Esto es la predicción del modelo (mancha de riesgo)
        
        # Calculamos loss contra el suave para ser consistentes
        t_soft = gaussian_blur(target_bin, kernel_size=11, sigma=2.0)
        # Normalizar para que el pico de la zona sea 1.0
        max_val_target = t_soft.max()
        if max_val_target > 0:
            t_soft = t_soft / max_val_target
        loss = criterion(logits, t_soft)
        total_loss += loss.item()

        # --- LÓGICA DE VISUALIZACIÓN ---
        if display_count < num_display_images:
            # Buscar una muestra con fuego para visualizar
            # target_bin.sum(dim=[1,2,3]) nos da la suma de pixeles de fuego en cada batch
            fire_indices = (target_bin.sum(dim=[1,2,3]) > 0).nonzero(as_tuple=True)[0]
            
            if fire_indices.numel() > 0: # Si hay al menos un incendio en este batch
                # Tomar el primer incendio para visualizar
                sample_idx = fire_indices[0].item()
                
                original_target_np = target_bin[sample_idx, 0].cpu().numpy()
                soft_target_np = t_soft[sample_idx, 0].cpu().numpy()
                prediction_np = probs[sample_idx, 0].cpu().numpy()

                plt.figure(figsize=(15, 5))
                plt.suptitle(f"Epoch {epoch+1} - Sample {batch_idx*loader.batch_size + sample_idx + 1}")

                plt.subplot(1, 3, 1)
                plt.imshow(original_target_np, cmap='hot', vmin=0, vmax=1)
                plt.title("Target Binario (Fuego Real)")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(soft_target_np, cmap='hot', vmin=0, vmax=1)
                plt.title("Target Suavizado (Zona de Riesgo)")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(prediction_np, cmap='hot', vmin=0, vmax=1)
                plt.title("Predicción (Probabilidad de Zona)")
                plt.axis('off')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
                display_count += 1
        
        # --- MÉTRICAS CON TOLERANCIA ---
        preds_bin = (probs > 0.5).float() # Binarizamos la predicción para métricas
        target_dilated = F.max_pool2d(target_bin, kernel_size=2*tolerance_px+1, stride=1, padding=tolerance_px)
        
        g_tp += (preds_bin * target_dilated).sum().item()
        g_fp += (preds_bin * (1 - target_dilated)).sum().item()
        g_fn += ((1 - F.max_pool2d(preds_bin, 3, 1, 1)) * target_bin).sum().item()

    precision = g_tp / (g_tp + g_fp + 1e-6)
    recall = g_tp / (g_tp + g_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print(f"\n[VAL ZONAL] F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} (Tol: {tolerance_px}px)")
    return total_loss / len(loader), f1