import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from model import UNet3D,FireModelROI
from constants import LEARNING_RATE, EPOCHS

from sklearn.metrics import f1_score, jaccard_score

def train_epoch(model, loader, optimizer, scaler, device, criterion, accum_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader)):
        inputs = batch['x'].to(device)      # (B, R, T, C, H, W)
        targets = batch['label'].to(device) # (B, R, T, H, W)

        inputs = inputs[:,:,:-1,:,:,:] # (B, R, T-1, C, H, W)
        target_last = targets[:, :, -1, :, :].unsqueeze(2) # (B, R, 1, H, W)

        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            preds = model(inputs) # (B, R, 1, H, W)
            loss = criterion(preds, target_last)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
    
    return total_loss / len(loader)




def validate_zonal(model, loader, device, criterion, epoch=0):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            inputs = batch['x'].to(device)
            targets = batch['label'].to(device)
            inputs = inputs[:,:,:-1,:,:,:]  # (B, R, T-1, C, H, W)

            target_last = targets[:, :, -1, :, :].unsqueeze(2) # (B, R, 1, H, W)
            
            preds = model(inputs)
            loss = criterion(preds, target_last)
            total_loss += loss.item()

            # Procesamiento para métricas
            preds_prob = torch.sigmoid(preds)
            preds_bin = (preds_prob > 0.5).cpu().numpy()
            targets_np = (target_last > 0.5).cpu().numpy()

            # Guardamos para el cálculo global final
            all_preds.append(preds_bin.flatten())
            all_targets.append(targets_np.flatten())

            
            """if i % 5 == 0:
                # Tomamos el primer ROI del primer ejemplo del batch
                img_real = targets_np[0, 0, 0]
                img_prob = preds_prob[0, 0, 0].cpu().numpy()
                img_pred = preds_bin[0, 0, 0]

                # Calculamos IoU local para esta imagen específica
                intersection = np.logical_and(img_real, img_pred).sum()
                union = np.logical_or(img_real, img_pred).sum()
                iou_local = (intersection + 1e-6) / (union + 1e-6)

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img_real, cmap='gist_heat')
                plt.title(f"Real (Epoch {epoch+1})")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(img_prob, cmap='inferno')
                plt.title(f"Prob: {img_prob.max():.2f}")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(img_pred, cmap='gist_heat')
                plt.title(f"Pred (IoU Local: {iou_local:.3f})")
                plt.axis('off')

                plt.tight_layout()
                plt.show()"""

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    
    # Métricas Globales
    current_f1 = f1_score(y_true, y_pred, zero_division=0)
    current_iou = jaccard_score(y_true, y_pred, zero_division=0)

    print(f"Métricas Validación - F1: {current_f1:.4f} | IoU: {current_iou:.4f}")

    return total_loss / len(loader), current_f1, current_iou

def load_checkpoint(model, optimizer, scheduler):
    """Carga el último checkpoint disponible en la carpeta de checkpoints."""
    if not os.path.exists(CHECKPOINT_DIR):
        return 0, 0
    
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    
    if not files:
        return 0, 0
        
    try:
        files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        last_file = files[-1]
        filepath = os.path.join(CHECKPOINT_DIR, last_file)
        
        print(f"Intentando cargar último checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        
        print(f"✓ Checkpoint cargado exitosamente: continuando desde epoch {start_epoch}")
        return start_epoch, best_iou
    except Exception as e:
        print(f"⚠️ Error cargando checkpoint: {e}")
        return 0, 0
CHECKPOINT_DIR = 'saved/checkpoints'

def save_checkpoint(model, optimizer, scheduler, epoch, best_iou,filename=''):
    """Guarda el checkpoint de la época actual."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth' if not filename else filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou,
    }
    torch.save(checkpoint, filepath)
    print(f"  Checkpoint guardado: {filepath}")


class SoftZoneLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        Alpha y Beta controlan el balance entre Falsos Negativos y Falsos Positivos.
        Alpha > Beta penaliza más los incendios NO detectados.
        """
        super(SoftZoneLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logist, targets):
        # Aplicamos Sigmoid 
        preds = torch.sigmoid(logist)
        
        # (B*R, 1, H, W) -> (N,)
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        logist = logist.reshape(-1)
        # Tversky Loss 
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1 - tversky
        
        # Binary Cross Entropy con Logits (para estabilidad)
        preds = torch.clamp(preds, self.smooth, 1.0 - self.smooth)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logist,targets)

        
        # Combinación: El BCE da estabilidad y el Tversky maneja el desbalance
        return 0.5 * bce_loss + 0.5 * tversky_loss
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
    base_unet = UNet3D(in_channels=29, out_channels=1).to(device)
    model = FireModelROI(base_unet).to(device) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    criterion = SoftZoneLoss(alpha=0.8, beta=0.2)

    start_epoch, best_f1 = load_checkpoint(model, optimizer, scheduler)
    
    history_file = 'train_history.json'
    history = []
    if os.path.exists(history_file) and start_epoch > 0:
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except: pass

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        train_loss = train_epoch(
            model, dataloader_train, optimizer, scaler, 
            device, criterion, accum_steps=4
        )
        
        val_loss, current_f1 ,current_iou= validate_zonal(
            model, dataloader_val, device, criterion, epoch=epoch
        )
        
        scheduler.step(current_f1)
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'current_f1': current_f1,
            'current_iou':current_iou
        }
        history.append(epoch_metrics)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)

        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"[*] Nuevo mejor F1: {best_f1:.4f}. Guardando...")
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1)
        elif (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, filename="checkpoint_periodic.pth")

        print(f"Resumen Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best F1: {best_f1:.4f}")



if __name__ == "__main__":
    main()