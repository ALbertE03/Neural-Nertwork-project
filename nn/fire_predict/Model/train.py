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
from constants import EPOCHS, LEARNING_RATE, PRED_SEQ_LEN, ACCUM_STEPS
from model import UNet3D
import numpy as np
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        
        # Focal Loss (píxel a píxel)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        # Dice Loss (Global/Estructural)
        smooth = 1e-5
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        return (focal_loss * (1 - self.dice_weight)) + (dice_loss * self.dice_weight)



def train_epoch(model, loader, optimizer, scaler, device,criterion, pred_horizon=1, accum_steps=4):
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)                # (B, T, C, H, W)
        labels = batch['label'].float().to(device) # (B, T, H, W) 
        
        # La U-Net 3D espera la secuencia completa o N-1 para predecir el último
        input_seq = x[:, :-pred_horizon]  # (B, T-1, C, H, W)
        
        # Target: El último frame de la secuencia de etiquetas
        target = labels[:, -1].unsqueeze(1) # (B, 1, H, W)
        
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            # Forward: UNet3D internamente hará el permute a (B, C, T, H, W)
            pred_logits = model(input_seq) # Salida esperada: (B, 1, H, W)
            
            # Cálculo de pérdida unificado
            loss = criterion(pred_logits, target) / accum_steps

        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Métricas de visualización
        with torch.no_grad():
            probs = torch.sigmoid(pred_logits)
            preds = (probs > 0.5).float()
            correct = (preds == target).sum().item()
            total_correct += correct
            total_pixels += target.numel()
            total_loss += loss.item() * accum_steps
            
        pbar.set_postfix({'loss': f'{loss.item()*accum_steps:.4f}', 'acc': f'{correct/target.numel():.4f}'})

    return total_loss / len(loader), total_correct / total_pixels, 0.0



@torch.no_grad()
def validate(model, loader, device,criterion, pred_horizon=1):
    model.eval()
    total_loss = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    
    pbar = tqdm(loader, desc="Validation")
    
    for batch in pbar:
        x = batch['x'].to(device)
        labels = batch['label'].float().to(device)
        
        input_seq = x[:, :-pred_horizon]
        target = labels[:, -1].unsqueeze(1)
        
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            pred_logits = model(input_seq)
            loss = criterion(pred_logits, target)
            
        total_loss += loss.item()
        
        probs = torch.sigmoid(pred_logits)
        preds = (probs > 0.3).float() # Puedes ajustar este umbral a 0.3 si el fuego es muy escaso
        
        tp += (preds * target).sum().item()
        fp += (preds * (1 - target)).sum().item()
        fn += ((1 - preds) * target).sum().item()
        tn += ((1 - preds) * (1 - target)).sum().item()

    # Cálculo de métricas
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou_fire = tp / (tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

    print(f"\nVal F1: {f1:.4f} | IoU Fire: {iou_fire:.4f} | Acc: {accuracy:.4f}")
    return total_loss / len(loader), accuracy, f1, iou_fire

CHECKPOINT_DIR = 'saved/checkpoints'

def save_checkpoint(model, optimizer, scheduler, epoch, best_iou):
    """Guarda el checkpoint de la época actual."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou,
    }
    torch.save(checkpoint, filepath)
    print(f"  Checkpoint guardado: {filepath}")


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    plt.ion()
    
    # Calcular Peso Positivo (Deprecated by Focal Loss, but keeping variable for signature compatibility if needed)
    pos_weight = None 
    criterion = DiceFocalLoss(alpha=0.95, dice_weight=0.7).to(device)
    
    # Configuración del modelo
    model = UNet3D(in_channels=28, out_channels=1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    # Cargar último checkpoint automáticamente
    start_epoch, best_metric = load_checkpoint(model, optimizer, scheduler)
    
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Cargar o inicializar historial
    history_file = 'train_history.json'
    if os.path.exists(history_file) and start_epoch > 0:
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
    else:
        history = []
        
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc, _ = train_epoch(model, dataloader_train, optimizer, scaler, device, 
                                               criterion,
                                               pred_horizon=PRED_SEQ_LEN, accum_steps=ACCUM_STEPS)
        val_loss, val_acc, val_f1, val_iou = validate(model, dataloader_val, device,criterion, pred_horizon=PRED_SEQ_LEN)
        
        scheduler.step(val_f1) # Optimize for F1 instead of Acc in imbalanced binary
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | F1: {val_f1:.4f} | Mean IoU: {val_iou:.4f}")
        
        # Guardar métricas en historial
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_mean_iou': val_iou,
        }
        history.append(epoch_metrics)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        
        if val_f1 > best_metric: # Save best by F1
            best_metric = val_f1
            os.makedirs('saved', exist_ok=True)
            torch.save(model.state_dict(), 'saved/best_model.pth')
            print(f"  Mejor modelo guardado (F1: {best_metric:.4f}) en saved/best_model.pth")
        
        # Guardar checkpoint de esta época
        save_checkpoint(model, optimizer, scheduler, epoch, best_metric)
    
    print(f"\nEntrenamiento completado. Mejor F1: {best_metric:.4f}")


if __name__ == '__main__':
    main()

