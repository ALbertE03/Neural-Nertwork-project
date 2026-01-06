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
from model import FirePredictModel
from dataset import TSDataset, collate_fn
from constants import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Peso para la clase minoritaria (fuego)
        self.gamma = gamma  # Factor de enfoque
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: Logits (B, 1, H, W)
        # targets: (B, 1, H, W)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # probabilidad de la clase correcta
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        # alpha: penalización FP
        # beta: penalización FN (Queremos beta alto para priorizar Recall)
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (probs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * probs_flat).sum()
        FN = (targets_flat * (1 - probs_flat)).sum()
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky_index

def train_epoch(model, loader, optimizer, scaler, device, pos_weight=None, pred_horizon=1, accum_steps=4):
    """
    Entrena el modelo para clasificación BINARIA (Fire vs Non-Fire).
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    # Combined Loss: Focal (Hard examples) + Tversky (Recall oriented)
    # Focal parameters: alpha alto para dar importancia a clase 1
    focal_criterion = FocalLoss(alpha=0.75, gamma=2).to(device) 
    # Tversky parameters: beta > alpha para castigar más los falsos negativos
    tversky_criterion = TverskyLoss(alpha=0.3, beta=0.7).to(device)
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)                # (B, T, C_total, H, W)
        labels = batch['label'].float().to(device) # (B, T, H, W) 
        
        # Input: sequence until T-1
        input_seq = x[:, :-pred_horizon]  # (B, T-1, C, H, W)
        
        # Target: last day
        # Ensure target is (B, 1, H, W) for BCE
        target = labels[:, -1].unsqueeze(1) # (B, 1, H, W)
        
        # Binarize target for Loss rigorously (soft labels -> 0 or 1)
        target_bin = (target > FIRE_THRESHOLDS[0]).float()

        # Forward
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            # pred: (B, 1, H, W) logits
            pred_logits = model(input_seq)
            
            # Loss = Focal + Tversky
            loss_focal = focal_criterion(pred_logits, target_bin)
            loss_tversky = tversky_criterion(pred_logits, target_bin)
            
            loss = (loss_focal + loss_tversky) / accum_steps

        # Backward
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            # Gradient Clipping para estabilizar LSTM/RNN
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        current_loss = loss.item() * accum_steps
        
        # Accuracy estimativa (Threshold from constants on sigmoid)
        with torch.no_grad():
            preds = (torch.sigmoid(pred_logits) > FIRE_THRESHOLDS[0]).float()
            target_bin = (target > FIRE_THRESHOLDS[0]).float()
            correct = (preds == target_bin).sum().item()
            pixels = target.numel()
            
            total_loss += current_loss * x.size(0)
            total_correct += correct
            total_pixels += pixels
            
            acc = correct / max(pixels, 1)

        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{acc:.4f}'})
    
    if (batch_idx + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    avg_loss = total_loss / max(len(loader.dataset), 1)
    avg_acc = total_correct / max(total_pixels, 1)
    
    return avg_loss, avg_acc, 0.0


@torch.no_grad()
def validate(model, loader, device, pred_horizon=1, epoch=None, pos_weight=None):
    """
    Evalúa clasificación BINARIA usando sklearn metrics.
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # Validation uses consistent Combined Loss for reporting
    focal_criterion = FocalLoss(alpha=0.75, gamma=2).to(device)
    tversky_criterion = TverskyLoss(alpha=0.3, beta=0.7).to(device)
    
    # Store all preds and targets for sklearn robust metrics (on CPU)
    # Acumulamos en listas para concatenar al final
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Validation")
    
    for batch in pbar:
        x = batch['x'].to(device)
        labels = batch['label'].float().to(device)
        
        input_seq = x[:, :-pred_horizon]
        target = labels[:, -1].unsqueeze(1) # (B, 1, H, W)
        target_bin = (target > FIRE_THRESHOLDS[0]).float()
        
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            pred_logits = model(input_seq)
            
            # Loss combined
            loss_focal = focal_criterion(pred_logits, target_bin)
            loss_tversky = tversky_criterion(pred_logits, target_bin)
            loss = loss_focal + loss_tversky
            
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        
        # Binarize Predictions
        probs = torch.sigmoid(pred_logits)
        preds = (probs > FIRE_THRESHOLDS[0]).long().cpu().numpy().flatten()
        targets = (target > FIRE_THRESHOLDS[0]).long().cpu().numpy().flatten()
        
        all_preds.extend(preds)
        all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calcular Métricas con Sklearn
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    iou = jaccard_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    # Accuracy manual 
    accuracy = (all_preds == all_targets).mean()
    
    # Confusion Matrix unpacking
    # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n--- Validation Results (Binary: Fire vs No-Fire) ---")
    print(f"Global Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU (Intersection over Union): {iou:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # --- Plotting ---
    if epoch is not None:
        try:
            class_names = ['No Fire', 'Fire']
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Binary Confusion Matrix - Epoch {epoch+1}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.pause(0.1)
        except Exception as e:
            print(f"Error plotting matrices: {e}")
            
    avg_loss = total_loss / max(total_samples, 1)
    
    return avg_loss, accuracy, f1, iou


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


def calculate_pos_weight(loader):
    """Calcular peso para la clase positiva (fuego) para contrarrestar desbalanceo."""
    print("Calculando peso para clase positiva (muestreando dataset)...")
    total_pixels_fire = 0
    total_pixels_bg = 0
    limit = max(1, len(loader) // 5) # Sample 20%
    
    for i, batch in enumerate(loader):
        if i >= limit: break
        # Binarize labels using defined threshold to capture soft labels
        labels = (batch['label'] > FIRE_THRESHOLDS[0]).long() # (B, T, H, W)
        
        # Count 1s and 0s
        n_fire = (labels == 1).sum().item()
        n_bg = (labels == 0).sum().item()
        
        total_pixels_fire += n_fire
        total_pixels_bg += n_bg
    
    if total_pixels_fire == 0:
        print("Warning: No fire found in sample. Using default pos_weight=1.0")
        return torch.tensor([1.0])
        
    # pos_weight = Number of Negatives / Number of Positives
    weight = total_pixels_bg / total_pixels_fire
    
    print(f"Found: {total_pixels_bg} background pixels, {total_pixels_fire} fire pixels.")
    print(f"Calculated pos_weight: {weight:.2f}")
    
    return torch.tensor([weight])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    plt.ion()
    
    # 1. Dataset & Dataloader
    if not DATA_PATHS:
        print("Warning: DATA_PATHS está vacío. Usando ruta por defecto 'data/' si existe.")
        paths_to_use = ['data/'] # Placeholder
    else:
        paths_to_use = DATA_PATHS
        
    full_dataset = TSDataset(paths_to_use, SHAPES)
    
    # Simple split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Calcular Peso Positivo (Deprecated by Focal Loss, but keeping variable for signature compatibility if needed)
    pos_weight = None 
    
    model = FirePredictModel(
        input_channels=INPUT_CHANNELS,  
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
        num_classes=1 
    ).to(device)
    
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
                                               pos_weight=pos_weight, # Pass binary weight
                                               pred_horizon=PRED_SEQ_LEN, accum_steps=ACCUM_STEPS)
        val_loss, val_acc, val_f1, val_iou = validate(model, dataloader_val, device, PRED_SEQ_LEN, epoch=epoch, pos_weight=pos_weight)
        
        scheduler.step(val_f1) # Optimize for F1 instead of Acc in imbalanced binary
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | F1: {val_f1:.4f} | IoU: {val_iou:.4f}")
        
        # Guardar métricas en historial
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_iou': val_iou,
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

