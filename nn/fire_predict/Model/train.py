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
from constants import  EPOCHS, LEARNING_RATE
from model import UNet3D
CHECKPOINT_DIR='saved/checkpoints'
def train_epoch(model, loader, optimizer, scaler, device, criterion, accum_steps=4):
    model.train()
    total_loss = 0

    tp_total = torch.tensor(0.0).to(device)
    fp_total = torch.tensor(0.0).to(device)
    fn_total = torch.tensor(0.0).to(device)
    
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(tqdm(loader, desc="Training")):
  
        input_history = inputs[:, :-1, :, :, :].to(device, non_blocking=True)
        target_future = targets[:, -1, :, :].to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
           
            preds = model(input_history) 
            
           
            loss = criterion(preds, target_future)
            loss_scaled = loss / accum_steps

    
        scaler.scale(loss_scaled).backward()

        with torch.no_grad():
            preds_bin = (torch.sigmoid(preds) > 0.5).float()
            targets_bin = (target_future > 0.5).float()
            
            tp_total += (preds_bin * targets_bin).sum()
            fp_total += (preds_bin * (1 - targets_bin)).sum()
            fn_total += ((1 - preds_bin) * targets_bin).sum()


        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() 
    
    smooth = 1e-7
    precision = tp_total / (tp_total + fp_total + smooth)
    recall = tp_total / (tp_total + fn_total + smooth)
    train_f1 = 2 * (precision * recall) / (precision + recall + smooth)
    train_iou = tp_total / (tp_total + fp_total + fn_total + smooth)
    
    avg_loss = total_loss / len(loader)
    
    print(f"Train Loss: {avg_loss:.4f} | F1: {train_f1.item():.4f} | IoU: {train_iou.item():.4f}")
    
    return avg_loss, train_f1.item(), train_iou.item()


def validate_zonal(model, loader, device, criterion, epoch=0):
    model.eval()
    total_loss = 0
    
    tp_total = torch.tensor(0.0).to(device)
    fp_total = torch.tensor(0.0).to(device)
    fn_total = torch.tensor(0.0).to(device)
    tn_total = torch.tensor(0.0).to(device)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(loader, desc="Validating")):

            input_history = inputs[:, :-1, :, :, :].to(device, non_blocking=True)
            target_future = targets[:, -1, :, :].to(device, non_blocking=True).unsqueeze(1)
            
            preds = model(input_history)
            
            loss = criterion(preds, target_future)
            total_loss += loss.item()


            preds_bin = (torch.sigmoid(preds) > 0.5).float()
            targets_bin = (target_future > 0.5).float()

            tp_total += (preds_bin * targets_bin).sum()
            fp_total += (preds_bin * (1 - targets_bin)).sum()
            fn_total += ((1 - preds_bin) * targets_bin).sum()
            tn_total += ((1 - preds_bin) * (1 - targets_bin)).sum()


    smooth = 1e-7
    
    precision = tp_total / (tp_total + fp_total + smooth)
    recall = tp_total / (tp_total + fn_total + smooth)
    
    current_f1 = 2 * (precision * recall) / (precision + recall + smooth)
    current_iou = tp_total / (tp_total + fp_total + fn_total + smooth)

    print(f"\n[Epoch {epoch}] Val Loss: {total_loss/len(loader):.4f}")
    print(f"F1: {current_f1.item():.4f} | IoU: {current_iou.item():.4f} | Recall: {recall.item():.4f}")

    return total_loss / len(loader), current_f1.item(), current_iou.item()


def load_checkpoint(model, optimizer, scheduler):
    """Carga el último checkpoint disponible."""
    if not os.path.exists(CHECKPOINT_DIR):
        return 0, 0.0
    
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    if not files:
        return 0, 0.0
        
    try:
        # Intentar cargar el más reciente por fecha de modificación si el nombre varía
        files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
        last_file = files[-1]
        filepath = os.path.join(CHECKPOINT_DIR, last_file)
        
        print(f"Cargando checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        # Usamos F1 como métrica principal de guardado
        best_f1 = checkpoint.get('best_f1', 0.0) 
        
        return start_epoch, best_f1
    except Exception as e:
        print(f"⚠️ Error al cargar: {e}")
        return 0, 0.0

def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, filename=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if filename is None:
        filename = f'checkpoint3_epoch_{epoch+1}.pth'
    
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_f1': best_f1,
    }
    torch.save(checkpoint, filepath)


import torch.nn as nn
import torch.nn.functional as F

class SoftZoneLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(SoftZoneLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. BCE con Logits (Estabilidad numérica)
        # Recibe logits crudos
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # 2. Tversky Loss (Manejo de desbalance)
        # Necesita probabilidades (0 a 1)
        preds = torch.sigmoid(logits)
        
        preds = preds.view(-1)
        targets = targets.view(-1)

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        tversky_loss = 1 - tversky

        # Combinación pesada
        return 0.5 * bce_loss + 0.5 * tversky_loss
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet3D(in_channels=28, out_channels=1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    criterion = SoftZoneLoss(alpha=0.8, beta=0.2)

    start_epoch, best_f1 = load_checkpoint(model, optimizer, scheduler)
    
    history_file = 'train_history3.json'
    history = []
    if os.path.exists(history_file) and start_epoch > 0:
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except: pass

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Entrenamiento
        train_loss = train_epoch(
            model, dataloader_train, optimizer, scaler, 
            device, criterion, accum_steps=4
        )
        
        # Validación
        val_loss, current_f1, current_iou = validate_zonal(
            model, dataloader_val, device, criterion, epoch=epoch+1
        )
        

        scheduler.step(current_f1)
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'f1': float(current_f1),
            'iou': float(current_iou),
            'lr': float(optimizer.param_groups[0]['lr'])
        }
        history.append(epoch_metrics)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)

        # Guardar mejor modelo
        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"[*] Nuevo récord F1: {best_f1:.4f}. Guardando...")
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, filename="best_model.pth")
        
        # Guardado periódico de seguridad
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, filename=f"checkpoint_ep{epoch+1}.pth")

        print(f"Summary: Loss {val_loss:.4f} | F1 {current_f1:.4f} | IoU {current_iou:.4f} | Best F1: {best_f1:.4f}")



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet3D(in_channels=28, out_channels=1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    criterion = SoftZoneLoss(alpha=0.8, beta=0.2)

    start_epoch, best_f1 = load_checkpoint(model, optimizer, scheduler)
    
    history_file = 'train_history1.json'
    history = []
    if os.path.exists(history_file) and start_epoch > 0:
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except: pass

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Entrenamiento
        train_loss = train_epoch(
            model, dataloader_train, optimizer, scaler, 
            device, criterion, accum_steps=4
        )
        
        # Validación
        val_loss, current_f1, current_iou = validate_zonal(
            model, dataloader_val, device, criterion, epoch=epoch+1
        )
        

        scheduler.step(current_f1)
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'f1': float(current_f1),
            'iou': float(current_iou),
            'lr': float(optimizer.param_groups[0]['lr'])
        }
        history.append(epoch_metrics)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)

        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"[*] Nuevo récord F1: {best_f1:.4f}. Guardando...")
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, filename=f"best_model{epoch+1}.pth")
        
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_f1, filename=f"checkpoint_ep{epoch+1}.pth")

        print(f"Summary: Loss {val_loss:.4f} | F1 {current_f1:.4f} | IoU {current_iou:.4f} | Best F1: {best_f1:.4f}")