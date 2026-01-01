import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm
from utils import iou_score,dice_loss
from constants import *
from dataset import *

def train_epoch(model, loader, optimizer, scaler, device, pred_horizon=2, accum_steps=4):
    """
    Entrena el modelo para predecir label+1 y label+2.
    
    Con gradient accumulation para simular batch_size efectivo = batch_size * accum_steps
    """
    model.train()
    total_loss = 0
    total_samples = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)                # (B, T, C_total, H, W)
        labels = batch['label'].to(device)       # (B, T, H, W)
        label_mask = batch['label_mask'].to(device)  # (B, T, H, W)
        
        B, T, C, H, W = x.shape
        
        # Necesitamos al menos pred_horizon+1 días
        if T <= pred_horizon:
            continue
        
        # Input: toda la secuencia menos los últimos pred_horizon días
        input_seq = x[:, :-pred_horizon]  # (B, T-2, C, H, W)
        
        # Targets: los últimos pred_horizon días
        targets = []
        for k in range(1, pred_horizon + 1):
            target_idx = T - pred_horizon + k - 1
            targets.append(labels[:, target_idx])  # (B, H, W)
        
        target = torch.stack(targets, dim=1)  # (B, 2, H, W)
        target_mask = torch.stack([
            label_mask[:, T - pred_horizon + k - 1] 
            for k in range(1, pred_horizon + 1)
        ], dim=1)  # (B, 2, H, W)
        
        # Forward con autocast para AMP
        with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda'):
            pred = model(input_seq)  # (B, 2, H, W)
            
            # Loss: BCE + Dice, solo donde mask=1
            bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
            bce = (bce * target_mask).sum() / (target_mask.sum() + 1e-8)
            
            dice = dice_loss(pred, target, target_mask)
            loss = (bce + dice) / accum_steps  # Escalar loss por accumulation steps

        # Backward con scaler
        scaler.scale(loss).backward()
        
        # Actualizar pesos cada accum_steps batches
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps * B
        total_samples += B
    
    # Actualizar con gradientes restantes
    if (batch_idx + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, device, pred_horizon=2):
    """Evalúa el modelo en datos de validación."""
    model.eval()
    total_iou = 0
    total_samples = 0
    
    for batch in loader:
        x = batch['x'].to(device)
        labels = batch['label'].to(device)
        label_mask = batch['label_mask'].to(device)
        
        B, T, C, H, W = x.shape
        if T <= pred_horizon:
            continue
        
        input_seq = x[:, :-pred_horizon]
        
        targets = torch.stack([
            labels[:, T - pred_horizon + k - 1] 
            for k in range(1, pred_horizon + 1)
        ], dim=1)
        
        target_mask = torch.stack([
            label_mask[:, T - pred_horizon + k - 1] 
            for k in range(1, pred_horizon + 1)
        ], dim=1)
        
        with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda'):
            pred = model(input_seq)
        
        # IoU para cada día predicho
        for k in range(pred_horizon):
            iou = iou_score(pred[:, k], targets[:, k], target_mask[:, k])
            total_iou += iou * B
        
        total_samples += B * pred_horizon
    
    return total_iou / max(total_samples, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, filepath='checkpoint.pth'):
    """Guarda el checkpoint del entrenamiento."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou,
    }
    torch.save(checkpoint, filepath)
    print(f"  Checkpoint guardado: epoch {epoch+1}")


def load_checkpoint(model, optimizer, scheduler, filepath='checkpoint.pth'):
    """Carga el último checkpoint si existe."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        print(f"✓ Checkpoint cargado: continuando desde epoch {start_epoch}")
        return start_epoch, best_iou
    return 0, 0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Modelo
    model = FirePredictModel(
        input_channels=INPUT_CHANNELS,  
        hidden_channels=HIDDEN_CHANNELS,
        pred_horizon=PRED_SEQ_LEN,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler, 'checkpoint.pth')
    
    # Scaler para AMP
    scaler = torch.amp.GradScaler(enabled=(device.type != 'cpu'))
    
    
    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_epoch(model, dataloader_train, optimizer, scaler, device, PRED_SEQ_LEN, ACCUM_STEPS)
        val_iou = validate(model, dataloader_val, device, PRED_SEQ_LEN)
        
        scheduler.step(val_iou)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ Mejor modelo guardado (IoU: {best_iou:.4f})")
        
        save_checkpoint(model, optimizer, scheduler, epoch, best_iou, 'checkpoint.pth')
    
    print(f"\nEntrenamiento completado. Mejor IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()

