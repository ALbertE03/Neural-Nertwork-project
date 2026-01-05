import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
import json
from tqdm import tqdm
from constants import *
from model import FirePredictModel
from utils import dice_loss, iou_score, burned_area_error, f1_score, precision_score, recall_score

def train_epoch(model, loader, optimizer, scaler, device, pred_horizon=1, accum_steps=4):
    """
    Entrena el modelo para predecir label+1
    
    Con gradient accumulation para simular batch_size efectivo = batch_size * accum_steps
    """
    model.train()
    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_samples = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        x = batch['x'].to(device)                # (B, T, C_total, H, W)
        labels = batch['label'].to(device)       # (B, T, H, W)
        label_mask = batch['label_mask'].to(device)  # (B, T, H, W)
        
        B, T, C, H, W = x.shape
        
        
        # Input: toda la secuencia menos el último día
        input_seq = x[:, :-pred_horizon]  # (B, T-1, C, H, W)
        
        # Target: el último día
        target = labels[:, -1]  # (B, H, W)
        target_mask = label_mask[:, -1]  # (B, H, W)
        
        # Forward con autocast para AMP (Solo CUDA)
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            pred = model(input_seq)  # (B, H, W)
            
            # Loss: BCE + Dice, solo donde mask=1
            bce = nn.functional.binary_cross_entropy_with_logits(
                pred, target, reduction='none'
            )
            bce = (bce * target_mask).sum() / (target_mask.sum() + 1e-8)
            
            dice = dice_loss(pred.unsqueeze(1), target.unsqueeze(1), target_mask.unsqueeze(1), use_sigmoid=True)
            loss = (bce + dice) / accum_steps

        # Backward con scaler
        scaler.scale(loss).backward()
        
        # Actualizar pesos cada accum_steps batches
        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        current_loss = loss.item() * accum_steps
        current_bce = bce.item()
        current_dice = dice.item()
        
        total_loss += current_loss * B
        total_bce += current_bce * B
        total_dice += current_dice * B
        total_samples += B
        
        # Actualizar barra de progreso con el loss actual
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'bce': f'{current_bce:.4f}',
            'dice': f'{current_dice:.4f}'
        })
    
    # Actualizar con gradientes restantes
    if (batch_idx + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return (
        total_loss / max(total_samples, 1),
        total_bce / max(total_samples, 1),
        total_dice / max(total_samples, 1)
    )


@torch.no_grad()
def validate(model, loader, device, pred_horizon=1, threshold=None):
    """
    Evalúa el modelo. 
    - threshold=None: Usa métricas Soft Fuzzy (Min/Max) para no perder info.
    - threshold=0.5: Usa métricas Binarizadas (Clásicas).
    """
    model.eval()
    total_loss = 0
    total_iou = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_area_error = 0
    total_samples = 0
    pbar = tqdm(loader, desc="Validating")
    for batch in pbar:
        x = batch['x'].to(device)
        labels = batch['label'].to(device)
        label_mask = batch['label_mask'].to(device)
        pixel_area = batch['pixel_area'].to(device)
        
        B, T, C, H, W = x.shape
        if T <= pred_horizon:
            continue
        
        input_seq = x[:, :-pred_horizon]
        
        target = labels[:, -1]
        target_mask = label_mask[:, -1]
        target_area = pixel_area[:, -1]
        
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            pred = model(input_seq)
            
            # Calculate Validation Loss (BCE + Dice)
            bce = nn.functional.binary_cross_entropy_with_logits(
                pred, target, reduction='none'
            )
            bce = (bce * target_mask).sum() / (target_mask.sum() + 1e-8)
            dice = dice_loss(pred.unsqueeze(1), target.unsqueeze(1), target_mask.unsqueeze(1), use_sigmoid=True)
            loss = bce + dice
            
        total_loss += loss.item() * B
            
        pred_probs = torch.sigmoid(pred) 
        
        # IoU
        iou = iou_score(pred_probs, target, target_mask, threshold=threshold)
        total_iou += iou * B
        
        # F1, Precision, Recall
        f1 = f1_score(pred_probs, target, target_mask, threshold=threshold)
        prec = precision_score(pred_probs, target, target_mask, threshold=threshold)
        rec = recall_score(pred_probs, target, target_mask, threshold=threshold)
        
        total_f1 += f1 * B
        total_precision += prec * B
        total_recall += rec * B
        
        # Area Error (Soft)
        area_err = burned_area_error(
            pred_probs.unsqueeze(1), 
            target.unsqueeze(1), 
            target_area.unsqueeze(1), 
            target_mask.unsqueeze(1)
        )
        total_area_error += area_err * B
        
        total_samples += B
    
    return (
        total_loss / max(total_samples, 1),
        total_iou / max(total_samples, 1), 
        total_f1 / max(total_samples, 1),
        total_precision / max(total_samples, 1),
        total_recall / max(total_samples, 1),
        total_area_error / max(total_samples, 1)
    )


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
    
    # Listar archivos que cumplan el patrón checkpoint_epoch_X.pth
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    
    if not files:
        return 0, 0
        
    # Extraer número de época y ordenar
    # Formato esperado: checkpoint_epoch_123.pth
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
    
    model = FirePredictModel(
        input_channels=INPUT_CHANNELS,  
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    # Cargar último checkpoint automáticamente
    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler)
    
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
        train_loss, train_bce, train_dice = train_epoch(model, dataloader_train, optimizer, scaler, device, PRED_SEQ_LEN, ACCUM_STEPS)
        val_loss, val_iou, val_f1, val_prec, val_rec, val_area_err = validate(model, dataloader_val, device, PRED_SEQ_LEN)
        
        scheduler.step(val_iou)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice:.4f}) | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | Area Err: {val_area_err:.2e}")
        
        # Guardar métricas en historial
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_bce': train_bce,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_f1': val_f1,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_area_error': val_area_err,
    
        }
        history.append(epoch_metrics)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        
        if val_iou > best_iou:
            best_iou = val_iou
            os.makedirs('saved', exist_ok=True)
            torch.save(model.state_dict(), 'saved/best_model.pth')
            print(f"  Mejor modelo guardado (IoU: {best_iou:.4f}) en saved/best_model.pth")
        
        # Guardar checkpoint de esta época
        save_checkpoint(model, optimizer, scheduler, epoch, best_iou)
    
    print(f"\nEntrenamiento completado. Mejor IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()

