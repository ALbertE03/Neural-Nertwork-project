import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
import json
from tqdm import tqdm
from constants import *
from model import FirePredictModel
from utils import dice_loss, iou_score, burned_area_error, centroid_distance_error, f1_score, precision_score, recall_score, physical_area_consistency, cluster_count_error

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
        
        
        # Input: toda la secuencia menos el último día
        input_seq = x[:, :-pred_horizon]  # (B, T-1, C, H, W)
        
        # Target: el último día
        target = labels[:, -1]  # (B, H, W)
        target_mask = label_mask[:, -1]  # (B, H, W)
        
        # Forward con autocast para AMP
        with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda'):
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
        total_loss += current_loss * B
        total_samples += B
        
        # Actualizar barra de progreso con el loss actual
        pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
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
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_area_error = 0
    total_centroid_error = 0
    total_centroid_samples = 0
    total_phys_diff = 0
    total_phys_rel_err = 0
    total_cluster_error = 0
    total_samples = 0
    pbar = tqdm(loader, desc="Validating")
    for batch in pbar:
        x = batch['x'].to(device)
        labels = batch['label'].to(device)
        label_mask = batch['label_mask'].to(device)
        pixel_area = batch['pixel_area'].to(device)
        original_area = batch['original_area'].to(device)
        original_count = batch['original_count'].to(device)
        
        B, T, C, H, W = x.shape
        if T <= pred_horizon:
            continue
        
        input_seq = x[:, :-pred_horizon]
        
        target = labels[:, -1]
        target_mask = label_mask[:, -1]
        target_area = pixel_area[:, -1]
        target_orig_area = original_area[:, -1]
        target_orig_count = original_count[:, -1]
        
        with torch.amp.autocast(device_type=device.type if device.type != 'cpu' else 'cuda'):
            pred = model(input_seq)
            
        pred_probs = torch.sigmoid(pred) 
        
        # IoU
        iou = iou_score(pred_probs, target, target_mask)
        total_iou += iou * B
        
        # F1, Precision, Recall
        f1 = f1_score(pred_probs, target, target_mask)
        prec = precision_score(pred_probs, target, target_mask)
        rec = recall_score(pred_probs, target, target_mask)
        
        total_f1 += f1 * B
        total_precision += prec * B
        total_recall += rec * B
        
        # Area Error
        area_err = burned_area_error(
            pred_probs.unsqueeze(1), 
            target.unsqueeze(1), 
            target_area.unsqueeze(1), 
            target_mask.unsqueeze(1)
        )
        total_area_error += area_err * B

        # Centroid Error
        centroid_err_sum, centroid_count = centroid_distance_error(
            pred_probs.unsqueeze(1),
            target.unsqueeze(1),
            target_area.unsqueeze(1),
            target_mask.unsqueeze(1)
        )
        total_centroid_error += centroid_err_sum
        total_centroid_samples += centroid_count
        
        # Physical Area Consistency
        phys_metrics = physical_area_consistency(
            mask_proj=pred_probs, 
            pixel_area_proj=target_area, 
            area_orig=target_orig_area,
            threshold=0.5
        )
        total_phys_diff += phys_metrics['diff_mean'] * B
        total_phys_rel_err += phys_metrics['rel_error'] * B
        
        # Cluster Count Error
        cluster_err = cluster_count_error(
            pred=pred_probs,
            target_count=target_orig_count,
            mask=target_mask,
            threshold=0.5
        )
        total_cluster_error += cluster_err * B
        
        total_samples += B
    
    return (
        total_iou / max(total_samples, 1), 
        total_f1 / max(total_samples, 1),
        total_precision / max(total_samples, 1),
        total_recall / max(total_samples, 1),
        total_area_error / max(total_samples, 1),
        total_centroid_error / max(total_centroid_samples, 1),
        total_phys_diff / max(total_samples, 1),
        total_phys_rel_err / max(total_samples, 1),
        total_cluster_error / max(total_samples, 1)
    )


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
        checkpoint = torch.load(filepath,map_location='cpu')
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
    
    model = FirePredictModel(
        input_channels=INPUT_CHANNELS,  
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler, 'checkpoint.pth')
    
    scaler = torch.amp.GradScaler(enabled=(device.type != 'cpu'))
    
    # Cargar o inicializar historial
    history_file = 'train_history.json'
    if os.path.exists(history_file) and start_epoch > 0:
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, dataloader_train, optimizer, scaler, device, PRED_SEQ_LEN, ACCUM_STEPS)
        val_iou, val_f1, val_prec, val_rec, val_area_err, val_centroid_err, val_phys_diff, val_phys_rel, val_cluster_err = validate(model, dataloader_val, device, PRED_SEQ_LEN)
        
        scheduler.step(val_iou)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | Area Err: {val_area_err:.2e} | Centroid Err: {val_centroid_err:.2f}m | Phys Diff: {val_phys_diff:.2e} | Phys Rel Err: {val_phys_rel:.2e} | Cluster Err: {val_cluster_err:.2f}")
        
        # Guardar métricas en historial
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_iou': val_iou,
            'val_f1': val_f1,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_area_error': val_area_err,
            'val_centroid_error': val_centroid_err,
            'val_phys_diff': val_phys_diff,
            'val_phys_rel_err': val_phys_rel,
            'val_cluster_error': val_cluster_err
        }
        history.append(epoch_metrics)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  Mejor modelo guardado (IoU: {best_iou:.4f})")
        
        save_checkpoint(model, optimizer, scheduler, epoch, best_iou, 'checkpoint.pth')
    
    print(f"\nEntrenamiento completado. Mejor IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()

