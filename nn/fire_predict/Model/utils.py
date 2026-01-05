from scipy.ndimage import label as scipy_label
import numpy as np
import torch


def dice_loss(pred, target, mask=None, smooth=1.0, use_sigmoid=True):
    """
    pred: logits (si use_sigmoid=True) o probabilidades (si False)
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    if use_sigmoid:
        pred = torch.sigmoid(pred)
    
    pred_flat = (pred * mask).view(-1)
    target_flat = (target * mask).view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice

def iou_score(pred, target, mask=None, threshold=None):
    """
    IoU score. 
    - Si threshold != None: Binariza (Clásico).
    - Si threshold == None: Usa Lógica Difusa (Min/Max) para no perder info.
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    if threshold is not None:
        # Binarizado
        p = (pred > threshold).float() * mask
        t = (target > threshold).float() * mask
        intersection = (p * t).sum()
        union = (p + t - p * t).sum()
    else:
        # Soft Fuzzy (Min/Max) - Preserva intensidad
        p = pred * mask
        t = target * mask
        intersection = torch.min(p, t).sum()
        union = torch.max(p, t).sum()
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()

def precision_score(pred, target, mask=None, threshold=None):
    """
    Precision = Intersection / Prediction
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    if threshold is not None:
        p = (pred > threshold).float() * mask
        t = (target > threshold).float() * mask
        intersection = (p * t).sum()
    else:
        p = pred * mask
        t = target * mask
        intersection = torch.min(p, t).sum()
    
    predicted_positives = p.sum()
    
    if predicted_positives == 0:
        return 1.0
        
    return (intersection / predicted_positives).item()

def recall_score(pred, target, mask=None, threshold=None):
    """
    Recall = Intersection / Target
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    if threshold is not None:
        p = (pred > threshold).float() * mask
        t = (target > threshold).float() * mask
        intersection = (p * t).sum()
    else:
        p = pred * mask
        t = target * mask
        intersection = torch.min(p, t).sum()
    
    actual_positives = t.sum()
    
    if actual_positives == 0:
        return 1.0
        
    return (intersection / actual_positives).item()

def f1_score(pred, target, mask=None, threshold=None):
    """
    F1 score. Si threshold no es None, binariza las entradas.
    """
    prec = precision_score(pred, target, mask, threshold)
    rec = recall_score(pred, target, mask, threshold)
    
    if prec + rec == 0:
        return 0.0
        
    return 2 * (prec * rec) / (prec + rec)

def burned_area_error(pred, target, pixel_area, mask=None, threshold=None):
    """
    Calcula el error absoluto medio en área quemada usando soft labels.
    
    Args:
        pred: probabilidades [B, T, H, W]
        target: soft labels [B, T, H, W]
        pixel_area: área de cada píxel [B, T]
        mask: máscara de validez [B, T, H, W]
        threshold: Ignorado
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    # Aplicar máscara de validez
    pred_masked = pred * mask
    target_masked = target * mask
    
    # Sumar píxeles de fuego por imagen: (B, T)
    pred_pixels = pred_masked.sum(dim=(-1, -2))
    target_pixels = target_masked.sum(dim=(-1, -2))
    
    # Calcular área total
    pred_area = pred_pixels * pixel_area
    target_area = target_pixels * pixel_area
    
    # Error absoluto por imagen
    abs_error = torch.abs(pred_area - target_area)
    
    # Promedio sobre el batch y tiempo
    return abs_error.mean().item()


