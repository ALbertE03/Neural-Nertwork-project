import torch


def dice_loss(pred, target, mask=None, smooth=1.0):
    """
    Dice loss con soporte para máscara de validez.
    
    Args:
        pred: (B, ...) - probabilidades [0, 1]
        target: (B, ...) - binary mask
        mask: (B, ...) - 1 donde hay datos válidos, 0 donde ignorar
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    pred_flat = (pred * mask).view(-1)
    target_flat = (target * mask).view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice


def iou_score(pred, target, mask=None, threshold=0.5):
    """
    IoU score con soporte para máscara.
    
    Args:
        pred: probabilidades [0, 1]
        target: binary mask
        mask: máscara de validez
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    pred_bin = ((pred > threshold).float() * mask).view(-1)
    target_flat = (target * mask).view(-1)
    
    intersection = (pred_bin * target_flat).sum()
    union = pred_bin.sum() + target_flat.sum() - intersection
    
    if union == 0:
        return 1.0
    
    return (intersection / union).item()
