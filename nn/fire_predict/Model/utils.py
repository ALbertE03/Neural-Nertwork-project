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

def precision_score(pred, target, mask=None, threshold=0.5):
    """
    Precision score con soporte para máscara.
    Precision = TP / (TP + FP)
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    pred_bin = ((pred > threshold).float() * mask).view(-1)
    target_flat = (target * mask).view(-1)
    
    tp = (pred_bin * target_flat).sum()
    fp = (pred_bin * (1 - target_flat)).sum()
    
    if tp + fp == 0:
        return 1.0
        
    return (tp / (tp + fp)).item()

def recall_score(pred, target, mask=None, threshold=0.5):
    """
    Recall score con soporte para máscara.
    Recall = TP / (TP + FN)
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    pred_bin = ((pred > threshold).float() * mask).view(-1)
    target_flat = (target * mask).view(-1)
    
    tp = (pred_bin * target_flat).sum()
    fn = ((1 - pred_bin) * target_flat).sum()
    
    if tp + fn == 0:
        return 1.0
        
    return (tp / (tp + fn)).item()

def f1_score(pred, target, mask=None, threshold=0.5):
    """
    F1 score con soporte para máscara.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    prec = precision_score(pred, target, mask, threshold)
    rec = recall_score(pred, target, mask, threshold)
    
    if prec + rec == 0:
        return 0.0
        
    return 2 * (prec * rec) / (prec + rec)

def burned_area_error(pred, target, pixel_area, mask=None, threshold=0.5):
    """
    Calcula el error absoluto medio en área quemada (en las unidades de pixel_area, ej. m^2).
    
    Args:
        pred: probabilidades [B, T, H, W]
        target: binary mask [B, T, H, W]
        pixel_area: área de cada píxel [B, T]
        mask: máscara de validez [B, T, H, W]
        threshold: umbral para binarizar predicción
    """
    if mask is None:
        mask = torch.ones_like(pred)
    
    # Binarizar predicción
    pred_bin = (pred > threshold).float()
    
    # Aplicar máscara de validez
    pred_masked = pred_bin * mask
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

def centroid_distance_error(pred, target, pixel_area, mask=None, threshold=0.5):
    """
    Calcula la distancia euclidiana entre los centros de masa (en metros).
    """
    if mask is None:
        mask = torch.ones_like(pred)
        
    # Binarizar
    pred_bin = (pred > threshold).float() * mask
    target_bin = (target > 0.5).float() * mask
    
    B, T, H, W = pred.shape
    
    # Coordenadas (H, W)
    y_coords = torch.arange(H, device=pred.device).float().view(1, 1, H, 1).expand(B, T, H, W)
    x_coords = torch.arange(W, device=pred.device).float().view(1, 1, 1, W).expand(B, T, H, W)
    
    # Suma de masa (píxeles de fuego)
    pred_mass = pred_bin.sum(dim=(-2, -1))    # (B, T)
    target_mass = target_bin.sum(dim=(-2, -1)) # (B, T)
    
    # Evitar división por cero (solo calculamos donde ambos tengan fuego)
    valid_mask = (pred_mass > 0) & (target_mass > 0)
    
    if not valid_mask.any():
        return 0.0, 0
        
    # Calcular centroides (B, T)
    pred_cy = (pred_bin * y_coords).sum(dim=(-2, -1)) / (pred_mass + 1e-8)
    pred_cx = (pred_bin * x_coords).sum(dim=(-2, -1)) / (pred_mass + 1e-8)
    
    target_cy = (target_bin * y_coords).sum(dim=(-2, -1)) / (target_mass + 1e-8)
    target_cx = (target_bin * x_coords).sum(dim=(-2, -1)) / (target_mass + 1e-8)
    
    # Distancia en píxeles
    dist_px = torch.sqrt((pred_cx - target_cx)**2 + (pred_cy - target_cy)**2)
    
    # Convertir a metros: pixel_side = sqrt(pixel_area)
    # Asumimos píxeles cuadrados para la conversión de distancia
    pixel_side = torch.sqrt(pixel_area) # (B, T)
    dist_meters = dist_px * pixel_side
    
    # Retornar suma de errores y cantidad de muestras válidas para promediar correctamente fuera
    if valid_mask.any():
        return dist_meters[valid_mask].sum().item(), valid_mask.sum().item()
    else:
        return 0.0, 0

def physical_area_consistency(mask_proj, pixel_area_proj, area_orig, threshold=0.5):
    """
    Evalúa la consistencia del área física.
    Usa los valores suaves (probabilidades) de mask_proj para calcular el área esperada.
    """
    # NO binarizamos mask_proj. Usamos la probabilidad como fracción de área.
    # Si el modelo predice 0.2, asumimos que el 20% del píxel está quemado.
    # Esto coincide con la lógica de 'Resampling.average' en el dataset.
    
    # Sumar "fracciones de fuego" por muestra
    pixels_proj = mask_proj.view(mask_proj.shape[0], -1).sum(dim=1)
    
    # Calcular área física total (m^2)
    area_proj = pixels_proj * pixel_area_proj
    
    # Diferencia (Proj - Orig). 
    diff_area = area_proj - area_orig
    
    # Error relativo: Solo calcular donde hay fuego real para evitar división por cero/epsilon
    # Si area_orig es 0, el error relativo no tiene sentido (o es infinito).
    mask_valid = area_orig > 1e-4
    
    if mask_valid.any():
        rel_error = torch.abs(diff_area[mask_valid]) / (area_orig[mask_valid])
        rel_error_mean = rel_error.mean().item()
    else:
        rel_error_mean = 0.0
    
    return {
        'diff_mean': diff_area.mean().item(),
        'rel_error': rel_error_mean,
        'area_orig': area_orig.mean().item(),
        'area_proj': area_proj.mean().item()
    }

def cluster_count_error(pred, target_count, mask=None, threshold=0.5):
    """
    Calcula el error absoluto medio en el número de focos de incendio (clusters).
    
    Args:
        pred: (B, H, W) - Probabilidades o logits
        target_count: (B) - Número de clusters en la imagen original (pre-calculado)
        mask: (B, H, W) - Máscara de validez
        
    Returns:
        mae: Promedio de |Pred_Clusters - Orig_Clusters|
    """
    pred_bin = (pred > threshold).float()
    if mask is not None:
        pred_bin = pred_bin * mask
        
    # Mover a CPU para scipy
    pred_np = pred_bin.detach().cpu().numpy()
    
    diffs = []
    # Estructura para 8-conectividad (diagonales cuentan)
    structure = np.ones((3,3))
    
    for i in range(pred_np.shape[0]):
        # Contar clusters en predicción/reproyección
        _, n_clusters_pred = scipy_label(pred_np[i], structure=structure)
        
        n_clusters_orig = target_count[i].item()
        diffs.append(n_clusters_pred - n_clusters_orig)
        
    return np.mean(np.abs(diffs))


