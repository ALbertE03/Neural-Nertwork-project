import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (Batch, 1, H, W) - Logits from model
        # targets: (Batch, 1, H, W) - Binary masks (0, 1)
        
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        # inputs: Logits
        # targets: Binary Mask
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        
        return bce_loss + dice_loss

def calculate_iou(preds, labels, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for a batch of predictions.
    preds: Logits or Probabilities (Batch, 1, H, W)
    labels: Binary Mask (Batch, 1, H, W)
    """
    # Convert logits to binary predictions
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    
    # Flatten
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    
    if union == 0:
        return 1.0 # Perfect match (both empty)
        
    iou = intersection / union
    return iou.item()
