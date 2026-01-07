
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet3D
from dataset import TSDataset, collate_fn
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



def classification_metrics(pred, target, threshold=0.5):
    # pred, target: (B, 1, H, W) ambos binarios
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


if __name__ == "__main__":
    path = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=28, out_channels=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    dataset_test = TSDataset(
    path_valid=test,
    shapes=(256,256),
    train=False, augment=False, cache_dir=cache_base / "test"

)
    dataloader_test = torch.utils.data.DataLoader(
    dataset_test, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    
    pin_memory=use_cuda,
    collate_fn=collate_fn
)
    #
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader_test:
            x = batch['x'].to(device)
            labels = batch['label'].float().to(device)
            input_seq = x[:, :-1]  # (B, T-1, C, H, W)
            target = labels[:, -1].unsqueeze(1)  # (B, 1, H, W)
            pred_raw = model(input_seq)
            pred = torch.sigmoid(pred_raw)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)

    # Métricas de clasificación
    metrics = classification_metrics(pred, target, threshold=0.5)
    print("Métricas de clasificación (umbral 0.5):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- ROC Curve y AUC ---
    y_true = target.numpy().ravel()
    y_score = pred.numpy().ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"AUC ROC: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    # --- Precision-Recall Curve y AUC-PR ---
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)
    print(f"AUC-PR (Average Precision): {auc_pr:.4f}")

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {auc_pr:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

    # Visualización de algunos ejemplos
    num_examples = min(3, pred.shape[0])
    plt.figure(figsize=(12, 6))
    for i in range(num_examples):
        plt.subplot(3, num_examples, i+1)
        plt.title('Predicción (binaria)')
        plt.imshow((pred[i, 0] > 0.5).numpy(), cmap='gray')
        plt.axis('off')
        plt.subplot(3, num_examples, i+1+num_examples)
        plt.title('Target (binario)')
        plt.imshow(target[i, 0].numpy(), cmap='gray')
        plt.axis('off')
        plt.subplot(3, num_examples, i+1+2*num_examples)
        plt.title('Error (pred - target)')
        err = ((pred[i, 0] > 0.5).float() - target[i, 0]).numpy()
        plt.imshow(err, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar(shrink=0.6)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Visualización: pred binaria, target y mapa de error para los primeros ejemplos
    plt.figure(figsize=(12, 6))
    num_examples = min(3, pred.shape[0])
    for i in range(num_examples):
        plt.subplot(3, num_examples, i+1)
        plt.title('Predicción (binaria)')
        plt.imshow((pred[i, 0] > 0.5).cpu().numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(3, num_examples, i+1+num_examples)
        plt.title('Target (binario)')
        plt.imshow(target[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')

        plt.subplot(3, num_examples, i+1+2*num_examples)
        plt.title('Error (pred - target)')
        err = ((pred[i, 0] > 0.5).float() - target[i, 0]).cpu().numpy()
        plt.imshow(err, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar(shrink=0.6)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
