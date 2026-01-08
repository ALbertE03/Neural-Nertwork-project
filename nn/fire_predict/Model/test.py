import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet3D,FireModelROI
from dataset import TSDataset, collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, jaccard_score
from tqdm import tqdm



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


def test_zonal(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['x'].to(device)
            targets = batch['label'].to(device)
            target_last = targets[:, :, -1, :, :].unsqueeze(2)
            preds = model(inputs)
            loss = criterion(preds, target_last)
            total_loss += loss.item()

            preds_prob = torch.sigmoid(preds)
            preds_bin = (preds_prob > 0.5).cpu().numpy()
            targets_np = (target_last > 0.5).cpu().numpy()
            all_preds.append(preds_bin.flatten())
            all_targets.append(targets_np.flatten())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    test_f1 = f1_score(y_true, y_pred, zero_division=0)
    test_iou = jaccard_score(y_true, y_pred, zero_division=0)
    avg_loss = total_loss / len(loader)
    print(f"Test - F1: {test_f1:.4f} | IoU: {test_iou:.4f} | Loss: {avg_loss:.4f}")
    return avg_loss, test_f1, test_iou


def sliding_window_predict(model, image, device, patch_size=192):
    # image: (T, C, H, W) o (1, T, C, H, W)
    if image.dim() == 5:
        image = image[0]  # Quitar batch si es necesario
    T, C, H, W = image.shape
    stride = patch_size  # Sin solapamiento
    out_pred = torch.zeros((1, H, W), device=device)
    out_count = torch.zeros((1, H, W), device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            patch = image[:, :, y:y+patch_size, x:x+patch_size]
            # Padding si el parche es más pequeño que patch_size
            pad_h = patch_size - patch.shape[2]
            pad_w = patch_size - patch.shape[3]
            if pad_h > 0 or pad_w > 0:
                patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h))
            patch = patch.unsqueeze(0).to(device)  # Añadir batch
            with torch.no_grad():
                pred = model(patch)  # (1, 1, H, W)
                pred = torch.sigmoid(pred)
            # Quitar padding si lo hubo
            pred = pred[:, :, :patch.shape[2], :patch.shape[3]]
            out_pred[:, y:y+patch_size, x:x+patch_size] += pred[0]
            out_count[:, y:y+patch_size, x:x+patch_size] += 1

    # Promedio si hay solapamiento (aquí no, pero es buena práctica)
    out_pred = out_pred / (out_count + 1e-6)
    return out_pred


if __name__ == "__main__":
    path = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_unet = UNet3D(in_channels=28, out_channels=1).to(device)
    model = FireModelROI(base_unet).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    load_checkpoint(model,optimizer,scheduler)
    model.eval()
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        
        pin_memory=use_cuda,
        collate_fn=collate_fn
    )

    #
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader_test:
            x = batch['x'].to(device)  # (B, T, C, H, W)
            labels = batch['label'].float().to(device)
            for i in range(x.shape[0]):
                pred_full = sliding_window_predict(model, x[i], device, patch_size=192)
                # Aquí puedes comparar pred_full con labels[i, -1] (target del último paso)
                pred = pred_full.unsqueeze(0)
                target = labels[i, -1].unsqueeze(0)
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
