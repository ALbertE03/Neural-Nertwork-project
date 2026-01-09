import torch
import numpy as np
import matplotlib.pyplot as plt
from fire_predict.Model.test_dataset import TSDatasetTest
from constants import MAX_INPUT_SEQ_LEN, PRED_SEQ_LEN
from model import UNet3D, FireModelROI
from train import SoftZoneLoss
from sklearn.metrics import roc_curve, auc, recall_score

def load_checkpoint_test(model, optimizer, scheduler, path,device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"Checkpoint loaded from {path}")

def compute_metrics(pred, target):
    pred_bin = (pred > 0.5).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)
    tp = np.sum((pred_bin == 1) & (target_bin == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (target_bin == 1))
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return iou, f1, tp, fp, fn

def plot_roc_curve(y_true, y_pred, idx):
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic {idx}')
        plt.legend(loc="lower right")
        plt.show()

def plot_recall_curve(y_true, y_pred, idx):
        thresholds = np.linspace(0, 1, 50)
        recalls = [recall_score(y_true.ravel(), (y_pred.ravel() > thr).astype(np.uint8)) for thr in thresholds]
        plt.figure()
        plt.plot(thresholds, recalls, color='blue', lw=2)
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title(f'Recall vs Threshold {idx}')
        plt.show()
def visualize_results(pred, target, idx, save_path=None):
    
    pred_bin = (pred > 0.5).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)
    tp = (pred_bin == 1) & (target_bin == 1)
    fp = (pred_bin == 1) & (target_bin == 0)
    fn = (pred_bin == 0) & (target_bin == 1)
    vis = np.zeros((3, *pred_bin.shape), dtype=np.uint8)
    vis[0] = tp * 255
    vis[1] = fp * 255
    vis[2] = fn * 255
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(vis.transpose(1,2,0))
    axs[0].set_title('TP(red), FP(green), FN(blue)')
    axs[1].imshow(pred_bin[0], cmap='gray')
    axs[1].set_title('Prediction')
    axs[2].imshow(target_bin[0], cmap='gray')
    axs[2].set_title('Target')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/result_{idx}.png")
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_unet = UNet3D(in_channels=28, out_channels=1).to(device)
    model = FireModelROI(base_unet).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    load_checkpoint_test(model, optimizer, scheduler, 'saved/checkpoints/checkpoint1.pth', device=device)
    path_valid = [""]
    cache_dir = "cache_test"
    dataset = TSDatasetTest(path_valid, cache_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    iou_list, f1_list, auc_list, recall_list = [], [], [], []
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        T = batch.shape[1]
        for t in range(T-1):
            x = batch[:, t].unsqueeze(0).unsqueeze(0) # (1,1,1, C, H, W)
            y_true = batch[:, t+1, dataset._fire_channel_idx] # (1, H, W)
            with torch.no_grad():
                y_pred = model(x).sigmoid().cpu().numpy()
            y_pred = y_pred[0, 0] # (H, W)
            y_true = y_true.cpu().numpy() # (H, W)
            iou, f1, tp, fp, fn = compute_metrics(y_pred, y_true)
            iou_list.append(iou)
            f1_list.append(f1)
            # AUROC
            fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
            roc_auc = auc(fpr, tpr)
            auc_list.append(roc_auc)
            # Recall (umbral 0.5)
            recall = recall_score(y_true.ravel(), (y_pred.ravel() > 0.5).astype(np.uint8))
            recall_list.append(recall)
            visualize_results(y_pred[None, ...], y_true[None, ...], f"{batch_idx}_{t}")
            plot_roc_curve(y_true, y_pred, f"{batch_idx}_{t}")
            plot_recall_curve(y_true, y_pred, f"{batch_idx}_{t}")
            
    print(f"Mean IoU: {np.mean(iou_list):.4f}")
    print(f"Mean F1: {np.mean(f1_list):.4f}")
    print(f"Mean AUROC: {np.mean(auc_list):.4f}")
    print(f"Mean Recall: {np.mean(recall_list):.4f}")

if __name__ == "__main__":
    main()
