import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet3D

def dummy_batch(batch_size=3, seq_len=5, channels=28, height=32, width=32):
    x = torch.randn(batch_size, seq_len, channels, height, width)
    label = (torch.rand(batch_size, seq_len, height, width) > 0.95).float()  # 5% fire
    return {'x': x, 'label': label}

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=28, out_channels=1).to(device)
    model.eval()

    batch = dummy_batch(batch_size=3, seq_len=5, channels=28, height=32, width=32)
    x = batch['x'].to(device)
    labels = batch['label'].float().to(device)
    # UNet3D espera (B, C, T, H, W)
    input_seq = x
    target = labels[:, -1].unsqueeze(1)  # (B, 1, H, W)

    with torch.no_grad():
        pred_logits = model(input_seq)
        probs = torch.sigmoid(pred_logits)
        preds = (probs > 0.5).float()

    # Métricas
    tp = (preds * target).sum().item()
    fp = (preds * (1 - target)).sum().item()
    fn = ((1 - preds) * target).sum().item()
    tn = ((1 - preds) * (1 - target)).sum().item()
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"Matriz de confusión: [[TN={int(tn)}, FP={int(fp)}], [FN={int(fn)}, TP={int(tp)}]]")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Visualización
    plt.figure(figsize=(12, 4))
    num_examples = min(3, preds.shape[0])
    for i in range(num_examples):
        plt.subplot(3, num_examples, i+1)
        plt.title('Predicción')
        plt.imshow(preds[i,0].cpu().numpy(), cmap='hot')
        plt.axis('off')
        plt.subplot(3, num_examples, i+1+num_examples)
        plt.title('Target')
        plt.imshow(target[i,0].cpu().numpy(), cmap='hot')
        plt.axis('off')
        plt.subplot(3, num_examples, i+1+2*num_examples)
        plt.title('Diferencia')
        plt.imshow((preds[i,0] - target[i,0]).cpu().numpy(), cmap='bwr')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
