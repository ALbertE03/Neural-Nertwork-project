import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (f1_score, jaccard_score, precision_score, 
                             recall_score, accuracy_score, precision_recall_curve, 
                             auc, confusion_matrix, ConfusionMatrixDisplay)

input_shape = (3, 256, 256, 28)
model = build_unet3d(input_shape=input_shape)

weights_path = "saved/checkpoints/best_fire_model2.weights.h5"  
if Path(weights_path).exists():
    model.load_weights(weights_path)
    print(f"[*] Pesos cargados correctamente desde {weights_path}")
else:
    print("[!] Error: No se encontraron los pesos.")

ds_inference = InferenceTF(
    path_valid=test,
    cache_dir=cache_base/'test'
)



stats = []
all_y_true = []
all_y_pred = []

print(f"Iniciando inferencia en {len(ds_inference)} muestras...")

for i in tqdm(range(len(ds_inference))):
    sample = ds_inference[i]
    patches = sample["patches"]
    sample_id = sample["sample_id"]
    
    pred_512 = ds_inference.predict_full_image(model, patches)
    y_true_512 = ds_inference.get_ground_truth(i)
    
    y_true_flat = (y_true_512 > 0.5).astype(np.uint8).flatten()
    y_pred_probs = pred_512.flatten()
    y_pred_bin = (y_pred_probs > 0.5).astype(np.uint8)
    

    all_y_true.append(y_true_flat)
    all_y_pred.append(y_pred_probs)

    m = {
        "id": sample_id,
        "f1": f1_score(y_true_flat, y_pred_bin, zero_division=1),
        "iou": jaccard_score(y_true_flat, y_pred_bin, zero_division=1),
        "precision": precision_score(y_true_flat, y_pred_bin, zero_division=1),
        "recall": recall_score(y_true_flat, y_pred_bin, zero_division=1),
        "auc_pr": auc(*precision_recall_curve(y_true_flat, y_pred_probs)[1::-1])
    }
    stats.append(m)


all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)
all_y_pred_bin = (all_y_pred > 0.5).astype(np.uint8)

print("\n" + "="*40)
print("MÉTRICAS GLOBALES (TOTAL DATASET)")
print("="*40)

global_precision, global_recall, _ = precision_recall_curve(all_y_true, all_y_pred)
global_auc_pr = auc(global_recall, global_precision)

print(f"Global IoU (Clase 1):  {jaccard_score(all_y_true, all_y_pred_bin):.4f}")
print(f"Global F1-Score:      {f1_score(all_y_true, all_y_pred_bin):.4f}")
print(f"Global Precision:     {precision_score(all_y_true, all_y_pred_bin):.4f}")
print(f"Global Recall:        {recall_score(all_y_true, all_y_pred_bin):.4f}")
print(f"Global AUC-PR:        {global_auc_pr:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

cm = confusion_matrix(all_y_true, all_y_pred_bin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fuego', 'Fuego'])
disp.plot(ax=ax[0], cmap='Blues', values_format='d')
ax[0].set_title("Matriz de Confusión Global (Píxeles)")

ax[1].plot(global_recall, global_precision, color='red', lw=2, label=f'AUC-PR: {global_auc_pr:.4f}')
ax[1].fill_between(global_recall, global_precision, alpha=0.2, color='red')
ax[1].set_title("Curva Precision-Recall Global")
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].legend()

plt.tight_layout()
plt.show()

df_stats = pd.DataFrame(stats)
df_stats.to_csv("test_metrics_report_detailed.csv", index=False)