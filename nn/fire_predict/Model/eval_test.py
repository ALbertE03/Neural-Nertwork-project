import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import scipy.ndimage as ndimage
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc


def get_tolerant_labels(y_true, pixels=2):
    """Dilata las etiquetas reales para permitir un margen de error espacial (Buffer)."""
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, pixels)
    return ndimage.binary_dilation(y_true, structure=struct).astype(np.uint8)

def calc_metrics_group(tp, fp, fn):
    """Calcula el set estándar de métricas de segmentación."""
    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return prec, rec, f1, iou


input_shape = (3, 256, 256, 28)
model = build_convlstm_bottleneck128(input_shape=input_shape)
weights_path = "saved/checkpoints/best_fire_model7_convlstm.weights.h5"

if Path(weights_path).exists():
    model.load_weights(weights_path)
    print(f"[*] Pesos cargados correctamente desde {weights_path}")
else:
    print("[!] Error: No se encontraron los pesos en la ruta especificada.")

ds_inference = InferenceTF(path_valid=test, cache_dir=cache_base/'test')


metrics_accum = {
    'tp_ex': 0, 'fp_ex': 0, 'fn_ex': 0, 'tn_ex': 0,
    'tp_tol': 0, 'fp_tol': 0, 'fn_tol': 0
}

all_probs = []  
all_targets = []
output_vis_dir = Path("inference_results")
output_vis_dir.mkdir(exist_ok=True)


first_layer_weights = model.layers[1].get_weights()[0]

feature_importance_global = np.mean(np.abs(first_layer_weights), axis=(0, 1, 3))

print(f"Iniciando Inferencia Completa en {len(ds_inference)} muestras...")


for i in tqdm(range(len(ds_inference))):
    sample = ds_inference[i]
    patches = sample["patches"]
    sample_id = sample["sample_id"]
    
    #  Predicción y GT
    pred_probs = ds_inference.predict_full_image(model, patches)
    y_true_512 = ds_inference.get_ground_truth(i)
    
    y_true_bin = (y_true_512 > 0).astype(np.uint8)
    y_pred_bin = (pred_probs > 0.5).astype(np.uint8)
    y_true_tol = get_tolerant_labels(y_true_bin, pixels=2)

    #  Contadores Exactos
    tp_e = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    fp_e = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
    fn_e = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
    tn_e = np.sum((y_pred_bin == 0) & (y_true_bin == 0))

    # Contadores Tolerantes
    tp_t = np.sum((y_pred_bin == 1) & (y_true_tol == 1))
    fp_t = np.sum((y_pred_bin == 1) & (y_true_tol == 0))
   
    fn_t = fn_e 

 
    metrics_accum['tp_ex'] += tp_e; metrics_accum['fp_ex'] += fp_e
    metrics_accum['fn_ex'] += fn_e; metrics_accum['tn_ex'] += tn_e
    metrics_accum['tp_tol'] += tp_t; metrics_accum['fp_tol'] += fp_t
    metrics_accum['fn_tol'] += fn_t

    if i % 3 == 0:
        all_probs.append(pred_probs.flatten()[::25])
        all_targets.append(y_true_bin.flatten()[::25])


    if (np.sum(y_true_bin) > 0 or np.sum(y_pred_bin) > 0) and i % 10 == 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(y_true_bin, cmap='inferno'); axes[0].set_title("GT")
        axes[1].imshow(pred_probs, cmap='inferno'); axes[1].set_title("Pred")
        err = np.zeros((512, 512, 3))
        err[..., 1] = (y_pred_bin * y_true_bin) # Verde: TP
        err[..., 0] = (y_pred_bin * (1 - y_true_tol)) # Rojo: FP
        err[..., 2] = (y_true_bin * (1 - y_pred_bin)) # Azul: FN
        axes[2].imshow(err); axes[2].set_title("Error (V:TP, R:FP, A:FN)")
        plt.savefig(output_vis_dir / f"sample_{sample_id}.png")
        plt.close()



p_ex, r_ex, f1_ex, iou_ex = calc_metrics_group(metrics_accum['tp_ex'], metrics_accum['fp_ex'], metrics_accum['fn_ex'])
p_tol, r_tol, f1_tol, iou_tol = calc_metrics_group(metrics_accum['tp_tol'], metrics_accum['fp_tol'], metrics_accum['fn_tol'])


all_targets_cat = np.concatenate(all_targets)
all_probs_cat = np.concatenate(all_probs)

fig = plt.figure(figsize=(22, 14))
gs = fig.add_gridspec(2, 3)


ax0 = fig.add_subplot(gs[0, 0])
cm = np.array([[metrics_accum['tn_ex'], metrics_accum['fp_ex']], 
               [metrics_accum['fn_ex'], metrics_accum['tp_ex']]])
ConfusionMatrixDisplay(cm, display_labels=['Fondo', 'Fuego']).plot(ax=ax0, cmap='Reds', values_format='d')
ax0.set_title("Matriz de Confusión Global (Píxeles)")


ax1 = fig.add_subplot(gs[0, 1])
ch_indices = np.arange(28)
ax1.bar(ch_indices, feature_importance_global, color='darkcyan')
ax1.set_xticks(ch_indices)
ax1.set_xticklabels([f"CH_{i}" for i in ch_indices], rotation=90, fontsize=8)
ax1.set_title("Importancia de Features (Input Layer)")
ax1.set_ylabel("Magnitud Media de Pesos")


ax2 = fig.add_subplot(gs[0, 2])
prec_c, rec_c, _ = precision_recall_curve(all_targets_cat, all_probs_cat)
ax2.plot(rec_c, prec_c, color='darkgreen', lw=3, label=f'PR AUC: {auc(rec_c, prec_c):.3f}')
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Curva Precision-Recall Global")
ax2.legend()


ax3 = fig.add_subplot(gs[1, :])
ax3.axis('off')
metric_data = [
    ["F1-Score4", f"{f1_ex:.4f}", f"{f1_tol:.4f}"],
    ["IoU ", f"{iou_ex:.4f}", f"{iou_tol:.4f}"],
    ["Recall ", f"{r_ex:.4f}", f"{r_tol:.4f}"],
    ["Precision ", f"{p_ex:.4f}", f"{p_tol:.4f}"]
]
table = ax3.table(cellText=metric_data, colLabels=["Métrica", "Exacto (Píxel)", "Tolerante "], 
                  loc='center', cellLoc='center')
table.scale(1, 4)
table.set_fontsize(14)
ax3.set_title("Resumen de Performance del Modelo ConvLSTM", pad=20, fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# Guardar resultados
pd.DataFrame([metrics_accum]).to_csv("pixel_counts_final.csv", index=False)