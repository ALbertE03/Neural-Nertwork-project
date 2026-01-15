import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import scipy.ndimage as ndimage
from sklearn.metrics import jaccard_score, confusion_matrix, ConfusionMatrixDisplay

def get_tolerant_labels(y_true, pixels=2):
    """Dilata las etiquetas reales para permitir un margen de error."""
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, pixels)
    return ndimage.binary_dilation(y_true, structure=struct).astype(np.uint8)

input_shape = (3, 256, 256, 28)
model = build_convlstm_bottleneck128(input_shape=input_shape)
weights_path = "saved/checkpoints/best_fire_model7_convlstm.weights.h5"

if Path(weights_path).exists():
    model.load_weights(weights_path)
    print(f"[*] Pesos cargados: {weights_path}")
else:
    print("[!] Error: No se encontraron los pesos.")


ds_inference = InferenceTF(path_valid=test, cache_dir=cache_base/'test')

stats = []
g_tp_tol, g_fp_tol, g_fn_tol = 0, 0, 0
output_vis_dir = Path("inference_results")
output_vis_dir.mkdir(exist_ok=True)

print(f"Iniciando inferencia y visualización...")


for i in tqdm(range(len(ds_inference))):
    sample = ds_inference[i]
    patches = sample["patches"]
    sample_id = sample["sample_id"]
    
    # Predicción y GT (512x512)
    pred_probs = ds_inference.predict_full_image(model, patches)
    y_true_512 = ds_inference.get_ground_truth(i)
    
    y_true_bin = (y_true_512 > 0.5).astype(np.uint8)
    y_pred_bin = (pred_probs > 0.5).astype(np.uint8)
    
    # Lógica de Tolerancia para métricas
    y_true_tol = get_tolerant_labels(y_true_bin, pixels=2)
    
    # Cálculo de métricas locales
    tp_t = np.sum((y_pred_bin == 1) & (y_true_tol == 1))
    fp_t = np.sum((y_pred_bin == 1) & (y_true_tol == 0))
    fn_t = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
    
    p_tol = tp_t / (tp_t + fp_t + 1e-7)
    r_tol = tp_t / (tp_t + fn_t + 1e-7)
    f1_t = 2 * (p_tol * r_tol) / (p_tol + r_tol + 1e-7)
    
    # Acumular globales
    g_tp_tol += tp_t; g_fp_tol += fp_t; g_fn_tol += fn_t
    stats.append({"id": sample_id, "f1_tol": f1_t})


    if np.sum(y_true_bin) > 0 or np.sum(y_pred_bin) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Columna 1: Ground Truth
        axes[0].imshow(y_true_bin, cmap='inferno')
        axes[0].set_title(f"Target Real (GT)\nID: {sample_id}")
        
        # Columna 2: Predicción (Probabilidades)
        im = axes[1].imshow(pred_probs, cmap='inferno')
        axes[1].set_title(f"Predicción (Probabilidades)\nF1 Tol: {f1_t:.4f}")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Columna 3: Error Overlay (Donde falló)
        # Verde: TP, Rojo: FP, Azul: FN
        overlay = np.zeros((512, 512, 3))
        overlay[..., 1] = (y_pred_bin * y_true_bin) # Verde: Acierto exacto
        overlay[..., 0] = (y_pred_bin * (1 - y_true_tol)) # Rojo: Falso Positivo (lejos del fuego)
        overlay[..., 2] = (y_true_bin * (1 - y_pred_bin)) # Azul: Falso Negativo (olvido)
        
        axes[2].imshow(overlay)
        axes[2].set_title("Análisis de Error\nVerde:TP, Rojo:FP, Azul:FN")
        
        for ax in axes: ax.axis('off')
        
        plt.savefig(output_vis_dir / f"pred_{sample_id}.png", bbox_inches='tight')
        plt.close() 

final_f1 = 2 * ( (g_tp_tol/(g_tp_tol+g_fp_tol)) * (g_tp_tol/(g_tp_tol+g_fn_tol)) ) / ( (g_tp_tol/(g_tp_tol+g_fp_tol)) + (g_tp_tol/(g_tp_tol+g_fn_tol)) )
print(f"\n[!] Inferencia terminada. F1 Tolerante Global: {final_f1:.4f}")
pd.DataFrame(stats).to_csv("test_metrics_detailed.csv", index=False)