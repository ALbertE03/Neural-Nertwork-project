
model = build_unet3d(input_shape=(3, 256, 256, 28))


weights_path = "saved/checkpoints/best_fire_model2.weights.h5"  
model.load_weights(weights_path)
print("Pesos cargados correctamente.")


ds_inference= InferenceTF(
    path_valid=test,
    cache_dir=cache_base/'test'
)



from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# Listas para métricas globales
all_y_true = []
all_y_pred = []

for i in tqdm(range(len(ds_inference))):
    sample = ds_inference[i]
    patches = sample["patches"]
    sample_id = sample["sample_id"]
    
    # 1. Predicción y GT
    pred_512 = ds_inference.predict_full_image(model, patches)
    y_true_512 = ds_inference.get_ground_truth(i)
    
    # Aplanamos para métricas (0 a 1)
    y_true_flat = (y_true_512 > 0.5).astype(int).flatten()
    y_pred_flat = pred_512.flatten()
    
    # Guardar para curva global
    all_y_true.extend(y_true_flat)
    all_y_pred.extend(y_pred_flat)
    
    # 2. Calcular PR-Curve de esta muestra
    precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_flat)
    pr_auc = auc(recall, precision)
    ap_score = average_precision_score(y_true_flat, y_pred_flat)
    
    # 3. Visualización 1x3 (Original, Predicción, Curva PR)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # GT
    ax[0].imshow(y_true_512, cmap='inferno')
    ax[0].set_title(f"Original (GT)\n{sample_id}")
    ax[0].axis('off')
    
    # Predicción
    im = ax[1].imshow(pred_512, cmap='inferno')
    ax[1].set_title(f"Predicción Probabilística")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].axis('off')
    
    # Curva PR
    ax[2].plot(recall, precision, color='blue', lw=2, label=f'AUC-PR = {pr_auc:.4f}')
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_title('Curva Precision-Recall')
    ax[2].legend(loc="lower left")
    ax[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def plot_global_metrics(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='red', alpha=0.8, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='red')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Curva PR Global (Average Precision = {ap:.4f})')
    plt.grid(True)
    plt.show()

# Llamada al final
plot_global_metrics(all_y_true, all_y_pred)