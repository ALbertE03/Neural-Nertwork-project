import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import FirePredictModel
from utils import iou_score, f1_score, precision_score, recall_score, burned_area_error, dice_loss

def test_synthetic():
    print("Generando datos sintéticos para prueba...")
    
    # Dimensiones simuladas (más pequeñas para visualizar fácil)
    B = 2   # Batch size
    T = 5   # Time steps
    C = 28  # Channels
    H = 64  # Height
    W = 64  # Width
    PRED_HORIZON = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Crear datos aleatorios sintéticos
    # Input X: (B, T, C, H, W) - Valores normalizados aprox 0-1
    x = torch.randn(B, T, C, H, W).to(device)
    
    # Labels: (B, T, H, W) - Soft labels entre 0 y 1
    # Generamos algunos "fuegos" aleatorios (valores altos)
    labels = torch.zeros(B, T, H, W).to(device)
    # Simular un fuego en una zona cuadrada
    labels[:, :, 20:40, 20:40] = torch.rand(B, T, 20, 20).to(device) * 0.8 + 0.2 # Valores entre 0.2 y 1.0
    
    # Masks: (B, T, H, W) - 1 donde hay datos válidos
    label_mask = torch.ones(B, T, H, W).to(device)
    # Simular algunos datos faltantes
    label_mask[:, :, 0:5, 0:5] = 0
    
    # Pixel Area: (B, T) - Metros cuadrados por pixel (ej. 375x375 = ~140000)
    pixel_area = torch.ones(B, T).to(device) * 140625.0

    print(f"\nDimensiones de entrada: {x.shape}")
    print(f"Dimensiones de labels: {labels.shape}")

    # Inicializar Modelo
    print("\nInicializando modelo...")
    model = FirePredictModel(
        input_channels=C,
        hidden_channels=16,
        dropout=0.1
    ).to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parámetros: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    model.eval()


    # Input: toda la secuencia menos el horizonte de predicción
    input_seq = x[:, :-PRED_HORIZON] # (B, T-1, C, H, W)
    
    # Target: el último día 
    target = labels[:, -1]      # (B, H, W)
    target_mask = label_mask[:, -1]
    target_area = pixel_area[:, -1]

    print(f"Input Seq shape: {input_seq.shape}")
    print(f"Target shape: {target.shape}")

    # 4. Forward Pass
    print("\nEjecutando Forward Pass...")
    with torch.no_grad():
        # El modelo devuelve logits
        logits = model(input_seq) # (B, H, W)
        
        # Convertir a probabilidades (Soft Output)
        probs = torch.sigmoid(logits)

    print(f"Salida del modelo (Logits): {logits.shape}")
    print(f"Salida del modelo (Probs):  {probs.shape}")
    
    # Visualizar algunos valores
    print("\n--- Muestra de valores ---")
    print(f"Max probabilidad predicha: {probs.max().item():.4f}")
    print(f"Min probabilidad predicha: {probs.min().item():.4f}")
    print(f"Promedio probabilidad:     {probs.mean().item():.4f}")
    
    # 5. Calcular Métricas (Soft)
    print("\n--- Calculando Métricas Soft ---")
    
    # Loss (BCE)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce_val = (bce * target_mask).sum() / (target_mask.sum() + 1e-8)
    print(f"BCE Loss: {bce_val.item():.4f}")
    
    # Metrics
    iou = iou_score(probs, target, target_mask)
    f1 = f1_score(probs, target, target_mask)
    prec = precision_score(probs, target, target_mask)
    rec = recall_score(probs, target, target_mask)
    area_err = burned_area_error(probs.unsqueeze(1), target.unsqueeze(1), target_area.unsqueeze(1), target_mask.unsqueeze(1))

    print(f"Soft IoU:       {iou:.4f}")
    print(f"Soft F1:        {f1:.4f}")
    print(f"Soft Precision: {prec:.4f}")
    print(f"Soft Recall:    {rec:.4f}")
    print(f"Area Error:     {area_err:.2f} m^2")

    # 6. Visualizar Resultados
    print("\nGenerando imagen de prueba 'synthetic_output.png'...")
    
    # Tomar la primera muestra del batch
    idx = 0
    
    # Convertir a numpy para plotear
    target_np = target[idx].cpu().numpy()
    pred_np = probs[idx].cpu().numpy()
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Target
    im0 = axes[0].imshow(target_np, cmap='inferno', vmin=0, vmax=1)
    axes[0].set_title("Target (Realidad)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot Predicción
    im1 = axes[1].imshow(pred_np, cmap='inferno', vmin=0, vmax=1)
    axes[1].set_title("Predicción (Probabilidad)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot Diferencia
    diff = np.abs(target_np - pred_np)
    im2 = axes[2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title("Error Absoluto (|Target - Pred|)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('synthetic_output.png')
    print("✅ Imagen guardada: synthetic_output.png")

    print("\n✅ Prueba completada. El código funciona sin errores de dimensión.")


if __name__ == "__main__":
    test_synthetic()
