import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import FirePredictModel
from utils import compute_multiclass_metrics
from constants import FIRE_THRESHOLDS, NUM_CLASSES, UNDERESTIMATION_COST, OVERESTIMATION_COST

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
    x = torch.randn(B, T, C, H, W).to(device)
    
    # Labels: Necesitamos clases enteras 0, 1, 2, 3
    # Generamos floats 0-1 y digitalizamos como en dataset.py
    raw_labels = torch.zeros(B, T, H, W).float()
    
    # Simular evolución temporal:
    # t=0: poco fuego, t=4: mucho fuego
    for t in range(T):
        intensity = 0.2 + (t / T) * 0.8 # Sube intensidad con el tiempo
        raw_labels[:, t, 20:40, 20:40] = torch.rand(B, 20, 20) * intensity
    
    labels_np = raw_labels.numpy()
    labels_discretized = np.digitize(labels_np, FIRE_THRESHOLDS).astype(np.int64)
    labels = torch.from_numpy(labels_discretized).to(device)

    print(f"\nDimensiones de entrada: {x.shape}")
    print(f"Dimensiones de labels: {labels.shape}")
    print(f"Clases únicas en labels: {torch.unique(labels)}")

    # Inicializar Modelo
    print("\nInicializando modelo...")
    model = FirePredictModel(
        input_channels=C,
        hidden_channels=64,
        dropout=0.1,
        num_classes=NUM_CLASSES
    ).to(device)
    
    model.eval()

    # Input: secuencia hasta T-1
    input_seq = x[:, :-PRED_HORIZON] # (B, T-1, C, H, W)
    
    # Target Actual (T) y Anterior (T-1) para ver transiciones
    target = labels[:, -1]      # (B, H, W)
    prev_target = labels[:, -2] # (B, H, W)

    print(f"Input Seq shape: {input_seq.shape}")

    # 4. Forward Pass
    print("\nEjecutando Forward Pass...")
    with torch.no_grad():
        # El modelo devuelve logits (B, NumClasses, H, W)
        logits = model(input_seq) 
        
    print(f"Salida del modelo (Logits): {logits.shape}")

    # 5. Calcular Métricas Multiclase
    print("\n--- Calculando Métricas Multiclase ---")
    
    acc, f1, cm = compute_multiclass_metrics(logits, target, NUM_CLASSES)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("Matriz de Confusión (Truth vs Pred):")
    print(cm.numpy().astype(int))
    
    # 6. Calcular Matriz de Transición Temporal
    # Queremos ver probabilidad P(Pred=j | PrevTruth=i)
    # "Dado que ayer era clase i, ¿qué predijo el modelo hoy?"
    pred_classes = torch.argmax(logits, dim=1) # (B, H, W)
    
    # Achatamos
    prev_t_flat = prev_target.reshape(-1)
    p_flat = pred_classes.reshape(-1)
    
    idx_tm = prev_t_flat * NUM_CLASSES + p_flat
    tm_counts = torch.bincount(idx_tm, minlength=NUM_CLASSES**2)
    transition_matrix = tm_counts.view(NUM_CLASSES, NUM_CLASSES).float()
    
    # Normalizar por fila para tener probabilidades
    # Row sum = Total pixels que ayer eran clase i
    row_sums = transition_matrix.sum(dim=1, keepdim=True)
    tm_probs = transition_matrix / (row_sums + 1e-8)
    
    print("\n--- Matriz de Transiciones Temporales (PrevTruth -> CurrPred) ---")
    print("Filas: Verdad en T-1 (Ayer)")
    print("Cols:  Predicción en T (Hoy)")
    print(tm_probs.numpy())
    
    print("\nInterpretación:")
    print("Elemento [i, j] es la probabilidad de que el modelo prediga clase 'j' hoy,")
    print("dado que ayer la realidad era clase 'i'.")
    print("Ejemplo: [0, 1] alto significa que el modelo tiende a predecir inicio de fuego.")
    print("Ejemplo: [3, 3] alto significa que el modelo predice persistencia de fuego alto.")

if __name__ == "__main__":
    test_synthetic()

