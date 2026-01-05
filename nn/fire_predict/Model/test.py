import torch
from torch.utils.data import DataLoader
from model import FirePredictModel
from dataset import TSDataset, collate_fn
from constants import *
from train import validate
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

TEST_PATHS = [
    
]

def visualize_prediction(input_seq, target, pred, save_path="test_visualization.png"):
    """
    Genera una visualización comparativa:
    1. Último frame de entrada (Contexto)
    2. Target Real (Lo que debió pasar)
    3. Predicción del Modelo (Lo que el modelo cree)
    4. Diferencia (Error)
    """
    # Convertir a numpy y mover a CPU
    # input_seq: (B, T, C, H, W) -> Tomamos último frame, canal 0 (ej. fuego previo)
    last_input = input_seq[0, -1, 0].cpu().numpy()
    
    target_np = target[0].cpu().numpy()
    pred_np = torch.sigmoid(pred[0]).cpu().numpy()
    
    # Calcular diferencia absoluta
    diff = np.abs(target_np - pred_np)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Input (Último estado conocido)
    im0 = axes[0].imshow(last_input, cmap='inferno', vmin=0, vmax=1)
    axes[0].set_title("Input (t-1)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Target (Realidad)
    im1 = axes[1].imshow(target_np, cmap='inferno', vmin=0, vmax=1)
    axes[1].set_title("Target Real (t)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Predicción
    im2 = axes[2].imshow(pred_np, cmap='inferno', vmin=0, vmax=1)
    axes[2].set_title("Predicción Modelo (t)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. Diferencia
    im3 = axes[3].imshow(diff, cmap='Reds', vmin=0, vmax=1)
    axes[3].set_title("Error Absoluto")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualización guardada en: {save_path}")
    plt.close()

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # 1. Cargar Datos
    if not TEST_PATHS:
        print("⚠️ No se han definido rutas de test en TEST_PATHS.")
        if DATA_PATHS:
            print("Usando DATA_PATHS de constants.py como fallback...")
            paths = DATA_PATHS
        else:
            print("❌ Error: Debes editar test.py y añadir las rutas en la lista TEST_PATHS.")
            return
    else:
        paths = TEST_PATHS

    print(f"Cargando datos desde: {paths}")
    try:
        test_dataset = TSDataset(paths, SHAPES)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1,  
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0  
        )
        print(f"Datos de test cargados: {len(test_dataset)} muestras.")
    except Exception as e:
        print(f"Error cargando el dataset: {e}")
        return

    # 2. Cargar Modelo
    model = FirePredictModel(
        input_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT
    ).to(device)

    # Intentar cargar el mejor modelo primero
    model_path = 'saved/best_model.pth'
    if not os.path.exists(model_path):
        print(f"No se encontró {model_path}, buscando checkpoint...")
        # Buscar el último checkpoint
        checkpoint_dir = 'saved/checkpoints'
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if files:
                files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                model_path = os.path.join(checkpoint_dir, files[-1])
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Modelo cargado exitosamente desde {model_path}")
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            return
    else:
        print("❌ No se encontró ningún archivo de modelo (.pth). Entrena primero.")
        return

    # 3. Evaluar y Visualizar
    print("\nIniciando evaluación...")
    model.eval()
    
    # Visualizar el primer batch
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x = batch['x'].to(device)
            labels = batch['label'].to(device)
            
            input_seq = x[:, :-PRED_SEQ_LEN]
            target = labels[:, -1]
            
            pred = model(input_seq)
            
            # Generar visualización para el primer ejemplo
            if i == 0:
                visualize_prediction(input_seq, target, pred)
                break # Solo visualizamos uno y salimos para no tardar
    
    # Calcular métricas globales
    try:
        val_loss, iou, f1, prec, rec, area_err = validate(model, test_loader, device, PRED_SEQ_LEN)

        print("\n" + "="*40)
        print("   RESULTADOS DE TEST (Soft Metrics)")
        print("="*40)
        print(f"Loss:                     {val_loss:.4f}")
        print(f"IoU (Intersección/Unión): {iou:.4f}")
        print(f"F1 Score:                 {f1:.4f}")
        print(f"Precision:                {prec:.4f}")
        print(f"Recall:                   {rec:.4f}")
        print(f"Error de Área (Abs):      {area_err:.4e}")
        print("="*40)
        
    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
