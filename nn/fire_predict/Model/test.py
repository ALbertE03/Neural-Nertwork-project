import torch
from torch.utils.data import DataLoader
from model import FirePredictModel
from dataset import TSDataset, collate_fn
from constants import *
from train import validate
import os

TEST_PATHS = [
    
]

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
            batch_size=1,  # Batch size 1 para evaluación precisa
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
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"No se encontró {model_path}, buscando checkpoint.pth...")
        model_path = 'checkpoint.pth'
    
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

    # 3. Evaluar
    print("\nIniciando evaluación con métricas Soft...")
    try:
        iou, f1, prec, rec, area_err = validate(model, test_loader, device, PRED_SEQ_LEN)

        print("\n" + "="*40)
        print("   RESULTADOS DE TEST (Soft Metrics)")
        print("="*40)
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
