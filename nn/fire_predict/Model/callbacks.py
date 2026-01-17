import tensorflow as tf
import json
import matplotlib.pyplot as plt
import os

class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.history_dict = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(self.model.optimizer.learning_rate.numpy())

        current_logs = {
            "epoch": epoch + 1,
            "lr": lr,
        **{k: float(v) for k, v in logs.items()}}
        self.history_dict.append(current_logs)
        
        with open(self.filepath, 'w') as f:
            json.dump(self.history_dict, f, indent=4)

class VisualVerifyCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, save_dir='visual_logs'):
        super().__init__()
        self.dataset = dataset
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Tomar un batch de validación
        for inputs, targets in self.dataset.take(1):
            preds = self.model.predict(inputs, verbose=0)
            preds = tf.nn.sigmoid(preds).numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Frame de entrada (Canal 0 del último paso temporal)
            axes[0].imshow(inputs[0, -1, :, :, 0], cmap='viridis')
            axes[0].set_title("Input (Last Frame)")
            
            # Máscara Real
            axes[1].imshow(targets[0, :, :, 0], cmap='inferno')
            axes[1].set_title("Ground Truth")
            
            # Predicción
            axes[2].imshow(preds[0, :, :, 0], cmap='inferno')
            axes[2].set_title(f"Prediction (Epoch {epoch+1})")
            
            for ax in axes: ax.axis('off')
            
            plt.savefig(f"{self.save_dir}/epoch_{epoch+1}.png")
            plt.close()

