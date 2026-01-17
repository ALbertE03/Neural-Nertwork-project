import torch
import numpy as np
import rasterio
import tensorflow as tf
from rasterio.enums import Resampling
from dataset import TSDataset

class InferenceTS(TSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_size = 512
        self.patch_size = 256

    def _read_and_rescale(self, dataset, bands, window=None, is_label=False):
        """
        Lee y reescala una imagen a 512x512.
        """
        resampling_mode = Resampling.nearest if is_label else Resampling.bilinear
        
        data = dataset.read(
            bands,
            out_shape=(len(bands) if bands else dataset.count, self.target_size, self.target_size),
            resampling=resampling_mode
        )
        return self._normalize(data, is_label=is_label)

    def __getitem__(self, idx):
        info = self.samples[idx]
        region = self.raw_paths[info["region"]]
        t_start = info["t_start"]

        full_seq_512 = [] 
        
        for t in range(t_start, t_start + self.seq_len):
            # VIIRS Day 
            with rasterio.open(region["VIIRS_Day"][t]) as dsrc:
                day = self._read_and_rescale(dsrc, [1, 2, 3, 4, 5, 6])
                fire_today = self._read_and_rescale(dsrc, [7], is_label=True)

            # VIIRS Night
            with rasterio.open(region["VIIRS_Night"][t]) as nsrc:
               
                num_bands = nsrc.count
                night = self._read_and_rescale(nsrc, [num_bands-1, num_bands])

            # FirePred
            with rasterio.open(region["FirePred"][t]) as fsrc:
                firep = self._read_and_rescale(fsrc, None)

            combined_t = np.concatenate([day, fire_today, night, firep], axis=0)
            full_seq_512.append(combined_t)

        # Convertir a tensor
        full_seq_tensor = np.stack(full_seq_512)

        # Dividir en 4 parches de 256x256
        patches = []
        # Cuadrantes: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        offsets = [(0, 0), (0, 256), (256, 0), (256, 256)]
        
        for (y, x) in offsets:
            patch = full_seq_tensor[:, :, y:y+256, x:x+256]
            patches.append(torch.from_numpy(patch).float())

        return {
            "patches": torch.stack(patches), # [4, T, C, 256, 256]
            "sample_id": info['sample_id']
        }

    def reconstruct_image(self, model, patches_tensor, device="cuda"):
        """
        Toma el tensor de 4 parches, corre el modelo y une los resultados.
        """
        model.eval()
        model.to(device)
        preds = []
        
        with torch.no_grad():
            # patches_tensor: [4, T, C, 256, 256]
            for i in range(4):
                input_patch = patches_tensor[i].unsqueeze(0).to(device)
                output = model(input_patch) # Salida [1, 1, 256, 256]
                preds.append(output.squeeze().cpu().numpy())

        # Unir patches
        top = np.concatenate([preds[0], preds[1]], axis=1)
        bottom = np.concatenate([preds[2], preds[3]], axis=1)
        full_reconstruction = np.concatenate([top, bottom], axis=0)
        
        return full_reconstruction # [512, 512]




class InferenceTF(InferenceTS):
    def predict_full_image(self, model, patches_tensor):
        """
        Predice los 4 parches y los une. 
        """
        patches_np = patches_tensor.numpy()
        patches_tf = np.transpose(patches_np, (0, 1, 3, 4, 2))
        
        preds = []
        for i in range(4):
            patch = patches_tf[i:i+1] # Shape: (1, T, 256, 256, C)
            
            # (1, 256, 256, 1)
            p = model.predict(patch, verbose=0)
        
            p_prob = tf.nn.sigmoid(p).numpy()
            preds.append(np.squeeze(p_prob))

        # Reconstrucción 2x2 para formar la imagen de 512x512
        top = np.concatenate([preds[0], preds[1]], axis=1)
        bottom = np.concatenate([preds[2], preds[3]], axis=1)
        return np.concatenate([top, bottom], axis=0) # [512, 512]

    def get_ground_truth(self, idx):
        """Extrae el target real de 512x512 reescalado correctamente"""
        info = self.samples[idx]
        region = self.raw_paths[info["region"]]
        t_target = info["t_start"] + self.seq_len
        
        with rasterio.open(region["VIIRS_Day"][t_target]) as dsrc:
            
            y_true = dsrc.read(
                7, 
                out_shape=(self.target_size, self.target_size), 
                resampling=Resampling.nearest
            )
            return self._normalize(y_true, is_label=True)

    def evaluate_tolerantly(self, y_true, y_pred_prob, tol_ksize=5, threshold=0.5):
        """
        Calcula F1 y Recall usando la misma lógica de tolerancia que el entrenamiento.
        y_true: [512, 512]
        y_pred_prob: [512, 512] (probabilidades 0-1)
        """
        # Convertir a tensores de 4D para pooling [Batch, H, W, Channels]
        y_true_t = tf.cast(y_true[np.newaxis, ..., np.newaxis], tf.float32)
        y_pred_t = tf.cast(y_pred_prob[np.newaxis, ..., np.newaxis], tf.float32)
        
        y_pred_bin = tf.cast(y_pred_t > threshold, tf.float32)

        # Crear máscara de tolerancia 
        y_true_tol = tf.nn.max_pool2d(
            y_true_t, ksize=tol_ksize, strides=1, padding='SAME'
        )

        #  Métricas Tolerantes 
        # True Positives correcta dentro de zona tolerante
        tp = tf.reduce_sum(y_pred_bin * y_true_tol)
        
        # False Positives Fuera de la zona de tolerancia
        fp = tf.reduce_sum(y_pred_bin * (1.0 - y_true_tol))
        
        # False Negatives
        y_pred_tol = tf.nn.max_pool2d(y_pred_bin, ksize=tol_ksize, strides=1, padding='SAME')
        fn = tf.reduce_sum(y_true_t * (1.0 - y_pred_tol))

        # Cálculos finales
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        return {
            "f1_tolerant": f1.numpy(),
            "recall_tolerant": recall.numpy(),
            "precision_tolerant": precision.numpy()
        }