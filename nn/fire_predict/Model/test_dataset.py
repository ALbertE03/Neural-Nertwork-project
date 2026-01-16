import torch
import numpy as np
import rasterio
from rasterio.enums import Resampling
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score

class InferenceTS(TSDatasetFlat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_size = 512
        self.patch_size = 256

    def _read_and_rescale(self, dataset, bands, window=None, is_label=False):
        """
        Lee y reescala una imagen a 512x512 independientemente de su tamaño original.
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

        full_seq_512 = [] # [C, 512, 512]
        
        for t in range(t_start, t_start + self.seq_len):
        
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

        full_seq_tensor = np.stack(full_seq_512)


        patches = []
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
            for i in range(4):
                input_patch = patches_tensor[i].unsqueeze(0).to(device)
                output = model(input_patch) # [1, 1, 256, 256]
                preds.append(output.squeeze().cpu().numpy())

        # Unir patches
        top = np.concatenate([preds[0], preds[1]], axis=1)
        bottom = np.concatenate([preds[2], preds[3]], axis=1)
        full_reconstruction = np.concatenate([top, bottom], axis=0)
        
        return full_reconstruction # [512, 512]





class InferenceTF(InferenceTS):
    def predict_full_image(self, model, patches_tensor):
        # patches_tensor: [4, T, C, 256, 256] -> PyTorch
        patches_np = patches_tensor.numpy()
        # Transponer a formato TF: [4, T, 256, 256, C]
        patches_tf = np.transpose(patches_np, (0, 1, 3, 4, 2))
        
        preds = []
        for i in range(4):
            patch = patches_tf[i:i+1] # Shape: (1, 3, 256, 256, 28)
            
            
            # Predicción: salida (1, 256, 256, 1)
            p = model.predict(patch, verbose=0)
            preds.append(np.squeeze(p))

        # Reconstrucción 2x2
        top = np.concatenate([preds[0], preds[1]], axis=1)
        bottom = np.concatenate([preds[2], preds[3]], axis=1)
        return np.concatenate([top, bottom], axis=0)

    def get_ground_truth(self, idx):
        """Extrae el target real de 512x512 para comparar"""
        info = self.samples[idx]
        region = self.raw_paths[info["region"]]
        t_target = info["t_start"] + self.seq_len
        
        with rasterio.open(region["VIIRS_Day"][t_target]) as dsrc:
           
            y_true = dsrc.read(7, out_shape=(512, 512), resampling=rasterio.enums.Resampling.nearest)
            return self._normalize(y_true, is_label=True)