from dataset import TSDataset
import tensorflow as tf
import numpy as np
import rasterio
from rasterio.enums import Resampling

class InferenceTS(TSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_size = 512
        self.patch_size = 256

    def _read_and_rescale(self, dataset, bands, is_label=False):
        resampling_mode = Resampling.nearest if is_label else Resampling.bilinear
        band_indices = bands if bands is not None else list(range(1, dataset.count + 1))
        
        data = dataset.read(
            band_indices,
            out_shape=(len(band_indices), self.target_size, self.target_size),
            resampling=resampling_mode
        )
        return self._normalize(data, is_label=is_label)

    def __getitem__(self, idx):
        info = self.samples[idx]
        region = self.raw_paths[info["region"]]
        dates = info["dates"] 

        full_seq_512 = [] 
        for t_date in dates[:-1]:
            # VIIRS Day 
            with rasterio.open(region["VIIRS_Day"][t_date]) as dsrc:
                day = self._read_and_rescale(dsrc, [1, 2, 3, 4, 5, 6])
                fire_today = self._read_and_rescale(dsrc, [7], is_label=True)

            # VIIRS Night
            if t_date in region["VIIRS_Night"]:
                with rasterio.open(region["VIIRS_Night"][t_date]) as nsrc:
                    num_bands = nsrc.count
                    night = self._read_and_rescale(nsrc, [num_bands-1, num_bands])
            else:
                night = np.zeros((2, self.target_size, self.target_size), dtype=np.float32)

            # FirePred
            with rasterio.open(region["FirePred"][t_date]) as fsrc:
                firep = self._read_and_rescale(fsrc, None)

            combined_t = np.concatenate([day, fire_today, night, firep], axis=0)
            full_seq_512.append(combined_t)

        full_seq_tensor = np.stack(full_seq_512)

        patches = []
        offsets = [(0, 0), (0, 256), (256, 0), (256, 256)]
        for (y, x) in offsets:
            patch = full_seq_tensor[:, :, y:y+256, x:x+256]
            patches.append(patch)

        return {
            "patches": np.stack(patches), # [4, T, C, 256, 256]
            "sample_id": info['sample_id']
        }

    def predict_full_image(self, model, patches_tensor):
        """
        Predice los 4 parches y los une en una imagen de 512x512.
        """
        # patches_tensor viene como [4, T, C, 256, 256]
        patches_tf = np.transpose(patches_tensor, (0, 1, 3, 4, 2))
        
        preds = []
        for i in range(4):
            patch = patches_tf[i:i+1] # (1, T, 256, 256, C)
            p = model.predict(patch, verbose=0)
        
            p_prob = tf.nn.sigmoid(p).numpy()
            preds.append(np.squeeze(p_prob)) # (256, 256)

        # Reconstrucción de la cuadrícula 2x2
        top = np.concatenate([preds[0], preds[1]], axis=1)
        bottom = np.concatenate([preds[2], preds[3]], axis=1)
        return np.concatenate([top, bottom], axis=0) # [512, 512]

    def get_ground_truth(self, idx):
        info = self.samples[idx]
        region = self.raw_paths[info["region"]]
        target_date = info["dates"][-1]
        
        with rasterio.open(region["VIIRS_Day"][target_date]) as dsrc:
            y_true = dsrc.read(
                7, 
                out_shape=(self.target_size, self.target_size), 
                resampling=Resampling.nearest
            )
            return self._normalize(y_true, is_label=True)