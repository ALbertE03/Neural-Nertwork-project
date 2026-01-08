import os
import random
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
from pathlib import Path
from constants import MAX_INPUT_SEQ_LEN, PRED_SEQ_LEN
class TSDatasetROI(Dataset):
    def __init__(
        self,
        path_valid,
        cache_dir,
        train=False,
        augment=False,
        crop_size=192,
        max_rois=4,
        augment_factor=3  
    ):
        self.train = train
        self.augment = augment
        self.crop_size = crop_size
        self.max_rois = max_rois
        self.augment_factor = augment_factor if train else 1

        self.image_paths = self._find_tiff_files(path_valid)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _find_tiff_files(self, paths):
        r = {}
        idx = 0
        for base in paths:
            r[idx] = {k: [] for k in ["ESRI_LULC", "FirePred", "VIIRS_Day", "VIIRS_Night"]}
            for folder in os.listdir(base):
                p = Path(base, folder)
                if p.is_dir():
                    for tif in p.rglob("*.tif"):
                        if folder in r[idx]:
                            r[idx][folder].append(tif)
            for k in r[idx]:
                r[idx][k] = sorted(r[idx][k], key=lambda x: x.name)
            idx += 1
        return r

    def __len__(self):
        return len(self.image_paths) * self.augment_factor

    def _normalize(self, x, is_label=False):
        x = np.nan_to_num(x, nan=0.0)
        if is_label:
            x[x < 0] = 0.0
            return np.clip(x, 0.0, 1.0)
        
        x[x < -100] = 0.0
        for c in range(x.shape[0]):
            vmin, vmax = x[c].min(), x[c].max()
            if vmax - vmin > 1e-6:
                x[c] = (x[c] - vmin) / (vmax - vmin)
        return x



    def _apply_augmentation(self, x, y):
        k = random.randint(0, 3) # Rotaciones 90 deg
        if k > 0:
            x = np.rot90(x, k=k, axes=(-2, -1))
            y = np.rot90(y, k=k, axes=(-2, -1))

        f = random.choice([None, 'horizontal', 'vertical'])
        if f == 'horizontal':
            x = np.flip(x, axis=-1)
            y = np.flip(y, axis=-1)
        elif f == 'vertical':
            x = np.flip(x, axis=-2)
            y = np.flip(y, axis=-2)

        return x.copy(), y.copy()

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_paths)
        is_augmented_instance = idx >= len(self.image_paths)
        
        paths = self.image_paths[real_idx]
        needed = MAX_INPUT_SEQ_LEN + PRED_SEQ_LEN
        
        def last_n(l): return l[-needed:] if len(l) > needed else l
        
        day_p = last_n(paths["VIIRS_Day"])
        night_p = last_n(paths["VIIRS_Night"])
        firep_p = last_n(paths["FirePred"])
        lulc_p = paths["ESRI_LULC"][0]
        
        # Ahora centers siempre tiene longitud self.max_rois
        centers = self._find_rois(day_p)
        half = self.crop_size // 2
        
        with rasterio.open(day_p[0]) as ref_src:
            base_h, base_w = ref_src.height, ref_src.width

        def read_with_padding(src, cy, cx, is_fire_label=False):
            y1_s = max(0, cy - half)
            x1_s = max(0, cx - half)
            read_w = max(0, min(self.crop_size, src.width - x1_s))
            read_h = max(0, min(self.crop_size, src.height - y1_s))
            
            win_s = Window(x1_s, y1_s, read_w, read_h)
            data = src.read(window=win_s)
            
            pad_y = self.crop_size - read_h
            pad_x = self.crop_size - read_w
            
            if is_fire_label:
                if data.ndim == 3: data = data[-1] # Tomar última banda si es 3D
                pad_shape = [(0, pad_y), (0, pad_x)]
            else:
                pad_shape = [(0, 0), (0, pad_y), (0, pad_x)]
                
            if pad_y > 0 or pad_x > 0:
                data = np.pad(data, pad_shape, mode='constant', constant_values=0)
            return data

        xs_roi, ys_roi = [], []

        for roi_id, (cy, cx) in enumerate(centers):
            # Usamos el real_idx para el cache para que la aumentación sea al vuelo
            cache_f = self._cache_file(real_idx, roi_id)
            
            if cache_f.exists():
                try:
                    data = np.load(cache_f)
                    xs_roi.append(data["x"])
                    ys_roi.append(data["y"])
                    continue
                except: pass

            seq_x, seq_y = [], []
            for dp, np_, fp in zip(day_p, night_p, firep_p):
                with rasterio.open(dp) as dsrc, rasterio.open(np_) as nsrc, rasterio.open(fp) as fsrc:
                    day_full = read_with_padding(dsrc, cy, cx)
                    day = self._normalize(day_full[:-2])
                    fire = self._normalize(day_full[-1], is_label=True) # Canal count
                    night = self._normalize(read_with_padding(nsrc, cy, cx)[-2:])
                    firep = self._normalize(read_with_padding(fsrc, cy, cx))
                    
                    seq_x.append(np.concatenate([day, night, firep], axis=0))
                    seq_y.append(fire)

            # LULC
            with rasterio.open(lulc_p) as lsrc:
                scale_y, scale_x = lsrc.height / base_h, lsrc.width / base_w
                l_cy, l_cx = int(cy * scale_y), int(cx * scale_x)
                lulc_crop = read_with_padding(lsrc, l_cy, l_cx)
                for t in range(len(seq_x)):
                    seq_x[t] = np.concatenate([seq_x[t], lulc_crop], axis=0)

            # Padding Temporal
            t_len = len(seq_x)
            if t_len < needed:
                zeros_x = np.zeros((seq_x[0].shape[0], self.crop_size, self.crop_size))
                zeros_y = np.zeros((self.crop_size, self.crop_size))
                seq_x += [zeros_x] * (needed - t_len)
                seq_y += [zeros_y] * (needed - t_len)

            x_stack, y_stack = np.stack(seq_x), np.stack(seq_y)
            np.savez_compressed(cache_f, x=x_stack, y=y_stack)
            xs_roi.append(x_stack)
            ys_roi.append(y_stack)

        xs_final = np.stack(xs_roi) 
        ys_final = np.stack(ys_roi) 

        # Aplicar aumentación si es entrenamiento
        if self.augment and is_augmented_instance:
            xs_final, ys_final = self._apply_augmentation(xs_final, ys_final)

        return torch.from_numpy(xs_final).float(), torch.from_numpy(ys_final).float()

    def _cache_file(self, idx, roi_id):
        return self.cache_dir / f"sample_{idx:05d}_roi_{roi_id}.npz"

def collate_fn(batch):
    x, y = zip(*batch)
    
    # Limpieza de dimensiones (Batch, ROIs, T, C, H, W)
    x = [xx.squeeze(0) if xx.ndim == 6 else xx for xx in x]
    y = [yy.squeeze(0) if yy.ndim == 5 else yy for yy in y]

    # Aunque ahora siempre debería ser max_rois por el replace=True, 
    # mantenemos el padding por seguridad si cambias el código.
    max_rois = max([xx.shape[0] for xx in x])
    
    x_pad, y_pad = [], []
    for xx, yy in zip(x, y):
        pad_n = max_rois - xx.shape[0]
        if pad_n > 0:
            xx = torch.cat([xx, torch.zeros(pad_n, *xx.shape[1:])], dim=0)
            yy = torch.cat([yy, torch.zeros(pad_n, *yy.shape[1:])], dim=0)
        x_pad.append(xx)
        y_pad.append(yy)
            
    return {
        'x': torch.stack(x_pad),
        'label': torch.stack(y_pad)
    }