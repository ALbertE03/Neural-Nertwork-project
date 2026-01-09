import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from pathlib import Path
from constants import MAX_INPUT_SEQ_LEN, PRED_SEQ_LEN

class TSDatasetTest(Dataset):
    def __init__(self, path_valid, cache_dir, target_size=(596, 596)):
        self.target_size = target_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_paths = self._find_tiff_files(path_valid)

    def _find_tiff_files(self, paths):
        r = []
        for base in paths:
            d = {k: [] for k in ["FirePred", "VIIRS_Day", "VIIRS_Night"]}
            for folder in os.listdir(base):
                p = Path(base, folder)
                if p.is_dir():
                    for tif in p.rglob("*.tif"):
                        if folder in d:
                            d[folder].append(tif)
            for k in d:
                d[k] = sorted(d[k], key=lambda x: x.name)
            if not d[k]:
                continue
            r.append(d)
        return r

    def __len__(self):
        return len(self.image_paths)

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

    def _resize(self, img):
        # img: (C, H, W) or (H, W)
        import cv2
        if img.ndim == 2:
            img = img[None, ...]
        C, H, W = img.shape
        resized = np.zeros((C, self.target_size[0], self.target_size[1]), dtype=img.dtype)
        for c in range(C):
            
            if hasattr(self, "_fire_channel_idx"):
                fire_idx = self._fire_channel_idx
            else:
                fire_idx = None
            if fire_idx is not None and c == fire_idx:
                resized[c] = cv2.resize(img[c], self.target_size[::-1], interpolation=cv2.INTER_NEAREST)
            else:
                resized[c] = cv2.resize(img[c], self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
        return resized

    def __getitem__(self, idx):
        paths = self.image_paths[idx]
        needed = MAX_INPUT_SEQ_LEN + PRED_SEQ_LEN
        def last_n(l): return l[-needed:] if len(l) > needed else l
        day_p = last_n(paths["VIIRS_Day"])
        night_p = last_n(paths["VIIRS_Night"])
        firep_p = last_n(paths["FirePred"])
        seq_x = []
        for dp, np_, fp in zip(day_p, night_p, firep_p):
            with rasterio.open(dp) as dsrc, rasterio.open(np_) as nsrc, rasterio.open(fp) as fsrc:
                day_full = dsrc.read()
                day = self._normalize(day_full[:-2])
                fire = self._normalize(day_full[6], is_label=True)
                night = self._normalize(nsrc.read()[-2:])
                firep = self._normalize(fsrc.read())
                patch = np.concatenate([day, fire[None, ...], night, firep], axis=0)
                if not hasattr(self, "_fire_channel_idx"):
                    self._fire_channel_idx = day.shape[0]
                patch = self._resize(patch)
                seq_x.append(patch)
        x_stack = np.stack(seq_x) # (T, C, H, W)
        return torch.from_numpy(x_stack).float()

if __name__ == "__main__":
    path_valid = ['']
    cache_dir = "cache_test"
    dataset = TSDatasetTest(path_valid, cache_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(loader):
        print(f"Batch {i} shape: {batch.shape}")
