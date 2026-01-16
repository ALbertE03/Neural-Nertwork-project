import os
import json
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
class TSDatasetFlat(Dataset):
    def __init__(
        self,
        path_valid,
        cache_dir,
        crop_size=256,
        seq_len=3,
        stride_t=1,
        force_rebuild=False,
    ):
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.stride_t = stride_t
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.catalog_path = self.cache_dir / f"catalog_flat_s{seq_len}_st{stride_t}_c{crop_size}.json"
        
        self.raw_paths = self._find_tiff_files(path_valid)

        if self.catalog_path.exists() and not force_rebuild:
            print(f"Cargando catálogo desde {self.catalog_path}...")
            with open(self.catalog_path, "r") as f:
                self.samples = json.load(f)
        else:
            print("Generando catálogo")
            self.samples = self._create_flat_catalog()
            with open(self.catalog_path, "w") as f:
                json.dump(self.samples, f)

        print(f"Dataset listo: {len(self.samples)} muestras")

    def _find_tiff_files(self, paths):
        r = {}
        for idx, base in enumerate(paths):
            r[idx] = {k: [] for k in ["FirePred", "VIIRS_Day", "VIIRS_Night"]}
            for folder in os.listdir(base):
                p = Path(base, folder)
                if p.is_dir() and folder in r[idx]:
                    r[idx][folder] = sorted(list(p.rglob("*.tif")), key=lambda x: x.name)
        return r

    def _create_flat_catalog(self):
        catalog = []
        for region_idx, folders in tqdm(self.raw_paths.items()):
            num_days = min(
                len(folders["VIIRS_Day"]),
                len(folders["VIIRS_Night"]),
                len(folders["FirePred"]),
            )
            
         
            for t_start in range(0, num_days - self.seq_len, self.stride_t):
                sample_id = f"reg{region_idx}_t{t_start}"
                catalog.append({
                    "region": region_idx,
                    "t_start": t_start,
                    "sample_id": sample_id
                })
        return catalog

    def _normalize(self, x, is_label=False):
        x = np.nan_to_num(x, nan=0.0)
        if is_label:
            return np.clip(x, 0.0, 1.0)
        for c in range(x.shape[0]):
            vmin, vmax = x[c].min(), x[c].max()
            if vmax - vmin > 1e-6:
                x[c] = (x[c] - vmin) / (vmax - vmin)
        return x

    def _load_single_window(self, info):
        region = self.raw_paths[info["region"]]
        t_start = info["t_start"]
        
        seq_x = []
        # Solo cargamos la ventana de seq_len días
        for t in range(t_start, t_start + self.seq_len):
            #  VIIRS Day
            with rasterio.open(region["VIIRS_Day"][t]) as dsrc:
                y0 = (dsrc.height - self.crop_size) // 2
                x0 = (dsrc.width - self.crop_size) // 2
                win = rasterio.windows.Window(x0, y0, self.crop_size, self.crop_size)
                day = self._normalize(dsrc.read([1,2,3,4,5,6], window=win))
                fire_today = self._normalize(dsrc.read([7], window=win), is_label=True)

            #  VIIRS Night
            with rasterio.open(region["VIIRS_Night"][t]) as nsrc:
                night = self._normalize(nsrc.read( window=win)[-2:])

            #  FirePred
            with rasterio.open(region["FirePred"][t]) as fsrc:
                firep = self._normalize(fsrc.read(window=win))

            x_combined = np.concatenate([day, fire_today, night, firep], axis=0)
            seq_x.append(x_combined)

        # Target: El canal 7 del día siguiente (t_start + seq_len)
        with rasterio.open(region["VIIRS_Day"][t_start + self.seq_len]) as dsrc:
            y_target = self._normalize(dsrc.read([7], window=win), is_label=True)

        return np.stack(seq_x), y_target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        cache_f = self.cache_dir / f"{info['sample_id']}.npz"

        if not cache_f.exists():
            x, y = self._load_single_window(info)
            np.savez_compressed(cache_f, x=x, y=y)
        else:
            try:
                data = np.load(cache_f)
                x, y = data["x"], data["y"]
            except Exception as e:
                print(f"Archivo corrupto detectado: {cache_f}")  
                return {}
        return {
            "x": torch.from_numpy(x).float(), # [T, C, H, W]
            "y": torch.from_numpy(y).float(), # [1, H, W] (Día siguiente)
            "id": info['sample_id']
        }


class TSDatasetFlatTF(TSDatasetFlat): 
    def __getitem__(self, idx):
        info = self.samples[idx]
        cache_f = self.cache_dir / f"{info['sample_id']}.npz"

        try:
            if not cache_f.exists():
                x, y = self._load_single_window(info)
                np.savez_compressed(cache_f, x=x, y=y)
            else:
                data = np.load(cache_f)
                x, y = data["x"], data["y"]
            
            # PyTorch: [T, C, H, W] -> TF: [T, H, W, C]
            x = np.transpose(x, (0, 2, 3, 1)) 
            # PyTorch: [1, H, W] -> TF: [H, W, 1]
            y = np.transpose(y, (1, 2, 0))
            
            return x.astype(np.float32), y.astype(np.float32)

        except Exception as e:
            # Si un archivo está corrupto, lo borramos y cargamos el siguiente
            print(f"Error en {cache_f}, reintentando...")
            if cache_f.exists(): cache_f.unlink()
            return self.__getitem__((idx + 1) % len(self.samples))
        



def prepare_tf_dataset(py_dataset, batch_size=4, is_train=True):
    def generator():
        for i in range(len(py_dataset)):
            yield py_dataset[i]

    # x: (T, H, W, C), y: (H, W, 1)
    output_signature = (
        tf.TensorSpec(shape=(3, 256, 256, 28), dtype=tf.float32), 
        tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    if is_train:
        ds = ds.shuffle(buffer_size=100)
    
    ds = ds.batch(batch_size)
    
    ds = ds.prefetch(tf.data.AUTOTUNE).repeat()
    
    return ds

