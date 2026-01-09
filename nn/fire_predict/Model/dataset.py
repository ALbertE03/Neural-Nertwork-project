import os
import torch
import numpy as np
import rasterio
import json
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from rasterio.warp import reproject, Resampling

class TSDatasetSequences(Dataset):
    def __init__(
        self,
        path_valid,
        cache_dir,
        train=False,
        augment=False,
        crop_size=128,
        seq_len=6,
        stride_t=6,
        TARGET_SIZE=512,
        force_rebuild=False
    ):
        self.train = train
        self.augment = augment
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.stride_t = stride_t
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.TARGET_SIZE = TARGET_SIZE
        
        self.catalog_path = self.cache_dir / f"catalog_s{seq_len}_st{stride_t}_c{crop_size}_overlap.json"
        self.raw_paths = self._find_tiff_files(path_valid)

        if self.catalog_path.exists() and not force_rebuild:
            print(f"Cargando catálogo desde {self.catalog_path}...")
            with open(self.catalog_path, 'r') as f:
                self.samples = json.load(f)
        else:
            print("Generando catálogo con 4 bloques solapados...")
            self.samples = self._create_sequence_catalog()
            with open(self.catalog_path, 'w') as f:
                json.dump(self.samples, f)
        
        print(f"Dataset listo: {len(self.samples)} parches (4 por cada bloque 512).")

    def _find_tiff_files(self, paths):
        r = {}
        for idx, base in enumerate(paths):
            r[idx] = {k: [] for k in ["ESRI_LULC", "FirePred", "VIIRS_Day", "VIIRS_Night"]}
            for folder in os.listdir(base):
                p = Path(base, folder)
                if p.is_dir() and folder in r[idx]:
                    r[idx][folder] = sorted(list(p.rglob("*.tif")), key=lambda x: x.name)
        return r

    def _create_sequence_catalog(self):
        catalog = []
   
        p1 = 0
        p2 = 512 - 128 

        p2_overlap = 350 

        for region_idx, folders in tqdm(self.raw_paths.items()):
            num_days = min(len(folders["VIIRS_Day"]), len(folders["VIIRS_Night"]), len(folders["FirePred"]))

            for t_start in range(0, num_days - self.seq_len + 1, self.stride_t):
                block_id = f"reg{region_idx}_t{t_start}_full512"              
                quadrants = [
                    (p1, p1),                 # Top-Left
                    (p1, p2_overlap),         # Top-Right
                    (p2_overlap, p1),         # Bottom-Left
                    (p2_overlap, p2_overlap)  # Bottom-Right
                ]

                for i, (y0, x0) in enumerate(quadrants):
                    catalog.append({
                        'region': region_idx,
                        't_start': t_start,
                        'y_offset': y0,
                        'x_offset': x0,
                        'block_id': block_id,
                        'sample_id': f"reg{region_idx}_t{t_start}_q{i}"
                    })
        return catalog

    def _read_reproject_to_ref(self, src, ref_src, window, resampling, bands=None):
        count = len(bands) if bands is not None else src.count
        out = np.zeros((count, self.TARGET_SIZE, self.TARGET_SIZE), dtype=np.float32)
        
        dst_transform = rasterio.transform.from_bounds(
            *rasterio.windows.bounds(window, ref_src.transform),
            width=self.TARGET_SIZE,
            height=self.TARGET_SIZE
        )

        source_bands = rasterio.band(src, bands if bands is not None else list(range(1, src.count + 1)))
        
        reproject(
            source=source_bands,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=ref_src.crs,
            resampling=resampling
        )
        return out

    def _load_and_process_full_block(self, info):
        region = self.raw_paths[info['region']]
        t_range = slice(info['t_start'], info['t_start'] + self.seq_len)
        
        seq_x, seq_y = [], []
        for dp, np_, fp in zip(region["VIIRS_Day"][t_range], 
                                region["VIIRS_Night"][t_range], 
                                region["FirePred"][t_range]):
            
            with rasterio.open(dp) as dsrc:
                # Proyectamos la imagen original COMPLETA a 512x512
                full_win = rasterio.windows.Window(0, 0, dsrc.width, dsrc.height)
                
                day = self._normalize(self._read_reproject_to_ref(dsrc, dsrc, full_win, Resampling.bilinear, bands=list(range(1,7))))
                fire = self._normalize(self._read_reproject_to_ref(dsrc, dsrc, full_win, Resampling.nearest, bands=[7]), is_label=True)
                
                with rasterio.open(np_) as nsrc:
                    night = self._normalize(self._read_reproject_to_ref(nsrc, dsrc, full_win, Resampling.bilinear, bands=[1,2]))
                
                with rasterio.open(fp) as fsrc:
                    firep = self._normalize(self._read_reproject_to_ref(fsrc, dsrc, full_win, Resampling.bilinear))

                x_combined = np.concatenate([day, fire, night, firep], axis=0)
                seq_x.append(x_combined)
                seq_y.append(fire.squeeze(0))

        return np.stack(seq_x), np.stack(seq_y)

    def _normalize(self, x, is_label=False):
        x = np.nan_to_num(x, nan=0.0)
        if is_label:
            return np.clip(x, 0.0, 1.0)
        # Normalización por canal
        for c in range(x.shape[0]):
            m = x[c].max() - x[c].min()
            if m > 1e-6:
                x[c] = (x[c] - x[c].min()) / m
        return x

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        cache_f = self.cache_dir / f"{info['sample_id']}.npz"
    
        # Si el parche específico no existe, generamos los 4 de esa secuencia
        if not cache_f.exists():
            self._process_and_save_all_quadrants(info)
    
        # Cargar solo el parche pequeño de 128x128
        data = np.load(cache_f)
        return torch.from_numpy(data["x"]).float(), torch.from_numpy(data["y"]).float()

    def _process_and_save_all_quadrants(self, info):
        # 1. Cargamos el bloque de 512 completo (reproyección lenta)
        x_block, y_block = self._load_and_process_full_block(info)
        
        # 2. Definimos otra vez las coordenadas (deben coincidir con el catálogo)
        p1, p2_overlap = 0, 350
        quadrants = [(p1, p1), (p1, p2_overlap), (p2_overlap, p1), (p2_overlap, p2_overlap)]
        
        # 3. Cortamos y guardamos cada uno por separado
        for i, (y0, x0) in enumerate(quadrants):
            x_patch = x_block[:, :, y0:y0+128, x0:x0+128]
            y_patch = y_block[:, y0:y0+128, x0:x0+128]
            
            # Guardamos con el ID que el catálogo espera
            patch_id = f"reg{info['region']}_t{info['t_start']}_q{i}"
            path = self.cache_dir / f"{patch_id}.npz"
            np.savez_compressed(path, x=x_patch, y=y_patch)