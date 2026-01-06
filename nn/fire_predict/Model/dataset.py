from torch.utils.data import Dataset
import numpy as np
import rasterio
from pathlib import Path
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from rasterio import Affine
from rasterio.warp import reproject, Resampling
from constants import FIRE_THRESHOLDS, MAX_INPUT_SEQ_LEN, PRED_SEQ_LEN


class TSDataset(Dataset):
    def __init__(self, path_valid, shapes):
        self.total = 0
        self.image_paths = self._find_tiff_files(path_valid)
        self.shapes = shapes

    def _find_tiff_files(self, paths):
        r = {}
        idx = 1
        for i in paths:
            self.total += 1
            r[idx] = {}
            carpetas = os.listdir(i)
            d = []; n = []; l = []; f = []
            for j in carpetas:
                path = Path(i, j)
                for p in path.rglob("*.tif"):
                    if j == 'ESRI_LULC':
                        l.append(p)
                    elif j == 'FirePred':
                        f.append(p)
                    elif j == 'VIIRS_Day':
                        d.append(p)
                    else:
                        n.append(p)
            # Ordenar por nombre de archivo (contiene fecha)
            r[idx]['ESRI_LULC'] = sorted(l, key=lambda x: x.name)
            r[idx]['FirePred'] = sorted(f, key=lambda x: x.name)
            r[idx]['VIIRS_Day'] = sorted(d, key=lambda x: x.name)
            r[idx]['VIIRS_Night'] = sorted(n, key=lambda x: x.name)
            idx += 1
        return r

    def __len__(self):
        return self.total

    def open_tif(self, paths):
        imgs, transforms, crss = [], [], []
        for path in paths:
            with rasterio.open(path) as src:
                img = src.read()  # (C, H, W)
                imgs.append(img)
                transforms.append(src.transform)
                crss.append(src.crs)
        return imgs, transforms, crss

    def _resize_rasterio(self, img, src_transform, src_crs, target_shape=(256, 256), resampling=Resampling.bilinear):
        """
        img: ndarray (C, H, W)
        src_transform: Affine transform from original raster
        src_crs: CRS (e.g., 'EPSG:4326')
        target_shape: (H_new, W_new) — e.g., (256, 256)
        resampling: rasterio.warp.Resampling.*
        Returns: (new_img, new_transform)
        """
        H, W = img.shape[-2], img.shape[-1]
        H_new, W_new = target_shape
    
        scale_x = W / W_new
        scale_y = H / H_new
        new_transform = src_transform * src_transform.scale(scale_x, scale_y)
    
        # Allocate output
        dst_shape = (img.shape[0], H_new, W_new)
        dst_img = np.empty(dst_shape, dtype=img.dtype)
    
        reproject(
            source=img,
            destination=dst_img,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=new_transform,
            dst_crs=src_crs,  # keep same CRS
            resampling=resampling
        )
        return dst_img, new_transform

    
    def _normalize_data(self, data, is_label=False):
        """
        Normaliza los datos a rangos seguros [0, 1].
        Maneja NaNs, valores negativos de error y outliers.
        """
        # 1. Limpieza básica de NaNs
        data = np.nan_to_num(data, nan=0.0)
        
        if is_label:
            # Etiquetas: [0, 1] estricto
            # Eliminar valores negativos (NoData)
            data[data < 0] = 0.0
            # Clip a 1.0 
            data = np.clip(data, 0.0, 1.0)
        else:
            # Inputs: Limpiar basura negativa (ej. -9999)
            data[data < -100] = 0.0
            
            # Normalización Min-Max por canal para estabilizar entrenamiento
            # Si el array es 3D (C, H, W)
            if data.ndim == 3:
                for c in range(data.shape[0]):
                    channel = data[c]
                    min_val, max_val = channel.min(), channel.max()
                    # Solo normalizar si hay rango dinámico
                    if max_val - min_val > 1e-5:
                        data[c] = (channel - min_val) / (max_val - min_val)
                    else:
                        # Si es constante y grande, bajarlo a 1.0
                        if max_val > 1.0:
                            data[c] = 1.0
            # Si es 2D (H, W)
            elif data.ndim == 2:
                min_val, max_val = data.min(), data.max()
                if max_val - min_val > 1e-5:
                    data = (data - min_val) / (max_val - min_val)
                elif max_val > 1.0:
                    data[:] = 1.0
                    
        return data

    def _preprocess(self, imgs, transforms, crss, is_day=False, is_night=False, is_fireP=False, target_shape=(256, 256)):
        news_imgs = []
        labels = []
        pixel_areas = []
    
        for img, src_transform, src_crs in zip(imgs, transforms, crss):
            if img.ndim == 2:
                img = img[None, :, :]
            
            if is_day:
                # Separar datos y label
                img_data = img[:-2, :, :]  # (C-2, H, W) - datos
                
                raw_fire = img[-2, :, :]   # (H, W)
                
                # Crear máscara de validez basada en NaNs originales
                raw_mask = (~np.isnan(raw_fire)).astype(np.float32)
                
                # Normalizar Label (raw_fire) ANTES de resize para que el average funcione bien sobre 0-1
                raw_fire = self._normalize_data(raw_fire, is_label=True)
                
                # Procesar DATOS con bilinear
                img_resized, new_transform = self._resize_rasterio(
                    img_data, src_transform, src_crs, 
                    target_shape=target_shape, resampling=Resampling.bilinear
                )  # (C-2, 256, 256)

                # Calcular área del píxel nuevo
                area = abs(new_transform.a * new_transform.e)
                pixel_areas.append(area)
                
                # Procesar LABEL (Fire) con AVERAGE para soft labels
                fire_resized, _ = self._resize_rasterio(
                    raw_fire[None, :, :], src_transform, src_crs,
                    target_shape=target_shape, resampling=Resampling.average
                )
                
                # Procesar MASK con AVERAGE para consistencia y luego binarizar
                mask_resized, _ = self._resize_rasterio(
                    raw_mask[None, :, :], src_transform, src_crs,
                    target_shape=target_shape, resampling=Resampling.average
                )
                # válido si hay al menos un poco de información válida 
                
                mask_resized = (mask_resized > 0.0001).astype(np.float32)
                
                # Asignar
                fire_label = fire_resized[0] # Ya es soft label (0.0 - 1.0)
                label_mask = mask_resized[0]
                
                # Normalizar (Inputs)
                img_resized = self._normalize_data(img_resized, is_label=False)
                
                news_imgs.append(img_resized)
                labels.append((fire_label, label_mask))
                
            else:
                # Para otros casos, decidir resampling según tipo
                if is_fireP:
                    resampling = Resampling.nearest
                elif is_night:
                    resampling = Resampling.bilinear
                else:  # LULC: nearest 
                    resampling = Resampling.nearest
                
                # Resize con georeferencia
                img_resized, _ = self._resize_rasterio(
                    img, src_transform, src_crs, 
                    target_shape=target_shape, resampling=resampling
                )  # (C, 256, 256)
                
                # Normalizar DATOS (Inputs)
                # Si es fireP, quizás queramos tratarlo diferente, pero por ahora normalizamos todo
                img_resized = self._normalize_data(img_resized, is_label=False)
                
                # Para night, si tiene 5 canales, tomar solo los últimos 2
                if is_night and img_resized.shape[0] == 5:
                    img_resized = img_resized[-2:, :, :]
                
                news_imgs.append(img_resized)
    
        if is_day:
            label_imgs, label_masks = zip(*labels) if labels else ([], [])
            return news_imgs, list(label_imgs), list(label_masks), pixel_areas
        else:
            return news_imgs

    def __getitem__(self, idx):
        paths = self.image_paths[idx+1]
        
        needed_len = MAX_INPUT_SEQ_LEN + PRED_SEQ_LEN
        
        # Helper simple para coger los últimos N elementos
        def get_last_n(l, n):
            return l[-n:] if len(l) > n else l

        # Seleccionar solo los archivos necesarios (últimos N)
        day_paths = get_last_n(paths['VIIRS_Day'], needed_len)
        night_paths = get_last_n(paths['VIIRS_Night'], needed_len)
        fire_paths = get_last_n(paths['FirePred'], needed_len)
        static_paths = paths['ESRI_LULC']

        seq_day, t_day, crs_day = self.open_tif(day_paths)
        seq_night, t_night, crs_night = self.open_tif(night_paths)
        seq_firepred, t_fire, crs_fire = self.open_tif(fire_paths)
        static_img, t_static, crs_static = self.open_tif(static_paths)

        # Preprocesar con máscaras
        day_imgs, labels, label_masks, pixel_areas = self._preprocess(seq_day, t_day, crs_day, 
                                                                   is_day=True, target_shape=self.shapes)
        night_imgs = self._preprocess(seq_night, t_night, crs_night, 
                                                  is_night=True, target_shape=self.shapes)
        fire_imgs = self._preprocess(seq_firepred, t_fire, crs_fire, 
                                                is_fireP=True, target_shape=self.shapes)
        lulc_imgs = self._preprocess(static_img, t_static, crs_static, 
                                                target_shape=self.shapes)

        # Convertir a numpy
        day_imgs = np.array(day_imgs).astype(np.float32)      # (T, C_day, H, W)
        night_imgs = np.array(night_imgs).astype(np.float32)  # (T, C_night, H, W)
        fire_imgs = np.array(fire_imgs).astype(np.float32)    # (T, C_fire, H, W)
        lulc_img = np.array(lulc_imgs[0]).astype(np.float32)  # (C_lulc, H, W)
        
        labels = np.array(labels) # (T, H, W)
        # Discretizar labels en clases (Cualquier valor > 0 es incendio)
        labels = (labels > 0).astype(np.int64)
        
        label_masks = np.array(label_masks).astype(np.float32)
        pixel_areas = np.array(pixel_areas).astype(np.float32) # (T,)

        # Obtener T (usando day como referencia, asumiendo todos tienen mismo T)
        T = day_imgs.shape[0]
        H, W = day_imgs.shape[-2], day_imgs.shape[-1]
        
        # Ajustar night y fire al mismo T que day (truncar o padear)
        if night_imgs.shape[0] != T:
            if night_imgs.shape[0] > T:
                night_imgs = night_imgs[:T]
            else:
                pad_t = T - night_imgs.shape[0]
                night_imgs = np.pad(night_imgs, ((0, pad_t), (0, 0), (0, 0), (0, 0)), mode='constant')
        
        if fire_imgs.shape[0] != T:
            if fire_imgs.shape[0] > T:
                fire_imgs = fire_imgs[:T]
            else:
                pad_t = T - fire_imgs.shape[0]
                fire_imgs = np.pad(fire_imgs, ((0, pad_t), (0, 0), (0, 0), (0, 0)), mode='constant')

        # Expandir LULC (estático) a T timesteps: (C_lulc, H, W) → (T, C_lulc, H, W)
        lulc_expanded = np.tile(lulc_img[None, :, :, :], (T, 1, 1, 1))

        # Combinar todos los canales en una sola imagen
        # combined: (T, C_day + C_night + C_fire + C_lulc, H, W)
        combined = np.concatenate([day_imgs, night_imgs, fire_imgs, lulc_expanded], axis=1)

        # To tensor
        return (
            torch.from_numpy(combined),      # (T, C_total, H, W)
            torch.from_numpy(labels),        # (T, H, W)
            torch.from_numpy(label_masks),   # (T, H, W)
            torch.from_numpy(pixel_areas),   # (T,)
        )


def collate_fn(batch):
    """
    Collate function simplificada para datos combinados.
    
    Input por muestra: (combined, labels, label_masks, pixel_areas)
    Output: dict con tensores paddeados
    """
    combined_list, labels_list, label_masks_list, pixel_areas_list = zip(*batch)

    # Padding temporal: (B, T_max, C, H, W)
    combined_padded = pad_sequence(combined_list, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=0.0)
    label_masks_padded = pad_sequence(label_masks_list, batch_first=True, padding_value=0.0)
    pixel_areas_padded = pad_sequence(pixel_areas_list, batch_first=True, padding_value=0.0)

    return {
        'x': combined_padded,              # (B, T, C_total, H, W)
        'label': labels_padded,            # (B, T, H, W)
        'label_mask': label_masks_padded,  # (B, T, H, W)
        'pixel_area': pixel_areas_padded,  # (B, T)
    }