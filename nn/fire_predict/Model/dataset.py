from torch.utils.data import Dataset
import numpy as np
import rasterio
from pathlib import Path
import os
from scipy.ndimage import zoom
import torch
from torch.nn.utils.rnn import pad_sequence
from rasterio import Affine
from rasterio.warp import reproject, Resampling

class TSDataset(Dataset):
    def __init__(self, path_valid,shapes):
        self.total = 0
        self.image_paths = self._find_tiff_files(path_valid)
        self.shapes=shapes

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


    def _resize_rasterio(self,img, src_transform, src_crs, target_shape=(256, 256), resampling=Resampling.bilinear):
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

    def _preprocess(self, imgs, transforms, crss, is_day=False, is_night=False, is_fireP=False, target_shape=(256, 256)):
        H_new, W_new = target_shape
        news_imgs = []
        labels = []
        masks = []
    
        for img, src_transform, src_crs in zip(imgs, transforms, crss):
            if img.ndim == 2:
                img = img[None, :, :]
            C, h, w = img.shape
    
            # Resize with georeferencing
            resampling = Resampling.nearest if is_fireP or not is_day else Resampling.bilinear
            img_resized, _ = self._resize_rasterio(
                img, src_transform, src_crs, target_shape=target_shape, resampling=resampling
            )  # (C, 256, 256)
    
            # Mask 
            mask = (~np.isnan(img_resized)).astype(np.float32)
    
            if is_day:
                raw_label = img_resized[-2, :, :]  # (256, 256)

     
                fire_mask = raw_label > 1e-5  


                label = fire_mask.astype(np.float32)  

                label_mask = (~np.isnan(raw_label)).astype(np.float32)  # (256, 256)


                img_data = img_resized[:-2, :, :]  # (C-2, 256, 256)
                data_mask = mask[:-2, :, :]        # (C-2, 256, 256)

                img_data = np.nan_to_num(img_data, nan=0.0)

                # Guardar
                news_imgs.append(img_data)
                masks.append(data_mask)
                labels.append((label, label_mask)) 
    
            elif is_night:
                if img_resized.shape[0] == 5:
                    img_resized = img_resized[-2:, :, :]
                    mask = mask[-2:, :, :]
                img_resized = np.nan_to_num(img_resized, nan=0.0)
                news_imgs.append(img_resized)
                masks.append(mask)
    
            elif is_fireP:
                # Para FirePred, usar nearest: evitar suavizar umbrales de temperatura/actividad
                img_resized = np.nan_to_num(img_resized, nan=200.0)
                news_imgs.append(img_resized)
                masks.append(mask)
    
            else:  # Static LULC: ¡nearest es obligatorio!
                img_resized = np.nan_to_num(img_resized, nan=0.0)
                news_imgs.append(img_resized)
                masks.append(mask)
    
        if is_day:
            label_imgs, label_masks = zip(*labels) if labels else ([], [])
            return news_imgs, masks, list(label_imgs), list(label_masks)
        else:
            return news_imgs, masks

    def __getitem__(self, idx):
        paths = self.image_paths[idx]
        seq_day, t_day, crs_day = self.open_tif(paths['VIIRS_Day'])
        seq_night, t_night, crs_night = self.open_tif(paths['VIIRS_Night'])
        seq_firepred, t_fire, crs_fire = self.open_tif(paths['FirePred'])
        static_img, t_static, crs_static = self.open_tif(paths['ESRI_LULC'])

        # Preprocesar con máscaras
        day_imgs, day_masks, labels, label_masks = self._preprocess(seq_day,t_day,crs_day, is_day=True,target_shape=self.shapes)
        night_imgs, night_masks = self._preprocess(seq_night,t_night,crs_night, is_night=True,target_shape=self.shapes)
        fire_imgs, fire_masks = self._preprocess(seq_firepred,t_fire,crs_fire, is_fireP=True,target_shape=self.shapes)
        lulc_imgs, lulc_masks = self._preprocess(static_img,t_static,crs_static,target_shape=self.shapes)

        # Convertir a numpy
        day_imgs = np.array(day_imgs).astype(np.float32)      # (T, C_day, H, W)
        night_imgs = np.array(night_imgs).astype(np.float32)  # (T, C_night, H, W)
        fire_imgs = np.array(fire_imgs).astype(np.float32)    # (T, C_fire, H, W)
        lulc_img = np.array(lulc_imgs[0]).astype(np.float32)  # (C_lulc, H, W)
        labels = np.array(labels).astype(np.float32)          # (T, H, W)
        label_masks = np.array(label_masks).astype(np.float32)

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
            torch.from_numpy(label_masks)    # (T, H, W)
        )


def collate_fn(batch):
    """
    Collate function simplificada para datos combinados.
    
    Input por muestra: (combined, labels, label_masks)
    Output: dict con tensores paddeados
    """
    combined_list, labels_list, label_masks_list = zip(*batch)

    # Padding temporal: (B, T_max, C, H, W)
    combined_padded = pad_sequence(combined_list, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=0.0)
    label_masks_padded = pad_sequence(label_masks_list, batch_first=True, padding_value=0.0)

    return {
        'x': combined_padded,              # (B, T, C_total, H, W)
        'label': labels_padded,            # (B, T, H, W)
        'label_mask': label_masks_padded,  # (B, T, H, W)
    }