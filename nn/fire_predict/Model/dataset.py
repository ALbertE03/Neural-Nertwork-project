import torch
from torch.utils.data import Dataset
import numpy as np

class FireDataset(Dataset):
    def __init__(self, config, transform=None):
        """
        config: Config object containing path and seq_len info
        """
        self.x_data = np.load(config.data_x_path).astype(np.float32)
        self.y_data = np.load(config.data_y_path).astype(np.float32)
        
        self.input_seq_len = config.input_seq_len
        self.pred_seq_len = config.pred_seq_len
        self.transform = transform
        
        # --- Sliding Window Pre-calculation ---
        # We need to map global dataset indices to specific (sample_idx, time_start_idx)
        # Assuming data is (Samples, Time, ...)
        # A valid sequence must have length input_seq_len + pred_seq_len
        
        self.samples = self.x_data.shape[0]
        self.time_steps = self.x_data.shape[1]
        
        self.valid_starts = []
        
        # Calculate valid start indices for every sample video
        possible_starts_per_sample = self.time_steps - (self.input_seq_len + self.pred_seq_len) + 1
        
        if possible_starts_per_sample <= 0:
            raise ValueError(f"Time dimension ({self.time_steps}) is too short for seq_len {input_seq_len} + {pred_seq_len}")
            
        for s_idx in range(self.samples):
            for t_idx in range(possible_starts_per_sample):
                self.valid_starts.append((s_idx, t_idx))
                
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        sample_idx, t_start = self.valid_starts[idx]
        
        # Input Window
        t_mid = t_start + self.input_seq_len
        # Target Window
        t_end = t_mid + self.pred_seq_len
        
        # Extract slices
        x_seq = self.x_data[sample_idx, t_start:t_mid] # (T_in, C, H, W)
        y_seq = self.y_data[sample_idx, t_mid:t_end]   # (T_out, 1, H, W)
        
        # Convert to Tensor
        x_tensor = torch.from_numpy(x_seq)
        y_tensor = torch.from_numpy(y_seq)
        
        # Apply normalization/transforms if needed
        if self.transform:
            x_tensor = self.transform(x_tensor)
            
        return x_tensor, y_tensor

# Example handling for MinMax Normalization (Optional utility)
def min_max_normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
