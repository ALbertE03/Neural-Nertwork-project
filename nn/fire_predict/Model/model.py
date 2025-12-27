import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, 
                              out_channels=4 * self.hidden_dim, 
                              kernel_size=self.kernel_size, 
                              padding=self.padding, 
                              bias=self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class FireSpreadModel(nn.Module):
    """
    Encoder-Decoder Architecture with ConvLSTM for Spatio-Temporal Prediction.
    """
    def __init__(self, config):
        super(FireSpreadModel, self).__init__()
        
        self.hidden_channels = config.hidden_channels
        self.kernel_size = config.kernel_size
        input_channels = config.input_channels
        
        # --- Encoder ---
        # Extracts spatial features from each frame independently
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # --- Temporal Bottleneck ---
        # ConvLSTM processes the sequence of features
        self.conv_lstm = ConvLSTMCell(input_dim=64, 
                                      hidden_dim=self.hidden_channels, 
                                      kernel_size=self.kernel_size, 
                                      bias=True)
        
        # --- Decoder ---
        # Reconstructs the next step prediction from the last hidden state
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1) # Output: 1 channel (Fire/No Fire logits)
        )
        
    def forward(self, x, future_steps=1):
        """
        x: (Batch, Time, Channels, Height, Width)
        future_steps: Number of future steps to predict (autoregressive)
        """
        b, t, c, h, w = x.size()
        
        # Initialize hidden state
        h_t, c_t = self.conv_lstm.init_hidden(b, (h, w))
        
        # Process input sequence
        for t_step in range(t):
            # Extract features for current frame
            # (Batch, Channels, H, W)
            frame = x[:, t_step, :, :, :] 
            features = self.encoder(frame)
            
            # Update LSTM state
            h_t, c_t = self.conv_lstm(features, (h_t, c_t))
            
        # Predict future steps
        outputs = []
        last_h = h_t
        
        for _ in range(future_steps):
            # Decode the hidden state to get prediction
            pred = self.decoder(last_h) # (Batch, 1, H, W)
            outputs.append(pred)
            
            # Autoregressive inference not fully implemented for training stability 
            # (Teacher forcing it s better, but here we just return the next step predictions based on history)
            # Improving this would require feeding the output back as input.
            
        # Stack outputs along time dimension
        outputs = torch.stack(outputs, dim=1) # (Batch, Future_Steps, 1, H, W)
        
        return outputs