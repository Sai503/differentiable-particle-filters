import torch.nn as nn
import torch.nn.functional as F
import torch

class LidarEncoder(nn.Module):
    def __init__(self, input_channels=2, output_dim=128, dropout_keep_prob=1.0):
        """
        1D Convolutional Network for encoding lidar data.

        Args:
            input_channels (int): Number of input channels for the lidar data.
            output_dim (int): Dimension of the output encoding.
            dropout_keep_prob (float): Dropout keep probability (default: 1.0, no dropout).
        """
        super(LidarEncoder, self).__init__()
        self.input_channels = input_channels
        self.conv = nn.Sequential( 
            nn.Conv1d(self.input_channels, 32, kernel_size=11, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 256, kernel_size=5, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=1 - dropout_keep_prob)
        
        self.fc = nn.Sequential( #fixed the missing parenthesis
            nn.Linear(21504, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, output_dim)
        ) #fixed the missing parenthesis


       


    def forward(self, x):
        """
        Forward pass for the lidar encoder.

        Args:
            x (torch.Tensor): Input lidar data of shape [B, input_channels, L],
                              where L is the sequence length.

        Returns:
            torch.Tensor: Encoded lidar features of shape [B, output_dim].
        """
        
        if x.dim() == 3 and x.size(-1) == 2:
            x = x.permute(0, 2, 1).contiguous()
        elif x.dim() == 4 and x.size(-1) == 2:
            x = x.permute(0, 1, 3, 2).contiguous()
        if x.dtype != torch.float32:
            x = x.float()
            
        reshape_needed = False    
        if x.dim() == 4:
            # If x has 4 dimensions, we need to reshape it to 3D for Conv1d
            reshape_needed = True # comine first two dimensions (batch and seq_len)
            # print('reshaping x')
            # print('x shape:', x.shape)
            batch_size, seq_len, channels, length = x.size()
            x = x.view(batch_size * seq_len, channels, length) 
            # print('x reshaped shape:', x.shape)
            
        x = self.conv(x)
        #debugging
        # print('x conv shape:', x.shape)
        x = self.flatten(x)
        # print('x flatten shape:', x.shape)
        x = self.fc(x)
        # print('x fc shape:', x.shape)
        # if reshape_needed:
        #     # If we reshaped x, we need to return it to its original shape
        #     # Assuming we want to return to [batch_size, seq_len, output_dim]
        #     x = x.view(batch_size, seq_len, -1)
        #     print('x reshaped back shape:', x.shape)
        
        return x
