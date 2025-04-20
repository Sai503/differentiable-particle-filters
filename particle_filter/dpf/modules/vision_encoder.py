import torch.nn as nn
import torch.nn.functional as F
import torch

class VisionEncoder(nn.Module):  # Renamed from Encoder to VisionEncoder
    def __init__(self, dropout_keep_prob):
        super(VisionEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=1 - dropout_keep_prob)
        self.linear = nn.Linear(1024, 128)

    def forward(self, o):
        if o.dim() == 4 and o.size(-1) == 3:
            o = o.permute(0, 3, 1, 2).contiguous()
        if o.dtype != torch.float32:
            o = o.float()
        x = self.conv(o)
        x = self.flatten(x)
        x = self.dropout(x)
        # print(x.shape)
        return F.relu(self.linear(x))
