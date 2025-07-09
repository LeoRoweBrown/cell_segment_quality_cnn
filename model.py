import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First Conv Layer: 2 input channels (binary mask and fluorescence), 64 output channels, 3x3x3 kernel, padding=1
        self.conv1_3d = nn.Conv3d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Second Conv Layer: 64 input channels, 128 output channels, 3x3x3 kernel, padding=1
        self.conv2_3d = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # MaxPooling Layer: 2x2x2 kernel, stride 2
        self.pool_3d = nn.MaxPool3d(kernel_size=2, stride=2)

        # For fully connected layer ? To pass global pooled into the sigmoid 
        self.linear_fc = nn.Linear(128, 128)
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        # Layer 1: Conv + ReLU + MaxPooling
        x = self.conv1_3d(x)     # Conv layer
        x = F.relu(x)         # Activation
        x = self.pool_3d(x)      # Downsampling (32x32 -> 16x16)

        # Layer 2: Conv + ReLU + MaxPooling
        x = self.conv2_3d(x)     # Conv layer

        x = F.relu(x)         # Activation
        x = self.pool_3d(x)      # Downsampling (16x16 -> 8x8)

        # Global pooling and then feed flattened data into sigmoid
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # fully flattened to [1x1x1] per channel (and batch)
        x = x.view(x.size(0), -1)  # so that it is actually 1D not [BxNx1x1]
        x = self.linear_fc(x) # fully connected linear layer
        x = self.linear(x)  # flatten the last dim so it's only [B]
        x = torch.sigmoid(x)  # for [0,1] classification
        
        return x