import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First Conv Layer: 3 input channels (RGB), 64 output channels, 3x3 kernel, padding=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Second Conv Layer: 64 input channels, 128 output channels, 3x3 kernel, padding=1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # MaxPooling Layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Layer 1: Conv + ReLU + MaxPooling
        x = self.conv1(x)     # Conv layer
        x = F.relu(x)         # Activation
        x = self.pool(x)      # Downsampling (32x32 -> 16x16)
        
        # Layer 2: Conv + ReLU + MaxPooling
        x = self.conv2(x)     # Conv layer
        x = F.relu(x)         # Activation
        x = self.pool(x)      # Downsampling (16x16 -> 8x8)
        
        return x

# Instantiate the model
model = SimpleCNN()

# Dummy input: batch of 1 image, 3 channels (RGB), 32x32 size
input_tensor = torch.randn(1, 3, 32, 32)

# Forward pass
output = model(input_tensor)

print("Output shape:", output.shape)
