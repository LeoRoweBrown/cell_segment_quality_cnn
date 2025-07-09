import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataset_handling import Tiff3DDataset

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First Conv Layer: 2 input channels (binary mask and fluorescence), 64 output channels, 3x3x3 kernel, padding=1
        self.conv1_3d = nn.Conv3d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Second Conv Layer: 64 input channels, 128 output channels, 3x3x3 kernel, padding=1
        self.conv2_3d = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # MaxPooling Layer: 2x2x2 kernel, stride 2
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # For fully connected layer
        self.linear = nn.Linear(128, 128)

    def forward(self, x):
        # Layer 1: Conv + ReLU + MaxPooling
        x = self.conv1(x)     # Conv layer
        x = F.relu(x)         # Activation
        x = self.pool(x)      # Downsampling (32x32 -> 16x16)
        
        # Layer 2: Conv + ReLU + MaxPooling
        x = self.conv2(x)     # Conv layer
        x = F.relu(x)         # Activation
        x = self.pool(x)      # Downsampling (16x16 -> 8x8)

        # Global pooling and then feed flattened data into sigmoid
        x = F.adaptive_avg_pool3d(x, (1, 1))  # fully flattened to 1x1 per channel
        x = x.view(x.size(0), -1)  # so that it is actually 1D not Nx1x1
        x = torch.sigmoid(x)  # for classification
        
        return x

# Instantiate the model
model = SimpleCNN()

# Loss function
torch.nn.MSELoss(reduction='sum')

# Optimizer, just use SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy input: batch of 1 image, 3 channels (RGB), 32x32 size
input_tensor = torch.randn(1, 3, 32, 32)

training_dir = ""
dataset = Tiff3DDataset(training_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# TODO split into the training and validation

# Should I do label smoothing? e.g., label = label * (1 - ε) + 0.5 * ε

# Forward pass
output = model(input_tensor)

print("Output shape:", output.shape)
