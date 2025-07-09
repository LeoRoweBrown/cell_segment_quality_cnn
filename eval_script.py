import torch
import numpy as np
from torch.utils.data import DataLoader
from model import SimpleCNN
from dataset_handling import Tiff3DDataset

def eval(model, data_loader):
    model.eval()
    outputs = np.array([])
    for i, data in enumerate(data_loader):
        output = model(data).detach().numpy()
        outputs = np.append(outputs, output)
    return outputs

model = SimpleCNN()

run_field = "run_0001_field_0002"
data_dir = "./data/" + run_field
plot_dir = "./data_plots2/" + run_field
dataset = Tiff3DDataset(data_dir=data_dir)
data_loader = DataLoader(dataset=dataset, batch_size=4)

outputs = eval(model, data_loader=data_loader)
print(outputs)