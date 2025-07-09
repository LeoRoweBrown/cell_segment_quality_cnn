from model import SimpleCNN
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset_handling import Tiff3DDataset, ClassificationDataset, CombinedTrainingSet
from train import Trainer

# Parameters, constants, etc.
batch_size = 4

# Instantiate the model
model = SimpleCNN()

# Loss function
loss_fn = torch.nn.MSELoss(reduction='sum')

# Optimizer, just use SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

training_img_dir = "./data/run_0001_field_0002/"
training_classif_dir = "./data_plots2/run_0001_field_0002/"

dataset = Tiff3DDataset(training_img_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

label_set = ClassificationDataset(training_classif_dir + 'ratings.json')
label_loader = DataLoader(label_set, batch_size=batch_size)

combined_training_set = CombinedTrainingSet(dataset, label_set)

training_set, validation_set = torch.utils.data.random_split(combined_training_set, [0.8, 0.2])
training_loader = DataLoader(training_set, batch_size=batch_size)
validation_loader = DataLoader(validation_set, batch_size=batch_size)

trainer = Trainer(model=model, train_loader=training_loader, validation_loader=validation_loader, n_epochs=20)
trainer.train()

# Should I do label smoothing? e.g., label = label * (1 - eps) + 0.5 * eps

# Forward pass
# # output = model(input_tensor)

# print("Output shape:", output.shape)
