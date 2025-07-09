from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from model import SimpleCNN
from dataset_handling import Tiff3DDataset, ClassificationDataset, CombinedTrainingSet

class Trainer:
    def __init__(self, model, train_loader, optimizer=None, loss_fn=None, validation_loader=None, n_epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(model.parameters(), lr=0.01)
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss(reduce='max')
        self.n_epochs = n_epochs


    def train(self):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/training_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = self.n_epochs

        best_vloss = 1.e6  # start with high loss (bad)

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    # print("validation input size", vinputs.size())
                    voutputs = self.model(vinputs)
                    # print("===done on eval model!===")

                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
        
        return self.model

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            labels = labels.view(-1, 1).float()  # ensure it keeps the [B, 1] size if batch goes to 1
            # print("size of input", inputs.size())
            # print("target output size (labels)", labels.size())

            self.optimizer.zero_grad()  # reset grad each batch with SGD

            # Make predictions for this batch
            outputs = self.model(inputs)
            # print("size of outputs", outputs.size())
            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report every 50 batches (100 samples?)
            running_loss += loss.item()
            if i % 50 == 49:
                last_loss = running_loss / 50 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
        return last_loss
