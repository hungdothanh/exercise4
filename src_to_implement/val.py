import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


best_ckp = 21
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('data.csv', sep=';')

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
batch_size = 64

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_data)
val_dataset = ChallengeDataset(val_data)

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# create an instance of our ResNet model
resnet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.BCELoss()
optimizer = t.optim.Adam(resnet.parameters(), lr=0.00005, weight_decay = 0.0005)
trainer = Trainer(resnet, criterion, optimizer, train_loader, val_loader, cuda=False, early_stopping_patience=5)

trainer.restore_checkpoint(best_ckp)
# trainer.set_early_stopping(patience=5)

# go, go, go... call fit on trainer
val_loss, val_f1, val_f1_mean = trainer.val_test()

print(f"Val F1 (crack): {val_f1[0]:.4f}, Val F1 (inactive): {val_f1[1]:.4f}, Mean F1: {val_f1_mean:.4f}")