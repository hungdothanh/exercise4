import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('./src_to_implement/data.csv', sep=';')

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_data)
val_dataset = ChallengeDataset(val_data)

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# create an instance of our ResNet model
resnet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.MSELoss()
optimizer = t.optim.Adam(resnet.parameters(), lr=0.001)
trainer = Trainer(resnet, criterion, optimizer, train_loader, val_loader, cuda=False, early_stopping_patience=5)
# trainer.set_early_stopping(patience=5)

# go, go, go... call fit on trainer
train_loss, val_loss = trainer.fit(epochs=10)

# plot the results
plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
plt.plot(np.arange(len(val_loss)), val_loss, label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')