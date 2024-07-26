import torch as t
from trainer import Trainer
import sys
import torchvision as tv
import model
import pandas as pd


resnet = model.ResNet()
criterion = t.nn.BCELoss()
optimizer = t.optim.Adam(resnet.parameters(), lr=0.001)
trainer = Trainer(resnet, criterion, cuda=False)

epoch = int(sys.argv[1])

trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))

