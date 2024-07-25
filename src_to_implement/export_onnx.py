import torch as t
from trainer import Trainer
import sys
import torchvision as tv
import model

resnet = model.ResNet()
epoch = int(sys.argv[1])
#TODO: Enter your model here

crit = t.nn.BCELoss()
trainer = Trainer(resnet, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
