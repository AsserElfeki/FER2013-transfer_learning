import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


hps = {
'CNN':    {
    'criterion' : nn.CrossEntropyLoss(),
    'optimizer' : optim.SGD,
    'activation_func': F.relu,
    'learning_rate' : 0.01,
    'epochs' : 100,
    'batch_size' : 32,
    'scheduler' : optim.lr_scheduler.ExponentialLR,
    'dropout_rate' : 0.1
    },
'RESNET': {
    'criterion' : nn.CrossEntropyLoss(),
    'optimizer' : optim.Adam,
    'activation_func': F.relu,
    'learning_rate' : 0.001,
    'epochs' : 10,
    'batch_size' : 32,
    'scheduler' : optim.lr_scheduler.ExponentialLR,
    'dropout_rate' : 0.1
}
}