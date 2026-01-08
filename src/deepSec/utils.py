import numpy as np
import torch
torch.manual_seed(42)
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import seaborn as sns
from importlib import resources

class Config:
    def __init__(self, config):
        for k, v in config.items():
            if k in ['NWD', 'LR', 'learning_rate']:
                setattr(self, k, eval(v))
            else:
                setattr(self, k, v)

class EarlyStop:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.val_loss_min = np.inf
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.counter = 0
            self.best_epoch = epoch
        elif val_loss > self.val_loss_min + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
