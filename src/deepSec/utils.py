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
            if k in ['NWD', 'learning_rate']:
                setattr(self, k, eval(v))
            else:
                setattr(self, k, v)

class CustomLoss(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.focal_loss = False
        if cfg and hasattr(cfg, 'focal_loss') and cfg.focal_loss:
            focal_loss = True
            self.gamma = 2.0
            self.alpha = None
            self.reduction = 'mean'
            if hasattr(cfg, 'gamma'):
                self.gamma = cfg.gamma
            if hasattr(cfg, 'alpha'):
                self.alpha = cfg.alpha
            if hasattr(cfg, 'reduction'):
                self.reduction = cfg.reduction
        if not self.focal_loss:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        if not self.focal_loss:
            return self.loss_fn(y_pred, y_true)
        else:
            log_probs = torch.nn.functional.log_softmax(y_pred, dim=1)
            probs = torch.exp(log_probs)
    
            log_pt = log_probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
            pt = probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
    
            loss = -((1 - pt) ** self.gamma) * log_pt
    
            if self.alpha is not None:
                alpha_t = self.alpha[y_true]
                loss = alpha_t * loss
    
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss


class EarlyStopping:
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
