import numpy as np


class EarlyStopper:
    def __init__(self, patience=None, delta=0.):
        if patience is None:
            patience = np.inf

        self.patience = patience
        self.patience_cnt = 0
        self.best_loss = np.inf
        self.best_idx = None
        self.delta = delta

    def __call__(self, curr_loss, index):

        stopping = False
        improved = False
        if curr_loss < self.best_loss - self.delta:
            self.best_loss = curr_loss
            self.best_idx = index
            self.patience_cnt = 0
            improved = True
        else:
            self.patience_cnt += 1
            if self.patience_cnt >= self.patience:
                stopping = True

        return stopping, improved

    def get_best_index(self):
        return self.best_idx

    def reset(self, loss):
        self.patience_cnt = 0
        self.best_loss = loss
        self.best_idx = None
