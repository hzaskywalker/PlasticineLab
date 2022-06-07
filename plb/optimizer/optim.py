import numpy as np
from ..config.utils import make_cls_config
from yacs.config import CfgNode as CN

class Optimizer:
    def __init__(self, parameters: np.ndarray, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.lr = self.cfg.lr
        self.bounds = self.cfg.bounds
        self.parameters = parameters
        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def _step(self, grads):
        raise NotImplementedError

    def step(self, grads):
        assert grads.shape == self.parameters.shape
        self.parameters[:] = self._step(grads).clip(*self.bounds)
        return self.parameters.copy()

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.lr = 0.1
        cfg.bounds = (-1., 1.)
        cfg.type = ''
        return cfg


class Momentum(Optimizer):
    def initialize(self):
        self.momentum_buffer = np.zeros_like(self.parameters).astype(np.float64)
        self.momentum = self.cfg.momentum

    def _step(self, grads):
        grads = self.momentum_buffer * self.momentum + grads * (1 - self.momentum)
        self.momentum_buffer[:] = grads
        return self.parameters[:] - self.lr * grads

    @classmethod
    def default_config(cls):
        cfg = Optimizer.default_config()
        cfg.momentum = 0.9
        return cfg


class Adam(Optimizer):
    def initialize(self):
        self.momentum_buffer = np.zeros_like(self.parameters).astype(np.float64)
        self.v_buffer = np.zeros_like(self.momentum_buffer).astype(np.float64)
        self.iter = 0

    def _step(self, grads):
        gd = grads.reshape(*self.parameters.shape)
        beta_1 = self.cfg.beta_1
        beta_2 = self.cfg.beta_2
        epsilon = self.cfg.epsilon
        m_t = beta_1 * self.momentum_buffer + (1 - beta_1) * gd  # updates the moving averages of the gradient
        v_t = beta_2 * self.v_buffer + (1 - beta_2) * (gd * gd)  # updates the moving averages of the squared gradient
        self.momentum_buffer[:] = m_t
        self.v_buffer[:] = v_t

        m_cap = m_t / (1 - (beta_1 ** (self.iter + 1)))  # calculates the bias-corrected estimates
        v_cap = v_t / (1 - (beta_2 ** (self.iter + 1)))  # calculates the bias-corrected estimates

        self.iter += 1
        return self.parameters - (self.lr * m_cap) / (np.sqrt(v_cap) + epsilon)

    @classmethod
    def default_config(cls):
        cfg = Optimizer.default_config()
        cfg.beta_1 = 0.9
        cfg.beta_2 = 0.999
        cfg.epsilon = 1e-8
        return cfg
