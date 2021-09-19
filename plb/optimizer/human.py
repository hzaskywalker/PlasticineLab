import taichi as ti
import cv2 as cv
import numpy as np
import time
import copy
from yacs.config import CfgNode as CN

from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config
from ..engine.losses import Loss
from ..interface import ChopsticksInterface

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}

class Solver:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger
        self.interface = ChopsticksInterface()

    def solve(self, init_actions=None, callbacks=()):
        env = self.env
        if init_actions is None:
            init_actions = self.init_actions(env, self.cfg)
        # initialize ...
        optim = OPTIMS[self.optim_cfg.type](init_actions, self.optim_cfg)
        # set softness ..
        env_state = env.get_state() # initial state
        self.total_steps = 0
        self.pc_cnt = 0
        action_buffer = []

        def forward(sim_state):
            if self.logger is not None:
                self.logger.reset()
            print(env._is_copy)
            env.set_state(sim_state, self.cfg.softness,True)
            for i in range(200):
                env.save_current_state('before/squeeze_before/{}'.format(self.pc_cnt))
                action = self.interface.squeeze()
                if not isinstance(action,np.ndarray):
                    break
                env.step(action)
                action_buffer.append(action)
                env.render()
                self.total_steps += 1
                self.pc_cnt += 1
                print("Step",self.total_steps)
                #env.compute_loss()
                env.save_current_state('after/squeeze_after/{}'.format(self.pc_cnt))

        forward(env_state['state'])
        #actions = optim.step(grad) # Here we have access to gradient with respect to all actions how about state
        env.set_state(**env_state)
        np.save('action_squeeze.npy',action_buffer)
        return None


    @staticmethod
    def init_actions(env, cfg):
        action_dim = env.primitives.action_dim
        horizon = cfg.horizon
        if cfg.init_sampler == 'uniform':
            return np.random.uniform(-cfg.init_range, cfg.init_range, size=(horizon, action_dim))
        else:
            raise NotImplementedError

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.optim = Optimizer.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 50

        cfg.init_range = 0.
        cfg.init_sampler = 'uniform'
        return cfg


def human_control(env, path, logger, args):
    import os, cv2
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    T = env._max_episode_steps
    solver = Solver(taichi_env, logger, None,
                    n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    solver.solve()