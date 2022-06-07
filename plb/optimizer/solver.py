import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN

from .optim import Optimizer, Adam, Momentum
from .utils import EarlyStopper
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

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
        # if patience is not specified, it is infinite,
        # and early stopper will work simply like a logger
        self.early_stopper = EarlyStopper(patience=self.cfg.patience)

    def solve(self, init_actions=None, callbacks=(), ignore_intermediate_loss=False):
        env = self.env
        if init_actions is None:
            init_actions = self.init_actions(env, self.cfg)
        # initialize ...
        optim = OPTIMS[self.optim_cfg.type](init_actions, self.optim_cfg)
        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        def forward(sim_state, action):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                for i in range(len(action)):
                    env.step(action[i])
                    self.total_steps += 1
                    if not ignore_intermediate_loss or i == len(action) - 1:
                        loss_info = env.compute_loss()
                        if self.logger is not None:
                            self.logger.step(None, None, loss_info['reward'], None, i == len(action) - 1, loss_info)
            loss = env.loss.loss[None]
            return loss, env.primitives.get_grad(len(action))

        best_action = None
        self.early_stopper.reset(np.inf)

        actions = init_actions
        for iter in range(self.cfg.n_iters):
            self.params = actions.copy()
            loss, grad = forward(env_state['state'], actions)
            stopping, improved = self.early_stopper(loss, iter)
            if stopping:
                print('[solver] loss has not improved - early stopping!')
                break
            if improved:
                best_action = actions.copy()
            actions = optim.step(grad)
            for callback in callbacks:
                callback(self, optim, loss, grad)

        env.set_state(**env_state)
        return best_action

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
        cfg.patience = None  # patience is infinite by default
        return cfg


def solve_action(env, path, logger, args):
    import os, cv2
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env

    T = args.horizon
    env._max_episode_steps = T
    solver = Solver(taichi_env, logger, None,
                    n_iters=(args.num_steps + T - 1) // T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    action = solver.solve()

    for idx, act in enumerate(action):
        env.step(act)
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])
