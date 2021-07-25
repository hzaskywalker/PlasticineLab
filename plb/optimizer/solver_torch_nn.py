import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN
import torch
from torch import nn

from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, output_dim)
        )
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.epochs = 5
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits


class SolverTorchNN:
    def __init__(self, env, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nn = MLP(env.observation_space.shape[0], env.action_space.shape[0]).double(
        ).to(self.device)

    def train(self):
        if self.logger is not None:
            self.logger.reset()
        taichi_env = self.env.unwrapped.taichi_env
        actions = []
        obs = self.env.reset()
        taichi_env.set_copy(False)
        print("=======================")
        with ti.Tape(loss=taichi_env.loss.loss):
            for i in range(self.cfg.horizon):
                state_tensor = torch.as_tensor(obs).to(self.device)
                action_var = self.nn(state_tensor)
                actions.append(action_var)
                action_np = action_var.data.cpu().clone().numpy()
                obs, reward, done, loss_info = self.env.step(action_np)

                if self.logger is not None:
                    self.logger.step(
                        None, None, reward, None, i == self.cfg.horizon-1, loss_info)
        loss = taichi_env.loss.loss[None]
        print('loss: ', loss)

        grads = taichi_env.primitives.get_grad(self.cfg.horizon)
        for action, grad in zip(actions, grads):
            grad_tensor = torch.as_tensor(grad).to('cuda')
            action.backward(grad_tensor, retain_graph=True)

        self.nn.optimizer.step()
        actions_np = [t.data.cpu().numpy() for t in actions]
        return loss, actions_np

    def solve(self, init_actions=None, callbacks=()):
        best_actions = None
        best_loss = 1e10
        for iter in range(self.cfg.n_iters):
            self.nn.optimizer.zero_grad()
            loss, actions = self.train()

            if loss < best_loss:
                best_loss = loss
                best_actions = actions.copy()
            # for callback in callbacks:
            #     callback(self, optim, loss, grad)

        self.env.reset()
        print("actions: ", best_actions)
        return best_actions

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


def solve_torch_nn(env, path, logger, args):
    import os
    import cv2
    os.makedirs(path, exist_ok=True)
    env.reset()

    T = env._max_episode_steps
    solver = SolverTorchNN(env, logger, None,
                           n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                           **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    actions = solver.solve()

    for idx, act in enumerate(actions):
        env.step(act)
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])
