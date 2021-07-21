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
        self.output_layer = nn.Linear(256, output_dim)
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            self.output_layer
        )
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.epochs = 5
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits


class SolverTorchNN:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nn = MLP(4 + env.n_primitive,
                      env.primitives.action_dim).to(self.device)

    def solve(self, init_actions=None, callbacks=()):
        env = self.env
        # if init_actions is None:
        #     init_actions = self.init_actions(env, self.cfg)
        # initialize ...
        # optim = OPTIMS[self.optim_cfg.type](init_actions, self.optim_cfg)
        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        def forward_ng(sim_state):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)
            # TODO: tensor actions for backward, and np actions for taichi env step

            actions = []
            with ti.Tape(loss=env.loss.loss):
                for i in range(self.cfg.horizen):
                    action_tensor = self.nn(env.get_state())
                    actions.append(action_tensor)
                    action_np = action_tensor.numpy()
                    env.step(action_np)
                    self.total_steps += 1
                    loss_info = env.compute_loss()
                    if self.logger is not None:
                        self.logger.step(
                            None, None, loss_info['reward'], None, i == self.cfg.horizen-1, loss_info)
            loss = env.loss.loss[None]

            grads = env.primitives.get_grad(self.cfg.horizen)
            for action, grad in zip(actions, grads):
                grad_tensor = torch.FloatTensor(grad)
                action.backward(grad_tensor, retain_graph=True)

            self.nn.optimizer.step()
            actions_np = np.concatenate([t.numpy() for t in actions], axis=0)
            return loss, actions_np

        def forward(sim_state, actions):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                for i in range(len(actions)):
                    env.step(actions[i])
                    self.total_steps += 1
                    loss_info = env.compute_loss()
                    if self.logger is not None:
                        self.logger.step(
                            None, None, loss_info['reward'], None, i == len(actions)-1, loss_info)
            loss = env.loss.loss[None]
            return loss, env.primitives.get_grad(len(actions))
        # put taichi grad into pytorch tensor
        # actions now randomly generated, later we need to generate actions from nn(states)
        # use the new pytorch tensor grad to optimize the torch nn

        best_actions = None
        best_loss = 1e10
        # actions = init_actions
        for iter in range(self.cfg.n_iters):
            # self.params = actions.copy()  # not used?
            # loss, grad = forward(env_state['state'], actions)

            # edit here
            self.nn.optimizer.zero_grad()
            loss, actions = forward_ng(env_state['state'])
            ###

            if loss < best_loss:
                best_loss = loss
                best_actions = actions.copy()
            # actions = optim.step(grad)
            # for callback in callbacks:
            #     callback(self, optim, loss, grad)

        env.set_state(**env_state)
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


def solve_action(env, path, logger, args):
    import os
    import cv2
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    T = env._max_episode_steps
    solver = SolverTorchNN(taichi_env, logger, None,
                           n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                           **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    actions = solver.solve()

    for idx, act in enumerate(actions):
        env.step(act)
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])
