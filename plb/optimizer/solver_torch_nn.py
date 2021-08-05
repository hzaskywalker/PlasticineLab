import taichi as ti
import numpy as np
import pandas as pd
import seaborn as sns
from yacs.config import CfgNode as CN
import torch
from torch import nn

from .optim import Optimizer
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, l1=256, l2=256, activation=nn.Tanh):
        super(MLP, self).__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(input_dim, l1),
            activation(),
            nn.Linear(l1, l2),
            activation(),
            nn.Linear(l2, output_dim)
        )

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits


class SolverTorchNN:
    def __init__(self, env, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.env = env
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nn = MLP(env.observation_space.shape[0], env.action_space.shape[0]).double(
        ).to(self.device)
        self.learning_rate = self.cfg.optim.lr
        self.optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=self.learning_rate)

    def train(self):
        if self.logger is not None:
            self.logger.reset()
        taichi_env = self.env.unwrapped.taichi_env
        actions = []
        obs = self.env.reset()
        taichi_env.set_copy(False)
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

        grads = taichi_env.primitives.get_grad(self.cfg.horizon)
        for action, grad in zip(actions, grads):
            grad_tensor = torch.as_tensor(grad).to('cuda')
            action.backward(grad_tensor, retain_graph=True)

        self.optimizer.step()
        actions_np = [t.data.cpu().numpy() for t in actions]
        return loss, actions_np

    def solve(self, callbacks=()):
        best_actions = None
        best_loss = 1e10
        for iter in range(self.cfg.n_iters):
            self.optimizer.zero_grad()
            loss, actions = self.train()

            if loss < best_loss:
                best_loss = loss
                best_actions = actions.copy()
                torch.save(self.nn.state_dict(), 'model_weights.pth')
            for callback in callbacks:
                callback(loss, actions)

        self.env.reset()
        return best_actions

    def inference(self):
        self.nn.load_state_dict(torch.load('model_weights.pth'))
        self.nn.eval()
        actions = []
        obs = self.env.reset()
        for i in range(self.cfg.horizon):
            state_tensor = torch.as_tensor(obs).to(self.device)
            action_var = self.nn(state_tensor)
            actions.append(action_var)
            action_np = action_var.data.cpu().clone().numpy()
            obs, reward, done, loss_info = self.env.step(action_np)
        actions_np = [t.data.cpu().numpy() for t in actions]
        return actions_np

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

    def log_loss(loss, actions):
        print('loss: ', loss)

    actions = solver.solve(callbacks=[log_loss])
    # actions = solver.inference()

    rewards = np.zeros((T,))
    for idx, act in enumerate(actions):
        _, reward, _, _ = env.step(act)
        rewards[idx] = reward
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])

    df = pd.DataFrame(dict(time=np.arange(T),
                           value=np.cumsum(rewards)))
    g = sns.relplot(x="time", y="value", kind="line", data=df)
    g.fig.autofmt_xdate()
    g.savefig("output.png")
