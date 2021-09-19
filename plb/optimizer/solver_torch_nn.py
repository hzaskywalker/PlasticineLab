import os
import imageio
import taichi as ti
import numpy as np
import pandas as pd
import seaborn as sns
from yacs.config import CfgNode as CN
import torch
from torch import nn
import torch.nn.functional as F

from plb.algorithms.logger import Logger
from .optim import Optimizer
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

AF = {
    "Tanh": F.tanh,
    "ReLU": F.relu
}


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(256, 256), activation="Tanh"):
        super(MLP, self).__init__()
        self.af = AF[activation]
        dims = (input_dim,) + hidden + (output_dim,)
        self.linears = nn.ModuleList(
            [nn.Linear(dim, dims[i+1]) for i, dim in enumerate(dims[:-1])])

    def forward(self, x):
        for l in self.linears[:-1]:
            x = self.af(l(x))
        logits = self.linears[-1](x)
        logits = F.hardtanh(logits, -1., 1.)
        return logits

    @ classmethod
    def default_config(cls):
        cfg = CN()
        cfg.hidden = (256, 256)
        cfg.af = "Tanh"
        return cfg


class SolverTorchNN:
    def __init__(self, env, logger=None, data_dir='', **kwargs):
        self.cfg = make_cls_config(self, None, **kwargs)
        self.env = env
        self.logger = logger
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nn = MLP(env.observation_space.shape[0], env.action_space.shape[0],
                      hidden=self.cfg.nn.hidden, activation=self.cfg.nn.af).double().to(self.device)
        self.learning_rate = self.cfg.optim.lr
        self.optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=self.learning_rate)

    def train(self, epoch):
        if self.logger is not None:
            self.logger.reset()
        taichi_env = self.env.unwrapped.taichi_env
        actions = []
        obs = self.env.reset()
        taichi_env.set_copy(False)
        with ti.Tape(loss=taichi_env.loss.loss):
            for i in range(self.cfg.horizon):
                state_tensor = torch.as_tensor(obs).to(self.device)
                action_var = self.nn(state_tensor) # Need to be wrapped
                actions.append(action_var)
                action_np = action_var.data.cpu().numpy()
                obs, reward, done, loss_info = self.env.step(action_np)

                if self.logger is not None:
                    self.logger.step(
                        None, None, reward, None, i == self.cfg.horizon-1, loss_info)
        loss = taichi_env.loss.loss[None]

        grads = taichi_env.primitives.get_grad(self.cfg.horizon)
        self.logger.summary_writer.writer.add_histogram(
            'primitives grad', grads, epoch)

        for action, grad in zip(actions, grads):
            grad_tensor = torch.as_tensor(grad).to('cuda')
            action.backward(grad_tensor, retain_graph=True)

        self.optimizer.step()
        actions_np = [t.data.cpu().numpy() for t in actions]
        return loss, actions_np

    def solve(self, callbacks=()):
        best_actions = None
        best_model = None
        best_loss = 1e10
        for iter in range(self.cfg.n_iters):
            self.optimizer.zero_grad()
            loss, actions = self.train(iter)

            if loss < best_loss:
                best_loss = loss
                best_actions = actions.copy()
                best_model = self.nn.state_dict().copy()

            for callback in callbacks:
                callback(loss, actions)

        torch.save(best_model, os.path.join(
            self.data_dir, 'model_weights.pth'))

        self.env.reset()
        # self.logger.summary_writer.writer.add_graph(self.nn)
        self.logger.summary_writer.writer.close()
        return best_actions

    def inference(self):
        self.nn.load_state_dict(torch.load(
            os.path.join(self.data_dir, 'model_weights.pth')))
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

    @ classmethod
    def default_config(cls):
        cfg = CN()
        cfg.optim = Optimizer.default_config()
        cfg.nn = MLP.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 50

        return cfg


def solve_torch_nn(env, args):
    import os
    import cv2

    exp_name = f"mlp_{args.env_name}_hidden-{args.hidden}_lr-{args.lr}_af-{args.af}"
    path = f"data/{exp_name}/{exp_name}_s{args.seed}"
    os.makedirs(path, exist_ok=True)
    logger = Logger(path, exp_name)
    env.reset()

    T = env._max_episode_steps
    solver = SolverTorchNN(env, logger, data_dir=path,
                           n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                           **{"optim.lr": args.lr, "nn.hidden": args.hidden, "nn.af": args.af})

    actions = solver.solve()
    # actions = solver.inference()

    with imageio.get_writer(f"{path}/output.gif", mode="I") as writer:
        for idx, act in enumerate(actions):
            _, reward, _, _ = env.step(act)
            img = env.render(mode='rgb_array')
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            writer.append_data(img)
