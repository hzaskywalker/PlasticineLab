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


class SolverLSTMv2:
    def __init__(self, env, logger=None, data_dir='', **kwargs):
        self.cfg = make_cls_config(self, None, **kwargs)
        self.env = env
        self.logger = logger
        self.data_dir = data_dir
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.nn = nn.LSTM(env.observation_space.shape[0], env.action_space.shape[0], 2).double().to(self.device)
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
        taichi_env.set_torch_nn(self.nn)
        with ti.Tape(loss=taichi_env.loss.loss):
            for i in range(self.cfg.horizon):
                action = taichi_env.act(obs)  # Need to be wrapped
                #action_np = action.data.cpu().numpy()
                obs, reward, done, loss_info = self.env.step(action)
                if self.logger is not None:
                    self.logger.step(
                        None, None, reward, None, i == self.cfg.horizon-1, loss_info)
        loss = taichi_env.loss.loss[None]

        # nn.utils.clip_grad_value_(self.nn.parameters(), clip_value=1.0)
        # nn.utils.clip_grad_norm_(self.nn.parameters(),
        #                          max_norm=1.0, norm_type=2)

        # self.logger.summary_writer.writer.add_histogram(
        #     'output layer grad', self.nn.linears[2].weight.grad, epoch)

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
        # cfg.nn = MLP.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 50

        return cfg


def solve_lstmv2(env, args):
    import os
    import cv2

    T = env._max_episode_steps

    # nn_name = f"lstmv2_gv-{100.0}"
    nn_name = "lstmv2"

    exp_name = f"{nn_name}_{args.env_name}_horizon-{T}_lr-{args.lr}"

    path = f"data/{exp_name}/{exp_name}_s{args.seed}"
    os.makedirs(path, exist_ok=True)
    logger = Logger(path, exp_name)
    env.reset()

    solver = SolverLSTMv2(env, logger, data_dir=path,
                           n_iters=200,
                           softness=args.softness, horizon=T,
                           **{"optim.lr": args.lr})

    actions = solver.solve()
    # actions = solver.inference()

    with imageio.get_writer(f"{path}/output.gif", mode="I") as writer:
        for idx, act in enumerate(actions):
            _, reward, _, _ = env.step(act)
            img = env.render(mode='rgb_array')
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            writer.append_data(img)
