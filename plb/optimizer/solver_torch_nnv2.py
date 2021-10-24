import os
import copy
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
from ..neurals import LatentPolicyEncoder, MIMOSequential

AF = {
    "Tanh": F.tanh,
    "ReLU": F.relu,
    "LeakyReLU": F.leaky_relu
}


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(256, 256), max_action=10, activation="LeakyReLU"):
        super(MLP, self).__init__()
        print(f"Use {activation}")
        self.af = AF[activation]
        self.dims = (input_dim+1,) + hidden + (output_dim,)
        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.dims[i+1]) for i, dim in enumerate(self.dims[:-1])])
        self.max_action = float(max_action)

    def forward(self, x,t):
        device = next(self.parameters()).device
        t = torch.tensor([t*0.2]).to(device)
        x = x.to(device)
        x = torch.cat([x,t])
        for l in self.linears[:-1]:
            x = self.af(l(x))
        logits = self.linears[-1](x)
        logits = F.hardtanh(logits,-self.max_action,self.max_action)
        #logits = self.max_action*F.tanh(logits)
        return logits

    def reset_parameters(self,device):
        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.dims[i+1]) for i, dim in enumerate(self.dims[:-1])]).double().to(device)

    @ classmethod
    def default_config(cls):
        cfg = CN()
        cfg.hidden = (256, 256)
        cfg.af = "Tanh"
        return cfg


class SolverTorchNN:
    def __init__(self, env, srl=False,logger=None, data_dir='',model_name='', **kwargs):
        self.cfg = make_cls_config(self, None, **kwargs)
        self.env = env
        self.logger = logger
        self.data_dir = data_dir
        self.obs_type = 'x' if srl else 'vx'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'
        if srl:
            print("Using State Representation Learning")
            encoder = LatentPolicyEncoder(
                n_particles = self.env.n_particles,
                n_layers = 5,
                feature_dim = 3,
                hidden_dim = 256,
                latent_dim = 1024,
                primitive_dim = env.observation_space.shape[0] - 6*self.env.n_particles).to(self.device)
            if model_name != '' and model_name != None:
                print(f"Load model at {model_name}")
                encoder.load_model(f'pretrain_model/{model_name}.pth')
            mlp = MLP(encoder.output_dim,env.action_space.shape[0],
                          hidden = self.cfg.nn.hidden, activation=self.cfg.nn.af).double().to(self.device)
            self.nn = MIMOSequential(encoder,mlp)
        else:
            print("Using sample based representation")
            self.nn = MLP(env.observation_space.shape[0], env.action_space.shape[0],
                          hidden=self.cfg.nn.hidden, activation=self.cfg.nn.af,max_action=env.action_space.high[0]).double().to(self.device)
        self.learning_rate = self.cfg.optim.lr

    def train(self, epoch):
        if self.logger is not None:
            self.logger.reset()
        taichi_env = self.env.unwrapped.taichi_env
        actions = []
        obs = self.env.reset()
        taichi_env.set_copy(False)
        taichi_env.set_torch_nn(self.nn)
        total_reward = 0
        with ti.Tape(loss=taichi_env.loss.loss):
            for i in range(self.cfg.horizon):
                action = taichi_env.act(obs,50-i,self.obs_type)  # Need to be wrapped
                actions.append(action)
                obs, reward, done, loss_info = self.env.step(action)
                total_reward += reward
                if self.logger is not None:
                    self.logger.step(
                        None, None, reward, None, i == self.cfg.horizon-1, loss_info)
        loss = taichi_env.loss.loss[None]
        last_iou = loss_info['incremental_iou']
        #self.logger.summary_writer.writer.add_histogram(
        #    'output layer grad', self.nn.linears[2].weight.grad, epoch)

        self.optimizer.step()
        return loss, actions, total_reward, last_iou

    def solve(self, exp_name, callbacks=()):
        best_rewards_actions = None
        best_last_iou_actions = None
        best_model = None
        base_model = self.nn
        best_loss = 1e10
        best_iou = 0
        runs = 5
        rewards = np.zeros((runs,self.cfg.n_iters))
        for r in range(runs):
            self.nn = copy.deepcopy(base_model)
            if isinstance(self.nn, MIMOSequential):
                self.nn[1].reset_parameters(self.device)
            else:
                self.nn.reset_parameters(self.device)
            self.optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=self.learning_rate)
            print('========================')
            print(f"==== Run {r} starts ====")
            print(f"====== Total {self.cfg.n_iters} ======")
            print('========================')
            for iter in range(self.cfg.n_iters):
                self.optimizer.zero_grad()
                loss, actions, total_reward, last_iou = self.train(iter)
                rewards[r,iter] = total_reward
                if loss < best_loss:
                    best_loss = loss
                    best_rewards_actions = actions.copy()
                    best_model = self.nn.state_dict().copy()
                if last_iou > best_iou:
                    best_iou = last_iou
                    best_last_iou_actions = actions.copy()

                for callback in callbacks:
                    callback(loss, actions)

        torch.save(best_model, os.path.join(
            self.data_dir, 'model_weights.pth'))
        if not os.path.exists('model_based_rewards'):
            os.mkdir('model_based_rewards')    
        np.save(f'model_based_rewards/{exp_name}.npy',rewards)

        self.env.reset()
        # self.logger.summary_writer.writer.add_graph(self.nn)
        self.logger.summary_writer.writer.close()
        return best_rewards_actions, best_last_iou_actions

    @ classmethod
    def default_config(cls):
        cfg = CN()
        cfg.optim = Optimizer.default_config()
        cfg.nn = MLP.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 50

        return cfg


def solve_torch_nnv2(env, args):
    import os
    import cv2

    T = env._max_episode_steps

    nn_name = f"nnv2_gv-{1.0}"
    nn_hidden = (256,256)
    nn_af = 'LeakyReLU'

    exp_name = f"{nn_name}_{args.env_name}_horizon-{T}_hidden-{nn_hidden}_lr-{args.lr}"

    path = f"data/{exp_name}/{exp_name}_s{args.seed}"
    os.makedirs(path, exist_ok=True)
    logger = Logger(path, exp_name)
    env.reset()

    solver = SolverTorchNN(env=env, logger=logger, srl = args.srl, data_dir=path,
                           model_name = args.model_name,n_iters=200,
                           softness=args.softness, horizon=T,
                           **{"optim.lr": args.lr, "nn.hidden": nn_hidden, "nn.af":nn_af})

    reward_actions, last_iou_actions = solver.solve(args.exp_name)
    if not os.path.exists("saved_videos"):
        os.mkdir("saved_videos")
    elif not os.path.exists(f'saved_videos/{args.exp_name}_raw'):
        os.mkdir(f'saved_videos/{args.exp_name}_raw')

    env.taichi_env.set_copy(True)
    if args.render:
        with imageio.get_writer(f"saved_videos/output_{args.exp_name}_best_iou.gif",mode="I") as writer:
            env.reset()
            for idx, act in enumerate(last_iou_actions):
                _, reward, _, _ = env.step(act)
                img = env.render(mode='rgb_array')
                img = cv2.resize(img, (0, 0), fx=1, fy=1)
                writer.append_data(img)
        env.reset()
        with imageio.get_writer(f"saved_videos/output_{args.exp_name}_best_rewards.gif", mode="I") as writer:
            env.reset()
            for idx, act in enumerate(reward_actions):
                _, reward, _, _ = env.step(act)
                img = env.render(mode='rgb_array')
                img = cv2.resize(img, (0, 0), fx=1, fy=1)
                writer.append_data(img)
