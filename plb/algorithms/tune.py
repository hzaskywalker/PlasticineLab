import argparse
import random
import numpy as np
import torch

from functools import partial
import os
import torch.nn as nn
import torch.nn.functional as F

from plb.envs import make
from plb.optimizer.solver_torch_nn import solve_torch_nn
from plb.optimizer.solver_lstm import solve_lstm
from plb.optimizer.solver_torch_nnv2 import solve_torch_nnv2


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Move-v1")
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=50 * 200)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=666.)

    args = parser.parse_args()

    return args


def tune_mlp():
    args = get_args()

    for env_name in ["Move-v1"]:  # ,
        args.env_name = env_name
        env = make(args.env_name, nn=False, sdf_loss=args.sdf_loss,
                   density_loss=args.density_loss, contact_loss=args.contact_loss,
                   soft_contact_loss=args.soft_contact_loss)

        for af in ['LeakyReLU', 'ReLU', 'Tanh']:  # 'ReLU' , 'LeakyReLU', 'Tanh'
            # (64, 64), (400, 300), (100, 50, 25)]:
            for hidden in [(256, 256)]:
                for seed in [0]:  # , 10, 20
                    args.seed = seed
                    args.hidden = hidden
                    args.af = af
                    args.lr = 1e-4

                    set_random_seed(args.seed)
                    env.reset()
                    env.seed(args.seed)

                    solve_torch_nnv2(env, args)


def tune_lstm():
    args = get_args()

    for env_name in ["Move-v1"]:  # ,
        args.env_name = env_name
        env = make(args.env_name, nn=False, sdf_loss=args.sdf_loss,
                   density_loss=args.density_loss, contact_loss=args.contact_loss,
                   soft_contact_loss=args.soft_contact_loss)

        for seed in [0]:
            args.seed = seed
            args.lr = 1e-3

            set_random_seed(args.seed)
            env.reset()
            env.seed(args.seed)

            solve_lstm(env, args)


if __name__ == '__main__':
    tune_mlp()
    # tune_lstm()
