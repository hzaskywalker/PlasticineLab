import argparse
import os

import random
import numpy as np
import torch

from plb.mpi import mpi_tools

if os.getenv("IN_MPI") is not None:
    cudaCore = mpi_tools.proc_id() % mpi_tools.NUM_CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cudaCore}'
    mpi_tools.msg(f"DEBUG GPU CORE>>>>>>> cuda:{cudaCore}")
    
from plb.engine import taichi_env
from plb.envs import make
from plb.algorithms.logger import Logger
from plb.algorithms.sac.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn
from plb.optimizer.solver_torch_nn import solve_torch_nn
from plb.optimizer.learn_latent import learn_latent
from plb.optimizer.focal_learn_latent import learn_latent_focal
from plb.optimizer.human import human_control
from plb.engine.losses import Loss, StateLoss, ChamferLoss, EMDLoss

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn', 'torch_nn']

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default=DIFF_ALGOS + RL_ALGOS)
    parser.add_argument("--env_name", type=str, default="Move-v1")
    parser.add_argument("--path", type=str, default='./tmp')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=None)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    parser.add_argument("--srl", action='store_true', default=False)
    parser.add_argument("--loss",type=str,default='chamfer')
    parser.add_argument("--batch_size",type=int,default=30)
    parser.add_argument("--exp_name",type=str,default=None,required=True)
    parser.add_argument("--model_name",type=str,default=None)
    parser.add_argument("--horizon",type=int,default=None)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    if args.num_steps is None:
        if args.algo in DIFF_ALGOS:
            args.num_steps = 50 * 200
        else:
            args.num_steps = 50000

    logger = Logger(args.path)
    set_random_seed(args.seed)
    if args.algo=='one_step' or args.algo=='focal':
        if args.loss == 'voxel_mae':
            loss_fn = StateLoss
        elif args.loss == 'chamfer':
            loss_fn = ChamferLoss
        elif args.loss == 'emd':
            loss_fn = EMDLoss
    else:
        loss_fn = Loss

    if args.algo == 'one_step':
        learn_latent(args, loss_fn)
    elif args.algo == 'focal':
        learn_latent_focal(args,loss_fn)
    else:
        taichi_env.init_taichi()
        env = make(args.env_name, nn=(args.algo=='nn'), sdf_loss=args.sdf_loss,loss_fn = loss_fn,
                                density_loss=args.density_loss, contact_loss=args.contact_loss,full_obs = args.srl,
                                soft_contact_loss=args.soft_contact_loss)
        env.seed(args.seed)
        if args.algo == 'sac':
            train_sac(env, args.path, logger, args)
        elif args.algo == 'action':
            solve_action(env, args.path, logger, args)
        elif args.algo == 'ppo':
            train_ppo(env, args.path, logger, args)
        elif args.algo == 'td3':
            train_td3(env, args.path, logger, args)
        elif args.algo == 'nn':
            solve_nn(env, args.path, logger, args)
        elif args.algo == 'torch_nn':
            solve_torch_nn(env, args)
        elif args.algo == 'human':
            human_control(env,args.path,logger,args)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    main()
