# Need to use the optimzier from pytorch
from argparse import Namespace
import copy
import os

import taichi as ti
import torch
from torch.utils.data.dataloader import DataLoader
from typing import Any, Tuple, Type, Union
from yacs.config import CfgNode as CN

from .optim import Optimizer
from .. import mpi
from ..config.utils import make_cls_config
from ..engine import taichi_env
from ..engine.losses import compute_emd
from ..engine.losses import state_loss, emd_loss, chamfer_loss, loss
from ..engine.taichi_env import TaichiEnv
from ..envs import make
from ..neurals.autoencoder import PCNAutoEncoder
from ..neurals.pcdataloader import ChopSticksDataset

HIDDEN_LAYERS = 256
LATENT_DIMS   = 1024
FEAT_DMIS     = 3
MPI_ENABLE    = False

if MPI_ENABLE:
    mpi.setup_pytorch_for_mpi()

class Solver:
    def __init__(self,
            env: TaichiEnv,
            model,
            optimizer,
            logger=None,
            cfg=None,
            decay_factor=0.99,
            steps=None,
            **kwargs
        ):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.env = env
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.num_substep = env.simulator.substeps
        # For Debug
        self.last_target = None
        self.decay_factor = decay_factor
        self.steps = steps

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

    # For multiple step target only support chamfer and emd loss cannot use default loss
    # Here the state might be problematic since only four frame is considered insided of primitives
    def solve_multistep(
            self, state, actions, targets, localDevice:torch.device
        ) -> Tuple[Union[list, None], Union[torch.Tensor, None], torch.Tensor, Any]:
        """ Run the model on the given state and action`s`. 

        The model will forward the input state for given steps, and
        observe how much it derives from the expected `target` to
        compute the losses and generate the gradient from optimizing.

        The method CAN be executed in a multi-process way.
        
        :param state: a list of states from dataloader
        :param actions: actions to be executed
        :param target: the expected target after the execution, from dataloader as well
        :param local_device: to which CPU/GPU device should the execution be loaded
        :return: a tuple of (resulting states, gradient, loss_first, loss)
        NOTE the first two elements can be None, when there is NAN in the gradient array
        """
        env = self.env
        def forward(state,targets,actions):
            env.set_state(state, self.cfg.softness, False)
            #print("Cursor Before:",env.simulator.cur)
            if self.steps == None:
                steps = len(targets)
            else:
                steps = self.steps if ((self.steps<len(targets))and (self.steps>0)) else len(targets)
            with ti.Tape(env.loss.loss):
                for i in range(steps):
                    env.set_target(targets[i])
                    env.step(actions[i])
                    env.compute_loss(copy_grad=False,decay=self.decay_factor)
                env.set_grad()
            loss = env.loss.loss[None]
            return loss, env.get_state_grad()
        x = torch.from_numpy(state[0]).double().to(localDevice)
        x_hat = self.model(x.float())
        loss_first,assignment = compute_emd(x, x_hat, 3000)
        x_hat_after = x_hat[assignment.detach().long()]
        x_hat = x_hat_after
        state_hat = copy.deepcopy(state)
        state_hat[0] = x_hat.cpu().double().detach().numpy()
        loss, (x_hat_grad,_) = forward(state_hat,targets,actions)
        x_hat_grad = torch.from_numpy(x_hat_grad).clamp(-1,1).to(localDevice)
        if not torch.isnan(x_hat_grad).any():
            return x_hat, x_hat_grad, loss_first, loss
        else:
            return None, None, loss_first, loss
                    
def _update_network_mpi(model: torch.nn.Module, optimizer, state, gradient, loss, use_loss=True):
    if state is not None and gradient is not None:
        optimizer.zero_grad()
        state.backward(gradient, retain_graph=True)
        if use_loss:
            loss.backward()
    if MPI_ENABLE: mpi.avg_grads(model)
    if state is not None and gradient is not None:
        optimizer.step()

# Need to create specific dataloader for such task
def _loading_dataset()->DataLoader:
    """ Load data to memory

    :return: a dataloader of ChopSticksDataset
    """
    dataset = ChopSticksDataset()
    dataloader = DataLoader(dataset,batch_size = mpi.num_procs() if MPI_ENABLE else 4)
    return dataloader

def _intialize_env(
    envName: str,
    sdfLoss: float, 
    lossFn:  Union[Type[chamfer_loss.ChamferLoss], Type[emd_loss.EMDLoss], Type[state_loss.StateLoss], Type[loss.Loss]],
    densityLoss: float,
    contactLoss: float,
    srl: bool, 
    softContactLoss: bool,
    seed: int
) -> Tuple[TaichiEnv, int]:
    """ Intialize the environment from the arguments

    The parameters all come from the arguments

    :return: the intialized taichi environment, together with the max episode step of this env
    """
    taichi_env.init_taichi()
    env = make(
        env_name          = envName,
        nn                = False,
        sdf_loss          = sdfLoss,
        loss_fn           = lossFn,
        density_loss      = densityLoss,
        contact_loss      = contactLoss,
        full_obs          = srl,
        soft_contact_loss = softContactLoss
    )
    env.seed(seed)
    env.reset()
    T = env._max_episode_steps
    return env.unwrapped.taichi_env, T

def _intialize_model(taichiEnv: TaichiEnv, device: torch.device)->PCNAutoEncoder:
    """ Intialize the model from a given TaichiEnv onto a certain device

    :param taichiEnv: the environment
    :param device: the device to which the model should be loaded to
    :return: the intialized encoding model
    """
    model = PCNAutoEncoder(taichiEnv.n_particles, HIDDEN_LAYERS, LATENT_DIMS, FEAT_DMIS)
    model.load_state_dict(torch.load("pretrain_model/network_emd_finetune.pth")['net_state_dict'])
    torch.save(model.encoder.state_dict(),'pretrain_model/emd_expert_encoder2.pth')
    model = model.to(device)
    return model

def learn_latent(
        args:Namespace,
        loss_fn:Union[Type[chamfer_loss.ChamferLoss], Type[emd_loss.EMDLoss], Type[state_loss.StateLoss], Type[loss.Loss]]
    ):
    """ Learn latent in the MPI way

    NOTE: neither the Taichi nor the PlasticineEnv shall be
    intialized outside, since the intialization must be
    executed in sub processes. 

    :param args: Arguments passed from the solver.py, determining
        the hyperparameters, the paths and the random seeds. 
    :param loss_fn: the loss function for environment intialization. 
    """
    # before MPI FORK: intialization & data loading
    os.makedirs(args.path, exist_ok=True)
    epochs, batch_loss, batch_cnt, batch_size = 2, 0, 0, args.batch_size, 

    # After MPI FORK
    if MPI_ENABLE: mpi.fork(mpi.best_mpi_subprocess_num(batch_size, procPerGPU=2))
    procLocalDevice = torch.device("cuda")

    dataloader = _loading_dataset()
    taichiEnv, T = _intialize_env(args.env_name, args.sdf_loss, loss_fn, args.density_loss,
                                  args.contact_loss, args.srl, args.soft_contact_loss, args.seed)
    model = _intialize_model(taichiEnv, procLocalDevice)
    optimizer = torch.optim.Rprop(model.parameters(), lr=args.lr)
    if MPI_ENABLE:
        mpi.msg(f"TaichiEnv Number of Particles:{taichiEnv.n_particles}")
    else:
        print(f"TaichiEnv Number of Particles:{taichiEnv.n_particles}")

    solver = Solver(
        env       = taichiEnv,
        model     = model,
        optimizer = optimizer,
        logger    = None,
        cfg       = None,
        steps     = args.horizon,
        softness  = args.softness, 
        horizon   = T,
        **{"optim.lr": args.lr, "optim.type":args.optim, "init_range":0.0001}
    )

    for i in range(epochs):
        total_loss = 0
        batch_cnt = 0
        for stateMiniBatch, targetMiniBatch, actionMiniBatch, indexMiniBatch in dataloader:
            stateProc = list(mpi.batch_collate(
                stateMiniBatch[0], stateMiniBatch[1], stateMiniBatch[2], stateMiniBatch[3], stateMiniBatch[4],
                toNumpy=True
            )) if MPI_ENABLE else stateMiniBatch
            targetProc, actionProc, indexProc = mpi.batch_collate(
                targetMiniBatch[0], actionMiniBatch, indexMiniBatch, 
                toNumpy=True
            ) if MPI_ENABLE else (targetMiniBatch, actionMiniBatch, indexMiniBatch)
            result_state, gradient, lossInBuffer, currentLoss = solver.solve_multistep(
                state=stateProc,
                actions=actionProc,
                targets=targetProc,
                localDevice = procLocalDevice
            )
            total_loss += currentLoss
            batch_loss += currentLoss

            _update_network_mpi(
                model=model,
                optimizer=optimizer,
                state=result_state,
                gradient=gradient,
                loss=lossInBuffer
            )

            if MPI_ENABLE: mpi.sync_params(model)
            if MPI_ENABLE:
                mpi.msg(f"Batch:{batch_cnt}, loss:{batch_loss}")
            else:
                print(f"Batch:{batch_cnt}, loss:{batch_loss}")
            batch_loss = 0
            batch_cnt += 1
        if MPI_ENABLE:
            mpi.msg(f"Epoch:{i}, average loss:{total_loss/batch_cnt}")
        else:
            print(f"Epoch:{i}, average loss:{total_loss/batch_cnt}")
    if MPI_ENABLE:
        mpi.msg(f"Total average loss: {total_loss/batch_cnt}")
    else:
        print(f"Total average loss: {total_loss/batch_cnt}")

    if not MPI_ENABLE or mpi.proc_id() == 0:
        # ONLY one proc can store the model
        torch.save(model.state_dict(),"pretrain_model/emd_finetune_expert2.pth")
        torch.save(model.encoder.state_dict(),"pretrain_model/emd_finetune_expert_encoder2.pth")
