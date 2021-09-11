# Need to use the optimzier from pytorch
from ChamferDistancePytorch.chamfer_python import batched_pairwise_dist
from argparse import Namespace
import copy
import os

import taichi as ti
import torch
from numpy import ndarray
from torch.utils.data.dataloader import DataLoader
from typing import Any, Generator, Tuple, Type, Union
from yacs.config import CfgNode as CN

from .optim import Optimizer
# from .. import mpi
from ..config.utils import make_cls_config
from ..engine import taichi_env
from ..engine.losses import compute_emd
from ..engine.losses import state_loss, emd_loss, chamfer_loss, loss
from ..engine.taichi_env import TaichiEnv
from ..envs import make
from ..neurals.autoencoder import PCNAutoEncoder
from ..neurals.pcdataloader import ChopSticksDataset, RopeDataset

HIDDEN_LAYERS = 256
LATENT_DIMS   = 1024
FEAT_DMIS     = 3


# mpi.setup_pytorch_for_mpi()

class Solver:
    def __init__(self,
            env: TaichiEnv,
            model,
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
            print("NAN Detected")
            return None, None, loss_first, loss


# Need to create specific dataloader for such task
def _loading_dataset(batchSize: int)->DataLoader:
    """ Load data to memory

    :return: a dataloader of ChopSticksDataset
    """
    #dataset = ChopSticksDataset()
    dataset = RopeDataset()
    dataloader = DataLoader(dataset,batch_size = batchSize)
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
    model.load_state_dict(torch.load("pretrain_model/network_emd_finetune_rope.pth")['net_state_dict'])
    torch.save(model.encoder.state_dict(),'pretrain_model/emd_expert_encoder_rope.pth')
    model = model.to(device)
    return model

def squeeze_batch(state):
    state = [state[0].squeeze().numpy(),state[1].squeeze().numpy(),
             state[2].squeeze().numpy(),state[3].squeeze().numpy(),
             state[4].squeeze().numpy()]
    return state

def _single_batch_collect(*batches: torch.Tensor, batchRank: int, batchSize: int) -> Generator[Union[torch.Tensor, ndarray], None, None]:

    batchSize = len(batches[0])
    assert all(len(batch) == batchSize for batch in batches), \
        "all batch must be of the same length, but the lengths " \
        + f"are {[len(batch) for batch in batches]}"
    assert all(hasattr(batch, "squeeze") for batch in batches), \
        "all batch must has squeeze attribution"
    
    return (batch[batchRank % batchSize].squeeze() for batch in batches)
    # batchLen = len(batchs[0]), 
    
    # batchPerProc = max(batchLen // procCnt, 1)
    
    # if batchPerProc == 1 and all(hasattr(batch, "squeeze") for batch in batchs):
    #     gen = (batch[rank % batchLen].squeeze() for batch in batchs)
    # else:
    #     gen = (batch[(rank * batchPerProc) % batchLen : ((rank + 1) * batchPerProc) % batchLen] for batch in batchs)
    
    # if toNumpy and all(isinstance(batch, torch.Tensor) for batch in batchs):
    #     return (batch.numpy() for batch in gen)
    # return gen


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
    epochs, rank = 10, 0

    procLocalDevice = torch.device("cuda")

    dataloader = _loading_dataset(args.batch_size)
    taichiEnv, T = _intialize_env(args.env_name, args.sdf_loss, loss_fn, args.density_loss,
                                  args.contact_loss, args.srl, args.soft_contact_loss, args.seed)
    model = _intialize_model(taichiEnv, procLocalDevice)
    print(f"TaichiEnv Number of Particles:{taichiEnv.n_particles}")

    solver = Solver(
        env       = taichiEnv,
        model     = model,
        logger    = None,
        cfg       = None,
        steps     = args.horizon,
        softness  = args.softness, 
        horizon   = T,
        **{"optim.lr": args.lr, "optim.type":args.optim, "init_range":0.0001}
    )
    
    epochAvgLoss = [0.0] * epochs
    for i in range(epochs):
        efficientStateCnt = 0
        for stateMiniBatch, targetMiniBatch, actionMiniBatch, indexMiniBatch in dataloader:
            batchLossSum = 0.0
            for rank in range(len(indexMiniBatch)):
                singleState = list(_single_batch_collect(
                    stateMiniBatch[0], stateMiniBatch[1], stateMiniBatch[2], stateMiniBatch[3], stateMiniBatch[4],
                    batchRank=rank
                ))
                singleTarget, singleAction, _ = _single_batch_collect(
                    targetMiniBatch[0], actionMiniBatch, indexMiniBatch, 
                    batchRank=rank
                )
                # RUN
                result_state, gradient, _, currentLoss = solver.solve_multistep(
                    state=singleState,
                    actions=singleAction,
                    targets=singleTarget,
                    localDevice = procLocalDevice
                )

                rank += 1
                if result_state is not None and gradient is not None:
                    batchLossSum += currentLoss
                    epochAvgLoss += currentLoss
                    efficientStateCnt += 1
            
            print(f"Batch:{rank}, loss:{batchLossSum / len(indexMiniBatch)}")
            break

        epochAvgLoss[i] /= efficientStateCnt
        print(f"Epoch:{i}, average loss:{epochAvgLoss[i]}")

    totalAverageLoss = sum(epochAvgLoss) / len(epochAvgLoss)
    print(f"Total average loss: {totalAverageLoss}")

    torch.save(model.state_dict(),"pretrain_model/rope_model.pth")
    torch.save(model.encoder.state_dict(),"pretrain_model/rope_encoder.pth")
