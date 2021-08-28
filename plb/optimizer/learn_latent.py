# Need to use the optimzier from pytorch
import copy
import os
from typing import Any, Tuple, Union


import taichi as ti
import torch
from torch.utils.data.dataloader import DataLoader
from yacs.config import CfgNode as CN

from .optim import Optimizer
from ..engine.losses import compute_emd
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config
from ..neurals.autoencoder import PCNAutoEncoder
from ..neurals.pcdataloader import ChopSticksDataset
from ..mpi import mpi_tools, mpi_pytorch

mpi_pytorch.setup_pytorch_for_mpi()
device = torch.device('cuda:0')

class Solver:
    def __init__(self,
                 env: TaichiEnv,
                 model,
                 optimizer,
                 logger=None,
                 cfg=None,
                 decay_factor=0.99,
                 steps=None,
                 **kwargs):
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
            self, state, actions, targets, local_device:torch.device=device
        ) -> Tuple[Union[list, None], Union[torch.Tensor, None], torch.Tensor, Union[torch.Tensor, Any]]:
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
            #print("Cursor After:", env.simulator.cur,"Substeps: ",env.simulator.substeps)
            return loss, env.get_state_grad()
        x = torch.from_numpy(state[0]).double().to(local_device)
        x_hat = self.model(x.float())
        loss_first,assignment = compute_emd(x, x_hat, 3000)
        x_hat_after = x_hat[assignment.detach().long()]
        x_hat = x_hat_after
        #print("Cursor:",env.simulator.cur)
        state_hat = copy.deepcopy(state)
        state_hat[0] = x_hat.cpu().double().detach().numpy()
        loss, (x_hat_grad,_) = forward(state_hat,targets,actions)
        x_hat_grad = torch.from_numpy(x_hat_grad).clamp(-1,1).to(local_device)
        if not torch.isnan(x_hat_grad).any():
            return state_hat, x_hat_grad, loss_first, loss
        else:
            return None, None, loss_first, loss
                    
def update_network_mpi(model: torch.nn.Module, optimizer, state, gradient, loss, use_loss=True):
    optimizer.zero_grad()
    state.backward(gradient, retain_graph=True)
    if use_loss:
        loss.backward()
    mpi_pytorch.mpi_avg_grads(model)
    optimizer.step()

# Need to create specific dataloader for such task
def learn_latent(env, path, args):
    # before MPI FORK: intialization & data loading
    os.makedirs(path,exist_ok=True)
    epochs, batch_loss, batch_cnt, batch_size = 2, 0, 0, args.batch_size
    ## Env related
    env.reset()
    taichiEnv: TaichiEnv = env.unwrapped.taichi_env
    T = env._max_episode_steps
    print("TaichiEnv Number of Particles:",taichiEnv.n_particles)
    ## Model related, NOTE load to device after MPI fork
    model = PCNAutoEncoder(n_particles = taichiEnv.n_particles,
                           hidden_dim = 256,
                           latent_dim = 1024,
                           feat_dim=3)
    model.load_state_dict(torch.load("pretrain_model/network_emd_finetune.pth")['net_state_dict'])
    torch.save(model.encoder.state_dict(),'pretrain_model/emd_expert_encoder2.pth')
    optimizer = torch.optim.Rprop(model.parameters(),lr=args.lr)
    ## Data related
    dataset = ChopSticksDataset()
    dataloader = DataLoader(dataset,batch_size=1)
    # ---- # ---- # ---- # ---- #
    # After MPI FORK
    mpi_tools.mpi_fork(mpi_tools.best_mpi_subprocess_num(batch_size))
    procLocalDevice = torch.device("cuda:%d"%(mpi_tools.proc_id() % 4))
    model = model.to(procLocalDevice)
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

    for _ in range(epochs):
        total_loss = 0
        batch_cnt = 0
        for state,target,action in dataloader:
            state = [state[0].squeeze().numpy(),state[1].squeeze().numpy(),
                     state[2].squeeze().numpy(),state[3].squeeze().numpy(),
                     state[4].squeeze().numpy()]
            targets = target[0].squeeze().numpy()
            actions = action.squeeze()
            result_state, gradient, lossInBuffer, currentLos = solver.solve_multistep(
                state=state,
                actions=actions,
                targets=targets,
                local_device = procLocalDevice
            )
            total_loss += currentLos
            batch_loss += currentLos

            if result_state is not None and gradient is not None:
                update_network_mpi(
                    model=model,
                    optimizer=optimizer,
                    state=result_state,
                    gradient=gradient,
                    loss=lossInBuffer
                )

            mpi_pytorch.sync_params(model)

            mpi_tools.msg("Batch:%d"%(batch_cnt//batch_size), "Loss:%d"%(batch_loss/batch_size))
            batch_loss = 0
            batch_cnt += 1
        mpi_tools.msg("Epoch:%d"%(), "loss: ", total_loss/batch_cnt)
    mpi_tools.msgn("", "Total Average Loss:",total_loss/batch_cnt)

    if mpi_tools.proc_id() == 0:
        # ONLY one proc can store the model
        torch.save(model.state_dict(),"pretrain_model/emd_finetune_expert2.pth")
        torch.save(model.encoder.state_dict(),"pretrain_model/emd_finetune_expert_encoder2.pth")
