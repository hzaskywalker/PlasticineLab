import taichi as ti
import numpy as np
import torch
from yacs.config import CfgNode as CN
import copy

from .optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config
from ..neurals.autoencoder import PCNAutoEncoder
from ..neurals.pcdataloader import ChopSticksDataset
from ..engine.losses import compute_emd
# Need to use the optimzier from pytorch
device = torch.device('cuda:0')

class Solver:
    def __init__(self,env: TaichiEnv,model,optimizer,logger=None, cfg=None, decay_factor=0.99,**kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.env = env
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.num_substep = env.simulator.substeps
        # For Debug
        self.last_target = None
        self.decay_factor = decay_factor

    # The logic for multistep solve should be handled later
    def solve(self,state,action,target,grad_buffer,state_buffer,loss_buffer):
        env = self.env
        def forward(state,target,action):
            action = [action]
            env.set_state(state, self.cfg.softness, False)
            # The target only use target position instead of target velocity
            env.set_target(target[0]) # same as previous loss but target is changed to next state
            with ti.Tape(env.loss.loss):
                env.step(action[0])
                env.compute_loss()
            loss = env.loss.loss[None]
            return loss, env.get_state_grad()

        x = torch.from_numpy(state[0]).double().to(device)
        x_hat = self.model(x.float())
        loss_first, assignment = compute_emd(x,x_hat,3000)
        x_hat_after = x_hat[assignment.detach().long()]
        x_hat = x_hat_after
        state_buffer.append(x_hat)
        loss_buffer.append(loss_first)
        state_hat = copy.deepcopy(state)
        state_hat[0] = x_hat.cpu().double().detach().numpy()
        # The gradient return is numpy array
        loss, (x_hat_grad,_) = forward(state_hat,target,action)
        x_hat_grad = torch.from_numpy(x_hat_grad).clamp(-1,1).to(device)
        if not torch.isnan(x_hat_grad).any():
            grad_buffer.append(x_hat_grad)
        else:
            if len(state_buffer)!=0:
                state_buffer.pop()
        return loss

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
    def solve_multistep(self,state,actions,targets,grad_buffer,state_buffer,loss_buffer):
        env = self.env
        def forward(state,targets,actions):
            env.set_state(state, self.cfg.softness, False)
            #print("Cursor Before:",env.simulator.cur)
            with ti.Tape(env.loss.loss):
                for i in range(len(targets)):
                    env.set_target(targets[i])
                    env.step(actions[i])
                    env.compute_loss(copy_grad=False,decay=self.decay_factor)
                env.set_grad()
            loss = env.loss.loss[None]
            #print("Cursor After:", env.simulator.cur,"Substeps: ",env.simulator.substeps)
            return loss, env.get_state_grad()
        x = torch.from_numpy(state[0]).double().to(device)
        x_hat = self.model(x.float())
        loss_first,assignment = compute_emd(x, x_hat, 3000)
        x_hat_after = x_hat[assignment.detach().long()]
        x_hat = x_hat_after
        #print("Cursor:",env.simulator.cur)
        state_buffer.append(x_hat)
        loss_buffer.append(loss_first)
        state_hat = copy.deepcopy(state)
        state_hat[0] = x_hat.cpu().double().detach().numpy()
        loss, (x_hat_grad,_) = forward(state_hat,targets,actions)
        x_hat_grad = torch.from_numpy(x_hat_grad).clamp(-1,1).to(device)
        if not torch.isnan(x_hat_grad).any():
            grad_buffer.append(x_hat_grad)
        else:
            if len(state_buffer)!=0:
                state_buffer.pop()
        return loss
                    
def update_network(optimizer,state_buffer,grad_buffer,loss_buffer,use_loss=True):
    optimizer.zero_grad()
    batch_size = len(state_buffer)
    for i,x_state in enumerate(state_buffer):
        x_state.backward(grad_buffer[i]/batch_size,retain_graph=True)
    if use_loss:
        batch_size = len(loss_buffer)
        for loss in loss_buffer:
            loss.backward()
    loss_buffer[:] = []
    #optimizer.step()
    state_buffer[:] = []
    grad_buffer[:] = []

# Need to create specific dataloader for such task
def learn_latent(env, path, args):
    import os
    os.makedirs(path,exist_ok=True)
    env.reset()
    taichi_env : TaichiEnv = env.unwrapped.taichi_env
    T = env._max_episode_steps
    print("TaichiEnv Number of Particles:",taichi_env.n_particles)
    model = PCNAutoEncoder(n_particles = taichi_env.n_particles,
                           hidden_dim = 256,
                           latent_dim = 1024,
                           feat_dim=3)
    model.load_state_dict(torch.load("pretrain_model/network_emd.pth")['net_state_dict'])
    model = model.to(device)
    #torch.save(model.encoder.state_dict(),'pretrain_model/emd_encoder.pth')
    #exit(0)
    optimizer = torch.optim.Rprop(model.parameters(),lr=args.lr)
    solver = Solver(taichi_env,model,optimizer,None,None,softness=args.softness,horizon=T,
                    **{"optim.lr": args.lr, "optim.type":args.optim, "init_range":0.0001})
    dataset = ChopSticksDataset()
    dataloader = DataLoader(dataset,batch_size=1)
    epochs = 5
    batch_loss = 0
    batch_cnt = 0
    batch_size = args.batch_size
    state_buffer = []
    grad_buffer = []
    loss_buffer = []
    for i in range(epochs):
        total_loss = 0
        batch_cnt = 0
        for state,target,action in dataloader:
            state = [state[0].squeeze().numpy(),state[1].squeeze().numpy(),
                     state[2].squeeze().numpy(),state[3].squeeze().numpy()]
            targets = target[0].squeeze().numpy()
            actions = action.squeeze()
            current_loss = solver.solve_multistep(state,actions,targets,grad_buffer,state_buffer, loss_buffer)
            total_loss += current_loss
            batch_loss += current_loss
            if batch_cnt % batch_size == 0 and batch_cnt != 0:
                
                update_network(optimizer=optimizer,
                               state_buffer=state_buffer,
                               grad_buffer = grad_buffer,
                               loss_buffer=loss_buffer)
                
                print("Batch:",batch_cnt//batch_size,"Loss: ",batch_loss/batch_size)
                batch_loss = 0
            batch_cnt += 1
        print("Loss: ",total_loss/batch_cnt)
    print("Total Average Loss:",total_loss/batch_cnt)
    torch.save(model.state_dict(),"pretrain_model/weight_after_lr2e-5.pth")
