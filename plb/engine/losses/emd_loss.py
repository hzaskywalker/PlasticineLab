import torch
import torch.nn as nn
from .emd_module import compute_emd
import taichi as ti
from ..mpm_simulator import MPMSimulator

class EMDLoss(nn.Module):
    def __init__(self, cfg, sim: MPMSimulator):
        self.cfg = cfg
        dtype = self.dtype = sim.dtype
        self.sim = sim
        self.loss = ti.field(dtype=ti.f64, shape=(), needs_grad=True)
        self.iters = 3000
        self.grad_buffer = []
        self.cur_buffer = []
        self.cum_loss = []

    def set_target(self,target):
        self.target = torch.from_numpy(target).float().cuda()

    def initialize(self):
        pass

    def clear(self):
        pass

    def reset(self):
        pass

    def compute_loss(self,cur,copy_grad,decay):
        output = torch.from_numpy(self.sim.get_x_nokernel()).float().cuda()
        output.requires_grad_()
        loss,_ = compute_emd(self.target,output,self.iters)
        loss = loss*decay**(cur//self.sim.substeps)
        self.loss[None] = float(loss)
        loss.backward()
        if copy_grad:
            self.sim.set_x_grad(output.grad.double().cpu(),cur)
        else:
            self.grad_buffer.append(output.grad.double().cpu())
            self.cur_buffer.append(cur)
            self.cum_loss.append(loss)

    def set_grad(self):
        for cur,grad in zip(self.cur_buffer,self.grad_buffer):
            self.sim.set_x_grad(grad,cur)
        self.loss[None] = sum(self.cum_loss)/len(self.grad_buffer)
        self.grad_buffer = []
        self.cur_buffer = []
        cum_loss = sum(self.cum_loss)
        self.cum_loss = []
        return cum_loss
