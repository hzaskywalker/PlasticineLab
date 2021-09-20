import os
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import taichi as ti
from ..mpm_simulator import MPMSimulator
from ...chamfer_distance import ChamferDistance
#from chamferdist import ChamferDistance
#sys.path.append(os.path.join("ChamferDistancePytorch"))

#from chamfer3D import dist_chamfer_3D

# Use Chamfer Loss which will be slower but more consistant with pretrain.

class ChamferLoss(nn.Module):
    def __init__(self,cfg,sim : MPMSimulator):
        super(ChamferLoss,self).__init__()
        self.cfg = cfg
        dtype = self.dtype = sim.dtype
        self.sim = sim
        self.loss = ti.field(dtype=ti.f64,shape=(), needs_grad=True)
        self.loss_fn = ChamferDistance()
        #self.loss_fn = nn.MSELoss()
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
        output = torch.from_numpy(self.sim.get_x(self.sim.cur)).float().cuda()
        output.requires_grad_()
        s = output.shape
        dist1,dist2 = self.loss_fn(self.target.view(1,s[0],s[1]), output.view(1,s[0],s[1]))
        loss = (torch.sqrt(dist1).mean(1)+torch.sqrt(dist2).mean(1))/2
        #loss = self.loss_fn(self.target.view(1,s[0],s[1]), output.view(1,s[0],s[1]))
        loss = (loss.mean()*10)/(decay**(cur//self.sim.substeps))
        self.loss[None] = float(loss)
        loss.backward()
        # set the gradient of x at step 1
        if copy_grad:
            self.sim.set_x_grad(cur,output.grad.double().cpu())
        else:
            self.grad_buffer.append(output.grad.double().cpu())
            self.cur_buffer.append(cur)
            self.cum_loss.append(float(loss))

    def set_grad(self):
        for cur,grad in zip(self.cur_buffer,self.grad_buffer):
            self.sim.set_x_grad(cur,grad)
        self.loss[None] = sum(self.cum_loss)/len(self.grad_buffer)
        self.grad_buffer = []
        self.cur_buffer = []
        cum_loss = self.cum_loss
        #print("Cum Loss",self.cum_loss)
        self.cum_loss = []
        return cum_loss