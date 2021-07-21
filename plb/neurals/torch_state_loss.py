import torch
import torch.nn as nn
import numpy as np
import taichi as ti
from ..engine.mpm_simulator import MPMSimulator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StateLoss(nn.Module):
    def __init__(self,density_weight,sdf_weight,sim : MPMSimulator,device):
        super(StateLoss,self).__init__()
        self.density_weight = density_weight
        self.sdf_weight = sdf_weight
        # Read related properties from sim
        self.res = sim.res
        self.n_grid = sim.n_grid
        self.dx = sim.dx
        self.inv_dx = sim.inv_dx
        self.p_mass = sim.p_mass
        self.inf = 1000
        self.n_particles = sim.n_pariticles
        dtype = sim.dtype
        if dtype == ti.f64:
            self.dtype = torch.float32
        elif dtype == ti.f64:
            self.dtype = torch.float64

    def computeDensity(self,x):
        density = torch.zeros(self.res).float().to(device)
        base = (x*self.inv_dx-0.5).int()
        fx = x*self.inv_dx-base.float()
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # Each particle will update its adjacent area
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    offset = torch.tensor([i,j,k]).to(device)
                    weight = torch.ones(self.n_particles).float().to(device)
                    for d in range(self.dim):
                        assert(weight.shape==w[offset[d]][:,d].shape)
                        weight *= w[offset[d]][:,d]
                    index = (base+offset).long()
                    unique_index,unique_id = index.unique(dim=0,return_inverse=True)
                    rep_id = unique_id.repeat(len(unique_index),1)
                    rep_weight = weight.repeat(len(unique_index),1)
                    mask = (rep_id == torch.arange(len(unique_index)).to(device).view(-1,1))
                    rep_blank = torch.zeros_like(rep_weight).to(device)
                    rep_blank[mask] = rep_weight[mask]
                    cum_weight = rep_blank.sum(1)
                    density[unique_index[:,0],unique_index[:,1],unique_index[:,2]] += cum_weight*self.p_mass
        return density

    def computeDensityLoss(self,density,density_ref):
        loss = torch.abs(density-density_ref).sum()
        return loss

    '''
    def computeSDFLoss(self,sdf,sdf_ref):
        loss = (sdf*sdf_ref).sum()
        return loss
    '''

    def forward(self,x,x_ref):
        density = self.computeDensity(x)
        density_ref = self.computeDensity(x_ref)
        #sdf = self.computeSDF(x)
        #sdf_ref = self.computeSDF(x_ref)
        density_loss = self.computeDensityLoss(density,density_ref)
        #sdf_loss = self.computeSDFLoss(sdf,sdf_ref)
        loss = density_loss
        return loss


