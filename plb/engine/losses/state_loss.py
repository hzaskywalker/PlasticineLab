import taichi as ti
import os
import numpy as np
from ..mpm_simulator import MPMSimulator

# Need the loss need to be learnd load from numpy point cloud
# Usage before entering the with ti.Tape() load the target point cloud to the loss funciton
# Within the tape compute loss
# The reset equivalence should be invoked within env loss. 

@ti.data_oriented
class StateLoss:
    def __init__(self,cfg, sim: MPMSimulator):
        self.cfg = cfg
        dtype = self.dtype = sim.dtype
        self.res = sim.res
        self.n_grid = sim.n_grid
        self.dx = sim.dx
        self.dim = sim.dim
        self.n_particles = sim.n_particles
        self.p_mass = sim.p_mass

        self.grid_mass = sim.grid_m
        self.particle_x = sim.x
        self.buffer_x = ti.Vector.field(self.dim,dtype=dtype,shape=self.n_particles)
        self.inv_dx = sim.inv_dx
        self.dx = sim.dx
        self.compute_grid_mass = sim.compute_grid_m_kernel

        self.target_density = ti.field(dtype=dtype,shape=self.res)
        self.target_sdf = ti.field(dtype=dtype,shape=self.res)
        self.target_sdf_copy = ti.field(dtype=dtype,shape=self.res)
        self.nearest_point = ti.Vector.field(self.dim,dtype=dtype, shape=self.res)
        self.nearest_point_copy = ti.Vector.field(self.dim,dtype=dtype,shape=self.res)
        #self.grid_mass = 
        self.inf = 1000

        self.sdf_loss = ti.field(dtype=dtype,shape=(),needs_grad = True)
        self.density_loss = ti.field(dtype=dtype,shape=(),needs_grad =True)
        self.loss = ti.field(dtype=dtype,shape=(),needs_grad = True)
        self.sdf_weight = ti.field(dtype=dtype, shape=())
        self.density_weight = ti.field(dtype=dtype,shape=())

    # Here we need to assume input is numpy array
    # WARNING: This function cannot be invoked with in grad tape

    def stencil_range(self):
        return ti.ndrange(*((3,)*self.dim))

    # How should we convert the field to be a tensor field
    def load_x_for_m(self,x):
        self.buffer_x.from_numpy(x)
    
    @ti.kernel
    def clear_target_density(self):
        for I in ti.grouped(self.target_density):
            self.target_density[I] = 0

    @ti.kernel
    def compute_grid_m_kernel(self):
        for p in range(self.n_particles):
            base = (self.buffer_x[p]*self.inv_dx-0.5).cast(int)
            fx = self.buffer_x[p]*self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.target_density[base + offset] += weight*self.p_mass

    @ti.kernel
    def update_target_sdf(self):
        for I in ti.grouped(self.target_sdf):
            self.target_sdf[I] = self.inf
            grid_pos = ti.cast(I * self.dx, self.dtype)
            if self.target_density[I] > 1e-4: #TODO: make it configurable
                self.target_sdf[I] = 0.
                self.nearest_point[I] = grid_pos
            else:
                for offset in ti.grouped(ti.ndrange(*(((-3, 3),)*self.dim))):
                    v = I + offset
                    if v.min() >= 0 and v.max() < self.n_grid and ti.abs(offset).sum() != 0:
                        if self.target_sdf_copy[v] < self.inf:
                            nearest_point = self.nearest_point_copy[v]
                            dist = self.norm(grid_pos - nearest_point)
                            if dist < self.target_sdf[I]:
                                self.nearest_point[I] = nearest_point
                                self.target_sdf[I] = dist
        for I in ti.grouped(self.target_sdf):
            self.target_sdf_copy[I] = self.target_sdf[I]
            self.nearest_point_copy[I] = self.nearest_point[I]

    def update_target(self):
        self.target_sdf_copy.fill(self.inf)
        for _ in range(self.n_grid*2):
            self.update_target_sdf()

    def set_target(self,target):
        self.clear_target_density()
        self.load_x_for_m(target)
        self.compute_grid_m_kernel()
        self.update_target()

    def set_weights(self,sdf=None,density=None):
        self.sdf_weight[None] = sdf if sdf != None else self.cfg.weight.sdf/(self.cfg.weight.sdf+self.cfg.weight.density)
        self.density_weight[None] = density if density != None else self.cfg.weight.density/(self.cfg.weight.sdf+self.cfg.weight.density)

    def initialize(self):
        self.set_weights()

    @ti.func
    def norm(self,x,eps=1e-8):
        return ti.sqrt(x.dot(x)+eps)

    # Here is assume that the grid mass will directly obtained from the simulator
    # Which seems to be problematic ...
    # The compute density loss has always has access to the grid content
    @ti.kernel
    def compute_density_loss_kernel(self):
        for I in ti.grouped(self.grid_mass):
            self.density_loss[None] += ti.abs(self.grid_mass[I] - self.target_density[I])
    
    @ti.kernel
    def compute_sdf_loss_kernel(self):
        for I in ti.grouped(self.grid_mass):
            self.sdf_loss[None] += self.target_sdf[I]*self.grid_mass[I]

    @ti.kernel
    def clear_losses(self):
        self.density_loss[None] = 0
        self.sdf_loss[None] = 0
        self.density_loss.grad[None] = 0
        self.sdf_loss.grad[None] = 0

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0

    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] += self.density_loss[None]*self.density_weight[None]
        self.loss[None] += self.sdf_loss[None]*self.sdf_weight[None]

    # Why the loss computation need complex kernel??
    # Here the f will be set to be zero
    @ti.complex_kernel
    def compute_loss_kernel(self,f):
        self.clear_losses()
        self.grid_mass.fill(0)
        self.compute_grid_mass(f)
        self.compute_density_loss_kernel()
        self.compute_sdf_loss_kernel()
        self.sum_up_loss_kernel()

    @ti.complex_kernel_grad(compute_loss_kernel)
    def compute_loss_kernel_grad(self,f):
        self.clear_losses()
        self.sum_up_loss_kernel.grad()
        self.grid_mass.fill(0.)
        self.grid_mass.grad.fill(0.)
        self.compute_grid_mass(f)
        self.compute_sdf_loss_kernel.grad()
        self.compute_density_loss_kernel.grad()
        self.compute_grid_mass.grad(f)

    def _extract_loss(self,f):
        self.compute_loss_kernel(f)
        return {'loss':self.loss[None],
                'density_loss':self.density_loss[None],
                'sdf_loss':self.sdf_loss[None]}

    def reset(self):
        self.clear_loss()
        loss_info = self._extract_loss(0)

    # For the state representation learning no need to use delta loss
    def compute_loss(self,f):
        loss_info = self._extract_loss(f)
        loss_info['reward'] = -loss_info['loss']
        loss_info['loss'] = loss_info['loss']
        return loss_info

    def clear(self):
        self.clear_loss()
        self.clear_losses()



    

        


