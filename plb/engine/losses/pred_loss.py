import taichi as ti
import os
import numpy as np
import torch
from ..mpm_simulator import MPMSimulator

@ti.data_oriented
class PredLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        dtype = self.dtype = ti.f64 # Which may be not
        cfg.SIMULATOR.defrost()
        quality = cfg.SIMULATOR.quality
        if cfg.SIMULATOR.dim == 3:
            quality = quality*0.5
        self.n_grid = int(128*quality)
        self.dx = 1/self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dim = cfg.SIMULATOR.dim
        self.n_particles = cfg.SIMULATOR.n_particles
        self.res = (self.n_grid,self.n_grid) if self.dim == 2 else (self.n_grid,self.n_grid,self.n_grid)


        self.grid_mass = ti.field(dtype=self.dtype,shape=self.res)
        self.particle_x = ti.Vector.field(self.dim,dtype=self.dtype,shape=self.n_particles)
        self.p_vol, self.p_rho = (self.dx*0.5)**2, 1
        self.p_mass = self.p_vol * self.p_rho

        #----------------------------------------
        self.target_density = ti.field(dtype=dtype, shape=self.res)
        self.target_sdf = ti.field(dtype=dtype, shape=self.res)
        self.nearest_point = ti.Vector.field(self.dim, dtype=dtype, shape=self.res)
        self.target_sdf_copy = ti.field(dtype=dtype, shape=self.res)
        self.nearest_point_copy = ti.Vector.field(self.dim, dtype=dtype, shape=self.res)
        self.inf = 1000

        self.sdf_loss = ti.field(dtype=dtype, shape=())
        self.density_loss = ti.field(dtype=dtype, shape=())
        self.loss = ti.field(dtype=dtype, shape=())

        self.sdf_weight = ti.field(dtype=dtype, shape=())
        self.density_weight = ti.field(dtype=dtype, shape=())

    def load_target_density(self, path=None, grids=None):
        if path is not None or grids is not None:
            if path is not None and len(path) > 0:
                grids = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', path))
            else:
                grids = np.array(grids)
            self.target_density.from_numpy(grids)
            self.update_target()

            self.grid_mass.from_numpy(grids)

    def initialize(self):
        self.sdf_weight[None] = self.cfg.ENV.loss.weight.sdf
        self.density_weight[None] = self.cfg.ENV.loss.weight.density

        target_path = self.cfg.ENV.loss.target_path
        self.load_target_density(target_path)

    def set_weights(self, sdf, density, contact, is_soft_contact):
        self.sdf_weight[None] = sdf
        self.density_weight[None] = density

    # -----------------------------------------------------------
    # preprocess target to calculate sdf
    # -----------------------------------------------------------
    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

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
        for i in range(self.n_grid * 2):
            self.update_target_sdf()

    # -----------------------------------------------------------
    # preprocess target to calculate sdf
    # -----------------------------------------------------------

    @ti.kernel
    def compute_density_loss_kernel(self):
        for I in ti.grouped(self.grid_mass):
            self.density_loss[None] += ti.abs(self.grid_mass[I] - self.target_density[I])

    @ti.kernel
    def compute_sdf_loss_kernel(self):
        for I in ti.grouped(self.grid_mass):
            self.sdf_loss[None] += self.target_sdf[I] * self.grid_mass[I]

    # -----------------------------------------------------------
    # compute total loss
    # -----------------------------------------------------------
    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] += self.density_loss[None] * self.density_weight[None]
        self.loss[None] += self.sdf_loss[None] * self.sdf_weight[None]

    @ti.kernel
    def clear_losses(self):
        self.density_loss[None] = 0
        self.sdf_loss[None] = 0

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0

    @ti.complex_kernel
    def compute_loss_kernel(self):
        self.clear_losses()
        #clear and compute grid mss()
        self.grid_mass.fill(0)
        self.compute_grid_mass()

        self.compute_density_loss_kernel()
        self.compute_sdf_loss_kernel()
        self.sum_up_loss_kernel()

    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    def _extract_loss(self):
        self.compute_loss_kernel()
        return self.loss[None]

    def reset(self):
        self.clear_loss()

    def compute_loss(self):
        loss = self._extract_loss()
        return loss

    def clear(self):
        self.clear_loss()

    @ti.kernel
    def compute_grid_mass(self):
        for p in range(0, self.n_particles):
            base = (self.particle_x[p]*self.inv_dx - 0.5).cast(int)
            fx = self.particle_x[p]*self.inv_dx - base.cast(self.dtype)
            w = [0.5*(1.5-fx)**2, 0.75 - (fx - 1)**2, 0.5*(fx - 0.5)**2]
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_mass[base + offset] += weight * self.p_mass

    @ti.kernel
    def set_particle_x(self,x:ti.ext_arr()):
        for p in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                self.particle_x[p][d] = x[p,d]

    def __call__(self,state):
        self.set_particle_x(state)
        loss = self.compute_loss()
        self.reset()
        return torch.tensor(loss).float()
