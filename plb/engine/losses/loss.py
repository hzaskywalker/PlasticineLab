import taichi as ti
import os
import numpy as np
from ..mpm_simulator import MPMSimulator

@ti.data_oriented
class Loss:
    def __init__(self, cfg, sim:MPMSimulator):
        self.cfg = cfg
        dtype = self.dtype = sim.dtype
        self.res = sim.res
        self.n_grid = sim.n_grid
        self.dx = sim.dx
        self.dim = sim.dim
        self.n_particles = sim.n_particles

        self.grid_mass = sim.grid_m
        self.particle_x = sim.x

        self.primitives = []
        for i in range(len(sim.primitives)):
            primitive = sim.primitives[i]
            if primitive.action_dim > 0: # only consider the moveable one
                self.primitives.append(primitive)

        self.compute_grid_mass = sim.compute_grid_m_kernel

        #----------------------------------------
        self.target_density = ti.field(dtype=dtype, shape=self.res)
        self.target_sdf = ti.field(dtype=dtype, shape=self.res)
        self.nearest_point = ti.Vector.field(self.dim, dtype=dtype, shape=self.res)
        self.target_sdf_copy = ti.field(dtype=dtype, shape=self.res)
        self.nearest_point_copy = ti.Vector.field(self.dim, dtype=dtype, shape=self.res)
        self.inf = 1000

        self.sdf_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.density_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.contact_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.loss = ti.field(dtype=dtype, shape=(), needs_grad=True)

        self.sdf_weight = ti.field(dtype=dtype, shape=())
        self.density_weight = ti.field(dtype=dtype, shape=())
        self.contact_weight = ti.field(dtype=dtype, shape=())
        self.soft_contact_loss = False

    def load_target_density(self, path=None, grids=None):
        if path is not None or grids is not None:
            if path is not None and len(path) > 0:
                grids = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', path))
            else:
                grids = np.array(grids)
            self.target_density.from_numpy(grids)
            self.update_target()

            self.grid_mass.from_numpy(grids)
            self.iou()
            self._target_iou = self._iou

    def initialize(self):
        self.sdf_weight[None] = self.cfg.weight.sdf
        self.density_weight[None] = self.cfg.weight.density
        self.contact_weight[None] = self.cfg.weight.contact
        self.soft_contact_loss = self.cfg.soft_contact

        target_path = self.cfg.target_path
        self.load_target_density(target_path)

    def set_weights(self, sdf, density, contact, is_soft_contact):
        self.sdf_weight[None] = sdf
        self.density_weight[None] = density
        self.contact_weight[None] = contact
        self.soft_contact_loss = is_soft_contact

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

    @ti.func
    def soft_weight(self, d):
        return 1/(1+d*d*10000)

    @ti.kernel
    def compute_contact_distance_normalize(self, f: ti.i32):
        for i in range(self.n_particles):
            for primitive in ti.static(self.primitives):
                d_ij = max(primitive.sdf(f, self.particle_x[f, i]), 0.)
                ti.atomic_add(primitive.dist_norm[None], self.soft_weight(d_ij))

    @ti.kernel
    def compute_contact_distance_kernel(self, f: ti.i32):
        for i in range(self.n_particles):
            for primitive in ti.static(self.primitives):
                d_ij = max(primitive.sdf(f, self.particle_x[f, i]), 0.)
                ti.atomic_min(primitive.min_dist[None], max(d_ij, 0.))

    @ti.kernel
    def compute_soft_contact_distance_kernel(self, f: ti.i32):
        for i in range(self.n_particles):
            for primitive in ti.static(self.primitives):
                d_ij = max(primitive.sdf(f, self.particle_x[f, i]), 0.)
                ti.atomic_add(primitive.min_dist[None], d_ij * self.soft_weight(d_ij)/primitive.dist_norm[None])

    @ti.kernel
    def compute_contact_loss_kernel(self):
        for j in ti.static(self.primitives):
            self.contact_loss[None] += j.min_dist[None] ** 2

    # -----------------------------------------------------------
    # compute density and sdf loss
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
        self.loss[None] += self.contact_loss[None] * self.contact_weight[None]
        self.loss[None] += self.density_loss[None] * self.density_weight[None]
        self.loss[None] += self.sdf_loss[None] * self.sdf_weight[None]

    @ti.kernel
    def clear_losses(self):
        self.contact_loss[None] = 0
        self.density_loss[None] = 0
        self.sdf_loss[None] = 0

        self.contact_loss.grad[None] = 0
        self.density_loss.grad[None] = 0
        self.sdf_loss.grad[None] = 0

        for p in ti.static(self.primitives):
            #p.min_dist[None] = 1000000
            p.min_dist[None] = 0 # only for softmin
            p.min_dist.grad[None] = 0
            p.dist_norm[None] = 0
            p.dist_norm.grad[None] = 0


    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0

    @ti.complex_kernel
    def compute_loss_kernel(self, f):
        self.clear_losses()
        if not self.soft_contact_loss:
            for p in self.primitives:
                p.min_dist[None] = 100000

        #clear and compute grid mss(f)
        self.grid_mass.fill(0)
        self.compute_grid_mass(f)

        self.compute_density_loss_kernel()
        self.compute_sdf_loss_kernel()

        if len(self.primitives) > 0:
            if self.soft_contact_loss:
                self.compute_contact_distance_normalize(f)
                self.compute_soft_contact_distance_kernel(f)
            else:
                self.compute_contact_distance_kernel(f)
            self.compute_contact_loss_kernel()

        self.sum_up_loss_kernel()

    @ti.complex_kernel_grad(compute_loss_kernel)
    def compute_loss_kernel_grad(self, f):
        self.clear_losses()
        if not self.soft_contact_loss:
            for p in self.primitives:
                p.min_dist[None] = 100000

        self.sum_up_loss_kernel.grad()

        if len(self.primitives)>0:
            if self.soft_contact_loss:
                self.compute_contact_distance_normalize(f)
                self.compute_soft_contact_distance_kernel(f)
            else:
                self.compute_contact_distance_kernel(f)
            self.compute_contact_loss_kernel.grad()
            if self.soft_contact_loss:
                self.compute_soft_contact_distance_kernel.grad(f)
                self.compute_contact_distance_normalize.grad(f)
            else:
                self.compute_contact_distance_kernel.grad(f)

        self.grid_mass.fill(0.)
        self.grid_mass.grad.fill(0.)
        self.compute_grid_mass(f) # get the grid mass tensor...
        self.compute_sdf_loss_kernel.grad()
        self.compute_density_loss_kernel.grad()
        self.compute_grid_mass.grad(f) # back to the particles..

    @ti.kernel
    def iou_kernel(self)->ti.float64:
        ma = ti.cast(0., self.dtype)
        mb = ti.cast(0., self.dtype)
        I = ti.cast(0., self.dtype)
        Ua = ti.cast(0., self.dtype)
        Ub = ti.cast(0., self.dtype)
        for i in ti.grouped(self.grid_mass):
            ti.atomic_max(ma, self.grid_mass[i])
            ti.atomic_max(mb, self.target_density[i])
            I += self.grid_mass[i]  * self.target_density[i]
            Ua += self.grid_mass[i]
            Ub += self.target_density[i]
        I = I/ma/mb
        U = Ua/ma + Ub/mb
        return I/(U - I)

    def iou2(self, a, b):
        I = np.sum(a * b)
        return I / (np.sum(a) + np.sum(b) - I)

    @ti.complex_kernel
    def iou(self):
        self._iou = self.iou_kernel()

    @ti.complex_kernel_grad(iou)
    def iou_grad(self):
        # no grad
        pass

    def _extract_loss(self, f):
        self.compute_loss_kernel(f)
        self.iou()
        return {
            'loss': self.loss[None],
            'contact_loss': self.contact_loss[None],
            'density_loss': self.density_loss[None],
            'sdf_loss': self.sdf_loss[None],
            'iou': self._iou,
            'target_iou': self._target_iou
        }

    def reset(self):
        self.clear_loss()
        loss_info = self._extract_loss(0)
        self._start_loss = loss_info['loss']
        self._init_iou = loss_info['iou']
        self._last_loss = 0 # in optim, loss will be clear after ti.Tape; for RL; we reset loss to zero in each step.

    def compute_loss(self, f):
        loss_info = self._extract_loss(f)
        r = self._start_loss - (loss_info['loss'] - self._last_loss)
        cur_step_loss = loss_info['loss'] - self._last_loss
        self._last_loss = loss_info['loss']

        incremental_iou = max(min((loss_info['iou']-self._init_iou)/(loss_info['target_iou'] - self._init_iou), 1), 0)
        loss_info['reward'] = r
        loss_info['incremental_iou'] = incremental_iou
        loss_info['loss'] = cur_step_loss
        return loss_info

    def clear(self):
        self.clear_loss()
        self._last_loss = 0