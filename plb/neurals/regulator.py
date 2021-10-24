import numpy as np
import taichi as ti
import torch
import torch.nn as nn
from ..engine.losses.emd_module import compute_emd

@ti.data_oriented
class CollisionDetector:
    def __init__(self, n_particles,primitives):
        self.dtype = ti.f64
        self.n_particles = n_particles
        self.primitives = primitives
        self.n_primitives = len(primitives)
        
        # Some terms are hardcoded for convenience
        self.dim = 3
        self.n_grid = 64
        self.dx, self.inv_dx = 1/self.n_grid, float(self.n_grid)
        p_vol, p_rho = (self.dx*0.5)**2, 1
        self.p_mass = self.p_vol * self.p_rho

        # Data Structure
        res = (self.n_grid, self.n_grid, self.n_grid)
        self.grid_m = ti.field(dtype=self.dtype, shape=res, needs_grad=True)
        self.x = ti.Vector.field(self.dim, dtype=self.dtype, shape=(self.n_particles,), needs_grad=True)
        self.x_next = ti.Vector.field(self.dim, dtype=self.dtype, shape=(self.n_particles,), needs_grad=True)
        self.collision_flag = ti.field(dtype=ti.i32, shape=res)
        self.loss = ti.field(dtype=self.dtype, shape=())
        
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    # Inside complex kernel solve
    @ti.kernel
    def enable_loss_grad(self):
        self.loss.grad[None] = 1

    @ti.kernel
    def set_x(self,pos: ti.ext_arr()):
        for i in range(0,self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x[i][j] = pos[i,j]

    @ti.kernel
    def get_new_x(self,pos:ti.ext_arr()):
        for i in range(0,self.n_particles):
            for j in ti.static(range(self.dim)):
                pos[i,j] = self.x_next[i][j]

    # Outside complex kernel solve
    @ti.kernel
    def get_x_grad_kernel(self,pos_grad:ti.ext_arr()):
        for i in range(0,self.n_particles):
            for j in ti.static(range(self.dim)):
                pos_grad[i,j] = self.x.grad[i][j]

    @ti.complex_kernel
    def get_x_grad(self):
        pos_grad = np.zeros((self.n_particles,self.dim))
        self.get_x_grad_kernel(pos_grad)
        return pos_grad

    @ti.complex_kernel_grad(get_x_grad)
    def get_x_grad_grad(self):
        return    

    @ti.kernel
    def set_new_x_grad_kernel(self,pos_grad:ti.ext_arr()):
        for i in range(0,self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x_next.grad[i][j] += pos_grad[i,j]

    @ti.complex_kernel
    def set_new_x(self,pos_grad):
        self.set_new_x_grad_kernel(pos_grad)

    @ti.complex_kernel_grad(set_new_x)
    def set_new_x_grad(self,pos_grad):
        return

    @ti.kernel
    def clear_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0

    @ti.kernel
    def clear_particles(self):
        zero = ti.Vector.zero(self.dtype,self.dim)
        for p in range(0,self.n_particles):
            self.x[p] = zero
            self.x.grad[p] = zero
            self.x_next[p] = zero
            self.x_next.grad[p] = zero

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0
        self.loss.grad[None] = 0

    def clear(self):
        self.clear_grid()
        self.clear_particles()
        self.clear_loss()

    @ti.kernel
    def compute_grid_m_kernel(self):
        for p in range(0, self.n_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1)
                 ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_m[base + offset] += weight * self.p_mass

    # This function should be designed to be non differentiable
    @ti.kernel
    def grid_collision_kernel(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-12:
                if ti.static(self.n_primitives>0):
                    for i in ti.static(range(self.n_primitives)):
                        dist = self.primitives[i].sdf(0,I*self.dx)
                        if dist <= 0:
                            self.collision_flag[I] = i+1

    # This function will modify pos and loss in place
    @ti.kernel
    def particle_collision(self):
        for p in range(0,self.n_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            collision_flag = self.collision_flag[base]
            if collision_flag > 0:
                new_pos_offset = self.primitives[collision_flag-1].solve(self.x[p])
                self.x_next[p] = self.x[p] + new_pos_offset
                self.loss -= self.primitives[collision_flag-1].sdf(0,self.x[p])
            else:
                self.x_next[p] = self.x[p]

    # Expect pos to be an torch tensor
    @ti.complex_kernel
    def solve(self,pos):
        pos = pos.detach().cpu().numpy()
        self.clear_grid()
        self.set_x(pos)
        self.compute_grid_m_kernel()
        self.grid_collision()
        self.particle_collision()
        new_pos = np.zeros_like(pos)
        self.get_new_x(new_pos)
        return torch.from_numpy(new_pos)

    @ti.complex_kernel_grad(solve)
    def solve_grad(self,pos):
        self.particle_collision.grad()
        self.compute_grid_m_kernel.grad()

@ti.data_oriented
class ConstraintRegulator:
    def __init__(self,n_particles,primitives,iters):
        self.collider = CollisionDetector(n_particles,primitives)
        self.iters = iters
        self.loss = 0
        self.x_new = None
        self.x_solved = None
    
    def regulate(self,x,x_ref):
        self.collider.clear()
        self.x_solved = self.collider.solve(x)
        self.x_solved.requires_grad_(True)
        self.loss,assignment = compute_emd(x_ref,self.x_solved,self.iters)
        self.x_new = self.x_solved[assignment.detach().long()]
        return self.x_new
        
    def set_grad(self,x_after_grad):
        self.loss.backward(retain_graph=True)
        self.x_new.backward(x_after_grad)
        x_grad = self.x_solved.grad.numpy()
        self.collider.set_new_x_grad(x_grad)
        self.collider.enable_loss_grad()

    def get_grad(self):
        return torch.from_numpy(self.collider.get_x_grad())
