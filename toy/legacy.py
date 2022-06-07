import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)  # Try to run on GPU. Use arch=ti.opengl on old GPUs
quality = 1  # Use a larger value for higher-res simulations
dim = 3
N = 13
substeps = 3
n_particles, n_grid = 3 * 13 ** dim * quality ** dim, 32 * quality
collider_id1 = n_particles
collider_id2 = n_particles + 1
collider_radius = n_particles + 2
gravity_id = n_particles + 3
print(n_particles)
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 8e-4 / quality
print(dt * substeps)
p_vol, p_rho = (dx * 0.5) ** dim, 1
p_mass = p_vol * p_rho
E, nu = 1e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
x = ti.Vector(dim, dt=ti.f32, shape=n_particles)  # position
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)  # velocity
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)  # affine velocity field
F = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)  # deformation gradient
material = ti.var(dt=ti.i32, shape=n_particles)  # material id
Jp = ti.var(dt=ti.f32, shape=n_particles)  # plastic deformation
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid, n_grid))  # grid node mass
current_t = ti.var(dt=ti.i32, shape=())


@ti.func
def sqr(x):
    return x ** 2


@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * sqr(1.5 - fx), 0.75 - sqr(fx - 1), 0.5 * sqr(fx - 0.5)]
        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]  # deformation gradient update
        h = ti.exp(10 * (1.0 - Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        h = max(0.1, min(5, h))
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(dim)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 3.5e-2), 1 + 6.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:  # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(ti.f32, dim)
            F[p][0, 0] = J
        elif material[p] == 2:
            F[p] = U @ sig @ V.T()  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.T()) @ F[p].T() + ti.Matrix.identity(ti.f32, dim) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:  # No need for epsilon here
            grid_v[I] = (1 / grid_m[I]) * grid_v[I]  # Momentum to velocity
            grid_v[I] += dt * x[gravity_id]

            collider_pos1 = x[collider_id1]
            collider_pos2 = x[collider_id2]
            collider_pos = collider_pos1 + (collider_pos2 - collider_pos1) * (current_t[None] * (1 / substeps))
            collider_v = (collider_pos2 - collider_pos1) * (1 / (substeps * dt))
            grid_pos = I * dx
            dd = grid_pos - collider_pos
            D = ti.normalized(dd)

            if dd.norm() < x[collider_radius][0]:
                collider_v_at_grid = ti.dot(D, collider_v) * D
                input_v = grid_v[I] - collider_v_at_grid
                grid_v_t = input_v - min(ti.dot(input_v, D), 0) * D
                grid_v[I] = grid_v_t + collider_v_at_grid

            for d in ti.static(range(3)):
                if I[d] < 3 and grid_v[I][d] < 0:          grid_v[I][d] = 0  # Boundary conditions
                if I[d] > n_grid - 3 and grid_v[I][d] > 0: grid_v[I][d] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * sqr(1.5 - fx), 0.75 - sqr(fx - 1.0), 0.5 * sqr(fx - 0.5)]
        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * ti.outer_product(g_v, dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection
        if p == 0:
            current_t[None] = (current_t[None] + 1) % substeps


@ti.func
def sin_rand(i):
    s = ti.sin(i * 12.9898) * 43758.5453123
    s -= ti.floor(s)
    return s


group_size = n_particles // 3


@ti.kernel
def initialize():
    for i in range(n_particles):
        # p = i // N ** 2 % N / N
        # q = (i // N % N / N)
        # r = (i % N / N)
        p = sin_rand(i * 3)
        q = sin_rand(i * 3 + 1)
        r = sin_rand(i * 3 + 2)
        x[i] = [p * 0.2 + 0.3 + 0.1 * (i // group_size), q * 0.2 + 0.05 + 0.32 * (i // group_size),
                r * 0.2 + 0.1 + 0.22 * (i // group_size)]
        material[i] = i // group_size * 2 % 3  # 0: fluid 1: jelly 2: snow
        v[i] = ti.Vector.zero(dt=ti.f32, n=dim)
        F[i] = ti.Matrix.identity(dt=ti.f32, n=dim)
        Jp[i] = 1


initialize()
substep()
gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)

import tqdm
for frame in tqdm.trange(20000):

    def pos(f):
        t = f * 0.05
        return [0.5 + 0.3 * math.cos(t), 0.1, 0.5 + 0.3 * math.sin(t)]


    x[collider_id1] = pos(frame)
    x[collider_id2] = pos(frame + 1)
    x[collider_radius][0] = 0.1
    x[gravity_id] = [0, -50, 0]

    for s in range(substeps):
        substep()
    #colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
    #pos = x.to_numpy()[:, :2]
    #gui.circles(pos, radius=1.5, color=colors[material.to_numpy()])
    #gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk