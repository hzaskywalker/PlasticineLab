import taichi as ti
import numpy as np

@ti.data_oriented
class MPMSimulator:
    def __init__(self, cfg, primitives=()):
        dim = self.dim = cfg.dim
        assert cfg.dtype == 'float64'
        dtype = self.dtype = ti.f64 #if cfg.dtype == 'float64' else ti.f64
        self._yield_stress = cfg.yield_stress
        self.ground_friction = cfg.ground_friction
        self.default_gravity = cfg.gravity
        self.n_primitive = len(primitives)

        quality = cfg.quality
        if self.dim == 3:
            quality = quality * 0.5
        n_particles = self.n_particles = cfg.n_particles
        n_grid = self.n_grid = int(128 * quality)

        self.dx, self.inv_dx = 1 / n_grid, float(n_grid)
        self.dt = 0.5e-4 / quality
        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        self.p_mass = self.p_vol * self.p_rho

        # material
        E, nu = cfg.E, cfg.nu
        self._mu, self._lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
        self.mu = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)
        self.lam = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)
        self.yield_stress = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)

        max_steps = self.max_steps = cfg.max_steps
        self.substeps = int(2e-3 // self.dt)
        self.x = ti.Vector.field(dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # position
        self.v = ti.Vector.field(dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # velocity
        self.C = ti.Matrix.field(dim, dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # affine velocity field
        self.F = ti.Matrix.field(dim, dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # deformation gradient

        self.F_tmp = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles), needs_grad=True)  # deformation gradient
        self.U = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)
        self.V = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)
        self.sig = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)

        self.res = res = (n_grid, n_grid) if dim == 2 else (n_grid, n_grid, n_grid)
        self.grid_v_in = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=dtype, shape=res, needs_grad=True)  # grid node mass
        self.grid_v_out = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity

        self.gravity = ti.Vector.field(dim, dtype=dtype, shape=()) # gravity ...
        self.primitives = primitives

    def initialize(self):
        self.gravity[None] = self.default_gravity
        self.yield_stress.fill(self._yield_stress)
        self.mu.fill(self._mu)
        self.lam.fill(self._lam)

    # --------------------------------- MPM part -----------------------------------
    @ti.kernel
    def clear_grid(self):
        zero = ti.Vector.zero(self.dtype, self.dim)
        for I in ti.grouped(self.grid_m):
            self.grid_v_in[I] = zero
            self.grid_v_out[I] = zero
            self.grid_m[I] = 0

            self.grid_v_in.grad[I] = zero
            self.grid_v_out.grad[I] = zero
            self.grid_m.grad[I] = 0

    @ti.kernel
    def clear_SVD_grad(self):
        zero = ti.Matrix.zero(self.dtype, self.dim, self.dim)
        for i in range(0, self.n_particles):
            self.U.grad[i] = zero
            self.sig.grad[i] = zero
            self.V.grad[i] = zero
            self.F_tmp.grad[i] = zero


    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for p in range(0, self.n_particles):  # Particle state update and scatter to grid (P2G)
            self.F_tmp[p] = (ti.Matrix.identity(self.dtype, self.dim) + self.dt * self.C[f, p]) @ self.F[f, p]

    @ti.kernel
    def svd(self):
        for p in range(0, self.n_particles):
            self.U[p], self.sig[p], self.V[p] = ti.svd(self.F_tmp[p])

    @ti.kernel
    def svd_grad(self):
        for p in range(0, self.n_particles):
            self.F_tmp.grad[p] += self.backward_svd(self.U.grad[p], self.sig.grad[p], self.V.grad[p], self.U[p], self.sig[p], self.V[p])

    @ti.func
    def backward_svd(self, gu, gsigma, gv, u, sig, v):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = v.transpose()
        ut = u.transpose()
        sigma_term = u @ gsigma @ vt

        s = ti.Vector.zero(self.dtype, self.dim)
        if ti.static(self.dim==2):
            s = ti.Vector([sig[0, 0], sig[1, 1]]) ** 2
        else:
            s = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]) ** 2
        F = ti.Matrix.zero(self.dtype, self.dim, self.dim)
        for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
            if i == j: F[i, j] = 0
            else: F[i, j] = 1./self.clamp(s[j] - s[i])
        u_term = u @ ((F * (ut@gu - gu.transpose()@u)) @ sig) @ vt
        v_term = u @ (sig @ ((F * (vt@gv - gv.transpose()@v)) @ vt))
        return u_term + v_term + sigma_term

    @ti.func
    def make_matrix_from_diag(self, d):
        if ti.static(self.dim==2):
            return ti.Matrix([[d[0], 0.0], [0.0, d[1]]], dt=self.dtype)
        else:
            return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=self.dtype)

    @ti.func
    def compute_von_mises(self, F, U, sig, V, yield_stress, mu):
        #epsilon = ti.Vector([0., 0., 0.], dt=self.dtype)
        epsilon = ti.Vector.zero(self.dtype, self.dim)
        sig = ti.max(sig, 0.05) # add this to prevent NaN in extrem cases
        if ti.static(self.dim == 2):
            epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1])])
        else:
            epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
        epsilon_hat = epsilon - (epsilon.sum() / self.dim)
        epsilon_hat_norm = self.norm(epsilon_hat)
        delta_gamma = epsilon_hat_norm - yield_stress / (2 * mu)

        if delta_gamma > 0:  # Yields
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            sig = self.make_matrix_from_diag(ti.exp(epsilon))
            F = U @ sig @ V.transpose()
        return F

    @ti.func
    def clamp(self, a):
        # remember that we don't support if return in taichi
        # stop the gradient ...
        if a>=0:
            a = max(a, 1e-6)
        else:
            a = min(a, -1e-6)
        return a

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(0, self.n_particles):
            base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[f, p] * self.inv_dx - base.cast(self.dtype)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_F = self.compute_von_mises(self.F_tmp[p], self.U[p], self.sig[p], self.V[p], self.yield_stress[p], self.mu[p])
            self.F[f + 1, p] = new_F

            J = (new_F).determinant()

            r = self.U[p] @ self.V[p].transpose()
            stress = 2 * self.mu[p] * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.identity(self.dtype, self.dim) * self.lam[p] * J * (J - 1)

            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass * self.C[f, p]
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(self.dtype) - fx) * self.dx
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                x = base + offset

                self.grid_v_in[base + offset] += weight * (self.p_mass * self.v[f, p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.kernel
    def grid_op(self, f: ti.i32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-12:  # No need for epsilon here, 1e-10 is to prevent potential numerical problems ..
                v_out = (1 / self.grid_m[I]) * self.grid_v_in[I]  # Momentum to velocity
                v_out += self.dt * self.gravity[None] * 30  # gravity

                if ti.static(self.n_primitive>0):
                    for i in ti.static(range(self.n_primitive)):
                        v_out = self.primitives[i].collide(f, I * self.dx, v_out, self.dt)

                bound = 3
                v_in2 = v_out
                for d in ti.static(range(self.dim)):
                    if I[d] < bound and v_out[d] < 0:
                        if ti.static(d != 1 or self.ground_friction == 0):
                            v_out[d] = 0  # Boundary conditions
                        else:
                            if ti.static(self.ground_friction < 10):
                                # TODO: 1e-30 problems ...
                                normal = ti.Vector.zero(self.dtype, self.dim)
                                normal[d] = 1.
                                lin = v_out.dot(normal) + 1e-30
                                vit = v_out - lin * normal - I * 1e-30
                                lit = self.norm(vit)
                                v_out = max(1. + ti.static(self.ground_friction) * lin / lit, 0.) * (vit + I * 1e-30)
                                v_out[1] = 0
                            else:
                                v_out = ti.Vector.zero(self.dtype, self.dim)

                    if I[d] > self.n_grid - bound and v_out[d] > 0: v_out[d] = 0

                self.grid_v_out[I] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        for p in range(0, self.n_particles):  # grid to particle (G2P)
            base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[f, p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(self.dtype, self.dim)
            new_C = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(self.dtype) - fx
                g_v = self.grid_v_out[base + offset]
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v[f + 1, p], self.C[f + 1, p] = new_v, new_C

            self.x[f + 1, p] = ti.max(ti.min(self.x[f, p] + self.dt * self.v[f + 1, p], 1.-3*self.dx), 0.)
            # advection and preventing it from overflow

    @ti.complex_kernel
    def substep(self, s):
        # centroids[None] = [0, 0] # manually clear the centroids...
        self.clear_grid()
        self.compute_F_tmp(s)
        self.svd()
        self.p2g(s)

        for i in range(self.n_primitive):
            self.primitives[i].forward_kinematics(s)

        self.grid_op(s)
        self.g2p(s)


    @ti.complex_kernel_grad(substep)
    def substep_grad(self, s):
        self.clear_grid()
        self.clear_SVD_grad()  # clear the svd grid

        self.compute_F_tmp(s)  # we need to compute it for calculating the svd decomposition
        self.svd()
        self.p2g(s)
        self.grid_op(s)

        self.g2p.grad(s)
        self.grid_op.grad(s)

        for i in range(self.n_primitive-1, -1, -1):
            self.primitives[i].forward_kinematics.grad(s)

        self.p2g.grad(s)
        self.svd_grad()
        self.compute_F_tmp.grad(s)


    # ------------------------------------ io -------------------------------------#
    @ti.kernel
    def readframe(self, f:ti.i32, x: ti.ext_arr(), v: ti.ext_arr(), F: ti.ext_arr(), C: ti.ext_arr()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.x[f, i][j]
                v[i, j] = self.v[f, i][j]
                for k in ti.static(range(self.dim)):
                    F[i, j, k] = self.F[f, i][j, k]
                    C[i, j, k] = self.C[f, i][j, k]

    @ti.kernel
    def readframe_grad(self,f:ti.i32, x_grad: ti.ext_arr(), v_grad: ti.ext_arr()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x_grad[i,j] = self.x.grad[f,i][j]
                v_grad[i,j] = self.v.grad[f,i][j]

    @ti.kernel
    def setframe(self, f:ti.i32, x: ti.ext_arr(), v: ti.ext_arr(), F: ti.ext_arr(), C: ti.ext_arr()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x[f, i][j] = x[i, j]
                self.v[f, i][j] = v[i, j]
                for k in ti.static(range(self.dim)):
                    self.F[f, i][j, k] = F[i, j, k]
                    self.C[f, i][j, k] = C[i, j, k]

    @ti.kernel
    def copyframe(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_particles):
            self.x[target, i] = self.x[source, i]
            self.v[target, i] = self.v[source, i]
            self.F[target, i] = self.F[source, i]
            self.C[target, i] = self.C[source, i]

        if ti.static(self.n_primitive>0):
            for i in ti.static(range(self.n_primitive)):
                self.primitives[i].copy_frame(source, target)

    def get_state(self, f):
        x = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        v = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        F = np.zeros((self.n_particles, self.dim, self.dim), dtype=np.float64)
        C = np.zeros((self.n_particles, self.dim, self.dim), dtype=np.float64)
        self.readframe(f, x, v, F, C)
        out = [x, v, F, C]
        for i in self.primitives:
            out.append(i.get_state(f))
        return out

    def get_current_state(self):
        return self.get_state(self.cur)

    def get_state_grad(self,f):
        x_grad = np.zeros((self.n_particles,self.dim), dtype = np.float64)
        v_grad = np.zeros((self.n_particles,self.dim), dtype = np.float64)
        primitive_pos_grad = np.zeros((self.n_primitive,self.dim), dtype = np.float64)
        primitive_rot_grad = np.zeros((self.n_primitive,self.dim+1), dtype = np.float64)
        self.readframe_grad(f,x_grad,v_grad)
        for i,primitive in enumerate(self.primitives):
            primitive_pos_grad[i] = primitive.get_pos_grad()
            primitive_rot_grad[i] = primitive.get_rot_grad()
        return x_grad, v_grad, primitive_pos_grad, primitive_rot_grad

    def set_state(self, f, state):
        self.setframe(f, *state[:4])
        for s, i in zip(state[4:], self.primitives):
            i.set_state(f, s)

    @ti.kernel
    def reset_kernel(self, x:ti.ext_arr()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x[0, i][j] = x[i, j]
            self.v[0, i] = ti.Vector.zero(self.dtype, self.dim)
            self.F[0, i] = ti.Matrix.identity(self.dtype, self.dim) #ti.Matrix([[1, 0], [0, 1]])
            self.C[0, i] = ti.Matrix.zero(self.dtype, self.dim, self.dim)

    def reset(self, x):
        self.reset_kernel(x)
        self.cur = 0

    @ti.kernel
    def get_x_kernel(self, f: ti.i32, x: ti.ext_arr()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.x[f, i][j]

    @ti.complex_kernel
    def get_x_tape(self,f:ti.i32, x: ti.ext_arr()):
        self.get_x_kernel(f,x)

    @ti.complex_kernel_grad(get_x_tape)
    def get_x_tape_grad(self,f: ti.i32,x:ti.ext_arr()):
        return

    def get_x(self, f):
        x = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.get_x_tape(f, x)
        return x

    @ti.kernel
    def get_x_grad_kernel(self,f:ti.i32, x_grad: ti.ext_arr()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x_grad[i,j] = self.x.grad[f,i][j]

    @ti.complex_kernel
    def get_x_grad_tape(self,f:ti.i32,x_grad:ti.ext_arr()):
        self.get_x_grad_kernel(f,x_grad)

    @ti.complex_kernel_grad(get_x_grad_tape)
    def get_x_grad_tape(self,f:ti.i32, x_grad:ti.ext_arr()):
        return

    def get_x_grad(self,f):
        x_grad = np.zeros((self.n_particles,self.dim), dtype=np.float64)
        self.get_x_grad_tape(f,x_grad)

    @ti.kernel
    def get_v_kernel(self, f: ti.i32, v: ti.ext_arr()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                v[i, j] = self.v[f, i][j]

    @ti.complex_kernel
    def get_v_tape(self,f:ti.i32,v:ti.ext_arr()):
        self.get_v_kernel(f,v)

    @ti.complex_kernel_grad(get_v_tape)
    def get_v_tape_grad(self,f:ti.i32,v:ti.ext_arr()):
        return

    def get_v(self, f):
        v = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.get_v_tape(f, v)
        return v

    @ti.kernel
    def set_x_grad_kernel(self,f:ti.i32,x_grad: ti.ext_arr()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.x.grad[f,i][j] = x_grad[i,j]

    @ti.complex_kernel
    def set_x_grad_tape(self,f:ti.i32,x_grad:ti.ext_arr()):
        self.set_x_grad_kernel(f,x_grad)

    @ti.complex_kernel_grad(set_x_grad_tape)
    def set_x_grad_tape_grad(self,f:ti.i32,x_grad:ti.ext_arr()):
        return

    def set_x_grad(self,f,x_grad):
        self.set_x_grad_tape(f,x_grad)

    def step(self, is_copy, action=None):
        start = 0 if is_copy else self.cur
        self.cur = start + self.substeps

        if action is not None:
            self.primitives.set_action(start//self.substeps, self.substeps, action)

        for s in range(start, self.cur):
            self.substep(s)
        if is_copy:
            self.copyframe(self.cur, 0) # copy to the first frame for simulation
            self.cur = 0


    # ------------------------------------------------------------------
    # for loss computation
    # ------------------------------------------------------------------
    @ti.kernel
    def compute_grid_m_kernel(self, f:ti.i32):
        for p in range(0, self.n_particles):
            base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[f, p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(self.stencil_range())):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_m[base + offset] += weight * self.p_mass

    def get_v_nokernel(self):
        v = np.zeros((self.n_particles,self.dim),dype=np.float64)
        for i in range(self.n_particles):
            for j in range(self.dim):
                v[i,j] = self.v[1,i][j]
        return v

    """
    @ti.complex_kernel
    def clear_and_compute_grid_m(self, f):
        self.grid_m.fill(0)
        self.compute_grid_m_kernel(f)

    @ti.complex_kernel_grad(clear_and_compute_grid_m)
    def clear_and_compute_grid_m_grad(self, f):
        self.compute_grid_m_kernel.grad(f)
    """