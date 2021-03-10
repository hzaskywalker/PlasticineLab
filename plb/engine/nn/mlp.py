"""
Neural network implemented with Taichi
"""
from typing import Tuple, Optional
import numpy as np
import taichi as ti
from ...config.utils import CfgNode as CN
from ..mpm_simulator import MPMSimulator
from ..primitive.primitives import Primitives, Chopsticks

@ti.data_oriented
class MLP:
    """
    A simple MLP network
    """
    def __init__(self,
                 simulator: MPMSimulator,
                 primitives: Primitives,
                 hidden_dims: Tuple[int, ...],
                 activation: Optional[str]='relu',
                 n_observed_particles=200,
                 ):
        self.simulator = simulator
        self.primitives = primitives
        #TODO: several env is not supported...
        for i in self.primitives.primitives:
            assert not isinstance(i, Chopsticks), "Chopstick is not supported now.."
        dtype = self.simulator.dtype

        self.n_observed_particles = n_observed_particles
        n_particle = self.simulator.n_particles
        self.obs_step = (n_particle// self.n_observed_particles)
        self.obs_num = n_particle//self.obs_step
        inp_dim = self.obs_num * 6 + primitives.state_dim

        self.substeps = self.simulator.substeps

        max_steps = self.simulator.max_steps // self.substeps + 2
        dims = (inp_dim,) + hidden_dims + (primitives.action_dim,)
        print("MLP dims:", dims)

        self.dims = dims
        self.n_layer = len(dims) - 1
        self.velocity_weight = ti.field(dtype, shape=(), needs_grad=False)  # record min distance to the point cloud..
        self.hidden = [ti.field(dtype=dtype, shape=(max_steps, dims[0]), needs_grad=True)]
        self.hidden_prev = [None]
        self.W = []
        self.b = []
        for i in range(self.n_layer):
            self.W.append(ti.field(dtype=dtype, shape=(dims[i+1], dims[i]), needs_grad=True))
            self.b.append(ti.field(dtype=dtype, shape=(dims[i+1],), needs_grad=True))
            self.hidden_prev.append(ti.field(dtype=dtype, shape=(max_steps, dims[i+1]), needs_grad=True))
            self.hidden.append(ti.field(dtype=dtype, shape=(max_steps, dims[i+1]), needs_grad=True))

        self.max_timesteps = max_steps
        self.kernels = self.input_kernels()

        for i in range(self.n_layer):
            self.kernels += self.create_forward_kernels(i, activation
                    if i!=self.n_layer-1 else None)
        self.kernels += self.output_kernels()

    def input_kernels(self):
        h = self.hidden[0]
        x = self.simulator.x
        v = self.simulator.v

        @ti.kernel
        def input_particles(t: ti.i32):
            for i in range(self.obs_num):
                for j in ti.static(range(3)):
                    h[t, i*6+j] = x[t*self.substeps, i * self.obs_step][j]
                for j in ti.static(range(3)):
                    h[t, i*6+j+3] = v[t*self.substeps, i * self.obs_step][j] * self.velocity_weight[None]

        base = self.obs_num * 6

        @ti.kernel
        def input_primitives(t: ti.i32):
            for i in ti.static(range(len(self.primitives))):
                for j in ti.static(range(3)):
                    h[t, base + i*7 + j] = \
                        self.primitives[i].position[t*self.substeps][j]
                for j in ti.static(range(4)):
                    h[t, base + i*7 + 3 + j] =\
                        self.primitives[i].rotation[t*self.substeps][j]
        return [input_particles, input_primitives]

    def output_kernels(self):
        h = self.hidden[-1]
        @ti.kernel
        def set_action(t: ti.i32):
            cur = 0
            for i in ti.static(range(len(self.primitives))):
                p = self.primitives[i]
                if ti.static(p.action_dim>0):
                    for j in ti.static(range(p.action_dim)):
                        p.action_buffer[t][j] = ti.max(ti.min(h[t, cur+j], 1.), -1.)
                    cur += p.action_dim
        return [set_action]

    def create_forward_kernels(self, layer, activation=None):
        W = self.W[layer]
        b = self.b[layer]
        h0 = self.hidden[layer]
        h1_prev = self.hidden_prev[layer+1]
        h1 = self.hidden[layer+1]
        d0 = self.dims[layer]
        d1 = self.dims[layer+1]

        @ti.kernel
        def weights(t: ti.i32):
            for i in range(d1):
                for j in range(d0):
                    h1_prev[t, i] += W[i, j] * h0[t, j]

        @ti.kernel
        def bias(t: ti.i32):
            for i in range(d1):
                act = h1_prev[t, i] + b[i]
                if ti.static(activation == 'relu'):
                    act = ti.max(act, 0.)
                if ti.static(activation == 'tanh'):
                    act = ti.tanh(act)
                h1[t, i] = act

        return [weights, bias]


    @ti.kernel
    def clear_kernel(self, t: ti.i32):
        for i in ti.static(range(self.n_layer)):
            for j in range(self.dims[i+1]):
                self.hidden_prev[i+1][t, j] = 0.

    @ti.complex_kernel
    def clear_no_grad(self, t):
        self.clear_kernel(t)
    @ti.complex_kernel_grad(clear_no_grad)
    def clear_no_grad_grad(self, t):
        pass

    def set_action(self, s, n_substeps):
        # step 1, compute the results into the buffer
        # step 2, store the results
        self.clear_no_grad(s)
        assert n_substeps == self.substeps
        for i in self.kernels:
            i(s)
        for i in self.primitives:
            if i.action_dim > 0:
                i.set_velocity(s, n_substeps)

    def get_grad(self):
        outs = []
        for i in range(self.n_layer):
            outs+=[self.W[i].grad.to_numpy().reshape(-1),
                        self.b[i].grad.to_numpy().reshape(-1)]
        return np.concatenate(outs)

    def get_params(self):
        outs = []
        for i in range(self.n_layer):
            outs += [self.W[i].to_numpy().reshape(-1),
                   self.b[i].to_numpy().reshape(-1)]
        return np.concatenate(outs)

    def set_params(self, param):
        for i in range(self.n_layer):
            shape = (self.dims[i+1], self.dims[i])
            n = shape[0] * shape[1]
            self.W[i].from_numpy(param[:n].reshape(shape))
            param = param[n:]

            n = self.dims[i+1]
            self.b[i].from_numpy(param[:n].reshape(n))
            param = param[n:]
        if len(param) == 1:
            self.velocity_weight[None] = param[-1]
        else:
            self.velocity_weight[None] = 1.
            assert len(param) == 0

