# this function will update the policy per-step...
import numpy as np
import torch
import taichi as ti
from .taichi_env import TaichiEnv
from torch.autograd import Function
import logging

@ti.data_oriented
class GradModel:
    def __init__(self, env: TaichiEnv, softness=666., init_sampler=None, output_grid=(), return_dist=True, return_deform_grad=False):
        self.env = env
        self.sim = env.simulator
        self.dim = self.sim.dim
        self.primitives = self.sim.primitives
        self.substeps = self.sim.substeps
        self.controllers = env.primitives
        self.softness = softness
        self._forward_func = None
        self.output_grid = output_grid

        self.return_dist = return_dist
        self.return_deform_grad = return_deform_grad
        self.init_state = self.env.get_state()['state']
        if init_sampler is None:
            init_sampler = lambda: self.init_state
        self.init_sampler = init_sampler

        if self.return_dist:
            #self.dists = [i.dists for i in self.primitives]
            self.dists = [ti.field(dtype=self.sim.dtype, shape=(self.sim.max_particles,), needs_grad=True) for i in self.primitives]

    def reset(self, initial_states=None, device='cuda:0', clear_grad=True):
        self.device = device
        if initial_states is None:
            initial_states = self.init_sampler()

        if clear_grad:
            self.sim.x.grad.fill(0)
            self.sim.v.grad.fill(0)
            self.sim.F.grad.fill(0)
            self.sim.C.grad.fill(0)
            for i in self.primitives:
                if i.action_dim > 0:
                    i.action_buffer.grad.fill(0)
                    i.min_dist.grad.fill(0)
                    i.dist_norm.grad.fill(0)
                i.position.grad.fill(0)
                i.rotation.grad.fill(0)
                i.v.grad.fill(0)
                i.w.grad.fill(0)

                if i.action_dim == 7:
                    i.gap.grad.fill(0)
                    i.gap_vel.grad.fill(0)

        self.env.set_state(initial_states, self.softness, False)
        return self.get_obs(0, self.device)

    @ti.kernel
    def decay_kernel(self, s: ti.int32, alpha: ti.float64):
        for i in range(self.sim.n_particles[None]):
            self.sim.x.grad[s, i] *= alpha
            self.sim.v.grad[s, i] *= alpha
            self.sim.F.grad[s, i] *= alpha
            self.sim.C.grad[s, i] *= alpha

        if ti.static(self.sim.n_primitive > 0):
            for i in ti.static(range(self.sim.n_primitive)):
                self.sim.primitives[i].position.grad[s] *= alpha
                self.sim.primitives[i].rotation.grad[s] *= alpha

    @ti.kernel
    def compute_min_dist(self, f: ti.int32):
        for j in ti.static(range(self.sim.n_primitive)):
            for i in range(self.sim.n_particles[None]):
                self.dists[j][i] = self.sim.primitives.primitives[j].sdf(f, self.sim.x[f, i])

    @ti.kernel
    def _get_obs(self, s: ti.int32, x: ti.ext_arr(), c: ti.ext_arr()):
        for i in range(self.sim.n_particles[None]):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.sim.x[s, i][j]
                x[i, j+self.dim] = self.sim.v[s, i][j]
        for idx, i in ti.static(enumerate(self.primitives)):
            if ti.static(i.action_dim>0):
                for j in ti.static(range(i.pos_dim)):
                    c[idx, j] = i.position[s][j]
                for j in ti.static(range(i.rotation_dim)):
                    c[idx, j+i.pos_dim] = i.rotation[s][j]

    @ti.kernel
    def dist2torch(self, dist: ti.ext_arr()):
        for i in range(self.sim.n_particles[None]):
            for j in ti.static(range(len(self.dists))):
                dist[i, j] = self.dists[j][i]

    @ti.kernel
    def torch2dist_grad(self, dist_grad: ti.ext_arr()):
        for i in range(self.sim.n_particles[None]):
            for j in ti.static(range(len(self.dists))):
                self.dists[j].grad[i] = dist_grad[i, j]

    def get_obs(self, s, device):
        f = s * self.sim.substeps
        x = torch.zeros(size=(self.sim.n_particles[None], self.dim * 2), device=device)
        c = torch.zeros(size=(len(self.primitives), self.primitives[0].state_dim), device=device)
        self._get_obs(f, x, c)

        if self.return_dist:
            self.compute_min_dist(f)
            dists = torch.zeros(size=(self.sim.n_particles[None], len(self.dists)), device=device)
            self.dist2torch(dists)
            x = torch.cat((x, dists), 1)

        outputs = x.clone(), c.clone()

        if len(self.output_grid) > 0:
            for i in self.output_grid:
                self.sim.grid_m.fill(0)
                self.sim.compute_grid_m_kernel(f, i)
                outputs = outputs + (self.sim.grid_m.to_torch(device),)

        if self.return_deform_grad:
            self.sim.compute_F_tmp(f)
            self.sim.svd()
            outputs = outputs + (self.sim.sig.to_torch(device=device)[:len(x)],)

        return outputs


    @ti.kernel
    def _set_obs_grad(self, s:ti.int32, x:ti.ext_arr(), c:ti.ext_arr()):
        for i in range(self.sim.n_particles[None]):
            for j in ti.static(range(self.dim)):
                self.sim.x.grad[s, i][j] += x[i, j]
                self.sim.v.grad[s, i][j] += x[i, j+self.dim]
        for idx, i in ti.static(enumerate(self.primitives)):
            if i.action_dim > 0:
                for j in ti.static(range(i.pos_dim)):
                    i.position.grad[s][j] += c[idx, j]
                for j in ti.static(range(i.rotation_dim)):
                    i.rotation.grad[s][j] += c[idx, j+i.pos_dim]

    def set_obs_grad(self, s, obs_grad, manipulator_grad, *args):
        f = s * self.sim.substeps

        if self.return_deform_grad:
            F_grad = args[-1]
            args = args[:-1]
            self.sim.compute_F_tmp(f)
            self.sim.svd()
            grad = torch.zeros((self.sim.max_particles, 3, 3), device=F_grad.device)
            grad[:len(F_grad)] = F_grad
            self.sim.U.grad.fill(0)
            self.sim.V.grad.fill(0)
            self.sim.F_tmp.grad.fill(0)
            self.sim.sig.grad.from_torch(grad)
            self.sim.svd_grad()
            self.sim.compute_F_tmp.grad(f)
            #self._set_F_grad(f, F_grad)

        if len(self.output_grid) > 0:
            for idx, i in enumerate(self.output_grid):
                self.sim.grid_m.fill(0)
                self.sim.compute_grid_m_kernel(f, i)
                self.sim.grid_m.grad.from_torch(args[idx])
                self.sim.compute_grid_m_kernel.grad(f, i)

        if self.return_dist:
            start = -len(self.dists)
            self.compute_min_dist(f)
            dist_grads = obs_grad[:, start:].clone().contiguous()
            self.torch2dist_grad(dist_grads)
            #for idx, i in enumerate(self.dists):
            #    i.grad.from_torch(obs_grad[:, start + idx].clone().contiguous())
            self.compute_min_dist.grad(f)
            obs_grad = obs_grad[..., :start].clone()

        obs_grad = obs_grad.reshape(-1, self.dim * 2)
        manipulator_grad = manipulator_grad.reshape(len(self.primitives), self.primitives[0].state_dim)
        self._set_obs_grad(f, obs_grad, manipulator_grad)

    def forward_step(self, s, a):
        #self.primitives.set_action(s, self.substeps, a)

        #action = np.asarray(action).reshape(-1).clip(-1, 1)
        a = a.reshape(-1).clamp(-1, 1)
        for i in range(len(self.primitives)):
            if self.primitives[i].action_dim > 0:
                self.primitives[i].set_action(s, self.substeps, a[self.primitives.action_dims[i]:self.primitives.action_dims[i+1]])

        for i in range(s*self.substeps, (s+1) * self.substeps):
            self.sim.substep(i)

    def backward_step(self, s):
        for i in range((s+1) * self.substeps-1, s*self.substeps-1, -1):
            self.sim.substep_grad(i)
        for i in self.primitives:
            if i.action_dim > 0:
                i.set_velocity.grad(s, self.substeps)

        grads = []
        for i in self.primitives:
            if i.action_dim > 0:
                grad = i.get_action_grad(s, 1)
                assert grad.shape[0] == 1
                grads.append(grad[0])
        return torch.tensor(np.concatenate(grads), device=self.device)

    @property
    def forward(self):
        if self._forward_func is None:
            class forward(Function):
                @staticmethod
                def forward(ctx, s, a, *past_obs):
                    ctx.save_for_backward(torch.tensor([s]), *[torch.zeros_like(i) for i in past_obs])
                    self.forward_step(s, a) # the model calculate one step forward
                    return self.get_obs(s+1, a.device) # get the observation at timestep s + 1

                @staticmethod
                def backward(ctx, *obs_grad):
                    # todo: use cur to check/force the backward orders..
                    tmp = ctx.saved_tensors
                    s = tmp[0].item()
                    self.set_obs_grad(s+1, *obs_grad) # add the gradient back into the tensors ...
                    actor_grad = self.backward_step(s)
                    return (None, actor_grad.reshape(-1)) + tmp[1:]
            self._forward_func = forward
        return self._forward_func.apply

    def make_func(self: Function, γ, λ):
        class forward(Function):
            @staticmethod
            def forward(ctx, s, a, *past_obs):
                ctx.save_for_backward(torch.tensor([s]), *[torch.zeros_like(i) for i in past_obs])
                self.forward_step(s, a)  # the model calculate one step forward
                obs = self.get_obs(s + 1, a.device)
                return obs

            @staticmethod
            def backward(ctx, *obs_grad):
                # todo: use cur to check/force the backward orders..
                tmp = ctx.saved_tensors
                s = tmp[0].item()
                self.set_obs_grad(s+1, *obs_grad)
                self.decay_kernel((s+1) * self.substeps, γ * λ)
                actor_grad = self.backward_step(s)
                return (None, actor_grad.reshape(-1)) + tmp[1:]
        return forward.apply


    def render(self, mode='human', f=0, **kwargs):
        assert f == 0
        is_copy = self.env._is_copy
        self.env._is_copy = True
        out = self.env.render(mode, **kwargs)
        self.env._is_copy = is_copy
        return out