import taichi as ti
import numpy as np
from .utils import length, qrot, qmul, w2quat
from ...config.utils import make_cls_config
from yacs.config import CfgNode as CN
from .utils import inv_trans, qrot


@ti.data_oriented
class Primitive:
    # single primitive ..
    state_dim = 7
    def __init__(self, cfg=None, dim=3, max_timesteps=1024, dtype=ti.f64, **kwargs):
        """
        The primitive has the following functions ...
        """
        self.cfg = make_cls_config(self, cfg, **kwargs)
        print('Building primitive')
        print(self.cfg)

        self.dim = dim
        self.max_timesteps = max_timesteps
        self.dtype = dtype

        self.rotation_dim = 4
        self.angular_velocity_dim = 3

        self.friction = ti.field(dtype, shape=())
        self.softness = ti.field(dtype, shape=())
        self.color = ti.Vector.field(3, ti.f32, shape=()) # positon of the primitive
        self.position = ti.Vector.field(3, dtype, needs_grad=True) # positon of the primitive
        self.rotation = ti.Vector.field(4, dtype, needs_grad=True) # quaternion for storing rotation

        self.v = ti.Vector.field(3, dtype, needs_grad=True) # velocity
        self.w = ti.Vector.field(3, dtype, needs_grad=True) # angular velocity

        ti.root.dense(ti.i, (self.max_timesteps,)).place(self.position, self.position.grad, self.rotation, self.rotation.grad,
                                                                       self.v, self.v.grad, self.w, self.w.grad)
        self.xyz_limit = ti.Vector.field(3, ti.f64, shape=(2,)) # positon of the primitive

        self.action_dim = self.cfg.action.dim

        if self.action_dim > 0:
            self.action_buffer = ti.Vector.field(self.action_dim, dtype, needs_grad=True, shape=(max_timesteps,))
            self.action_scale = ti.Vector.field(self.action_dim, dtype, shape=())
            self.min_dist = ti.field(dtype, shape=(), needs_grad=True) # record min distance to the point cloud..
            self.dist_norm = ti.field(dtype, shape=(), needs_grad=True) # record min distance to the point cloud..

    @ti.func
    def _sdf(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def _normal(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def sdf(self, f, grid_pos):
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return self._sdf(f, grid_pos)

    @ti.func
    def normal2(self, f, p):
        d = ti.cast(1e-8, ti.float64)
        n = ti.Vector.zero(self.dtype, self.dim)
        for i in ti.static(range(self.dim)):
            inc = p
            dec = p
            inc[i] += d
            dec[i] -= d

            n[i] = (0.5 / d) * (self.sdf(f, inc) - self.sdf(f, dec))
        return n/length(n)

    @ti.func
    def normal(self, f, grid_pos):
        #n2 = self.normal2(f, grid_pos)
        #xx = grid_pos
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return qrot(self.rotation[f], self._normal(f, grid_pos))

    @ti.func
    def collider_v(self, f, grid_pos, dt):
        inv_quat = ti.Vector(
            [self.rotation[f][0], -self.rotation[f][1], -self.rotation[f][2], -self.rotation[f][3]]).normalized()
        relative_pos = qrot(inv_quat, grid_pos - self.position[f])
        new_pos = qrot(self.rotation[f + 1], relative_pos) + self.position[f + 1]
        collider_v = (new_pos - grid_pos) / dt  # TODO: revise
        return collider_v

    @ti.func
    def collide(self, f, grid_pos, v_out, dt):
        dist = self.sdf(f, grid_pos)
        influence = min(ti.exp(-dist * self.softness[None]), 1)
        if (self.softness[None] > 0 and influence> 0.1) or dist <= 0:
            D = self.normal(f, grid_pos)
            collider_v_at_grid = self.collider_v(f, grid_pos, dt)

            input_v = v_out - collider_v_at_grid
            normal_component = input_v.dot(D)

            grid_v_t = input_v - min(normal_component, 0) * D

            grid_v_t_norm = length(grid_v_t)
            grid_v_t_friction = grid_v_t / grid_v_t_norm * max(0, grid_v_t_norm + normal_component * self.friction[None])
            flag = ti.cast(normal_component < 0 and ti.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30, self.dtype)
            grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
            v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence

            #print(self.position[f], f)
            #print(grid_pos, collider_v, v_out, dist, self.friction, D)
            #if v_out[1] > 1000:
            #print(input_v, collider_v_at_grid, normal_component, D)

        return v_out

    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        self.position[f+1] = max(min(self.position[f] + self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        # rotate in world coordinates about itself.
        self.rotation[f+1] = qmul(w2quat(self.w[f], self.dtype), self.rotation[f])

    # state set and copy ...
    @ti.func
    def copy_frame(self, source, target):
        self.position[target] = self.position[source]
        self.rotation[target] = self.rotation[source]

    @ti.kernel
    def get_state_kernel(self, f: ti.i32, controller: ti.ext_arr()):
        for j in ti.static(range(3)):
            controller[j] = self.position[f][j]
        for j in ti.static(range(4)):
            controller[j+self.dim] = self.rotation[f][j]

    @ti.kernel
    def set_state_kernel(self, f: ti.i32, controller: ti.ext_arr()):
        for j in ti.static(range(3)):
            self.position[f][j] = controller[j]
        for j in ti.static(range(4)):
            self.rotation[f][j] = controller[j+self.dim]

    def get_state(self, f):
        out = np.zeros((7), dtype=np.float64)
        self.get_state_kernel(f, out)
        return out

    def set_state(self, f, state):
        ss = self.get_state(f)
        ss[:len(state)] = state
        self.set_state_kernel(f, ss)

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot

    def initialize(self):
        cfg = self.cfg
        self.set_state(0, self.init_state)
        self.xyz_limit.from_numpy(np.array([cfg.lower_bound, cfg.upper_bound]))
        self.color[None] = cfg.color
        self.friction[None] = self.cfg.friction # friction coefficient
        if self.action_dim > 0:
            self.action_scale[None] = cfg.action.scale

    @ti.kernel
    def set_action_kernel(self, s: ti.i32, action: ti.ext_arr()):
        for j in ti.static(range(self.action_dim)):
            self.action_buffer[s][j] = action[j]

    @ti.complex_kernel
    def no_grad_set_action_kernel(self, s, action):
        self.set_action_kernel(s, action)
    @ti.complex_kernel_grad(no_grad_set_action_kernel)
    def no_grad_set_action_kernel_grad(self, s, action):
        return

    @ti.kernel
    def get_action_grad_kernel(self, s: ti.i32, n:ti.i32, grad: ti.ext_arr()):
        for i in range(0, n):
            for j in ti.static(range(self.action_dim)):
                grad[i, j] = self.action_buffer.grad[s+i][j]

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps:ti.i32):
        # rewrite set velocity for different
        for j in range(s*n_substeps, (s+1)*n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * self.action_scale[None][k]/n_substeps
            if ti.static(self.action_dim>3):
                for k in ti.static(range(3)):
                    self.w[j][k] = self.action_buffer[s][k+3] * self.action_scale[None][k+3]/n_substeps

    def set_action(self, s, n_substeps, action):
        # set actions for n_substeps ...
        if self.action_dim > 0:
            self.no_grad_set_action_kernel(s, action) #HACK: taichi can't compute gradient to this.
            self.set_velocity(s, n_substeps)

    def get_action_grad(self, s, n):
        if self.action_dim > 0:
            grad = np.zeros((n, self.action_dim), dtype=np.float64)
            self.get_action_grad_kernel(s, n, grad)
            return grad
        else:
            return None


    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.shape = ''
        cfg.init_pos = (0.3, 0.3, 0.3) #default color
        cfg.init_rot = (1., 0., 0., 0.) #default color
        cfg.color = (0.3, 0.3, 0.3) #default color
        cfg.lower_bound = (0., 0., 0.) #default color
        cfg.upper_bound = (1., 1., 1.) #default color
        cfg.friction = 0.9 #default color
        cfg.variations = None #TODO: not support now

        action = cfg.action = CN()
        action.dim = 0 # in this case it can't move ...
        action.scale = ()
        return cfg


