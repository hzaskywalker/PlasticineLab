import taichi as ti
import numpy as np
import yaml
from .primitive_base import Primitive
from yacs.config import CfgNode as CN
from .utils import qrot3d, qmul3d, w2quat3d

@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-30)

@ti.func
def normalize(n):
    return n/length(n)


class Sphere(Primitive):
    def __init__(self, **kwargs):
        super(Sphere, self).__init__(**kwargs)
        self.radius = self.cfg.radius

    @ti.func
    def sdf(self, f, grid_pos):
        return length(grid_pos-self.position[f]) - self.radius

    @ti.func
    def normal(self, f, grid_pos):
        return normalize(grid_pos-self.position[f])

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.radius = 1.
        return cfg

    def get_corners(self):
        phi = np.arange(20)/21 * np.pi * 2
        return np.stack((np.cos(phi), np.sin(phi)), axis=1) * self.radius

class Capsule(Primitive):
    def __init__(self, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.h = self.cfg.h
        self.r = self.cfg.r
        assert self.dim == 3

    @ti.func
    def _sdf(self, f, grid_pos):
        p2 = grid_pos
        p2[1] += self.h / 2
        p2[1] -= min(max(p2[1], 0.0), self.h)
        return length(p2) - self.r

    @ti.func
    def _normal(self, f, grid_pos):
        p2 = grid_pos
        p2[1] += self.h / 2
        p2[1] -= min(max(p2[1], 0.0), self.h)
        return normalize(p2)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.06
        cfg.r = 0.03
        return cfg


class RollingPin(Capsule):
    # rollingpin's capsule...
    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        vel = self.v[f]
        dw = vel[0]  # rotate about object y
        dth = vel[1]  # rotate about the world w
        dy = vel[2]  # decrease in y coord...
        y_dir = qrot3d(self.rotation[f], ti.Vector([0., -1., 0.]))
        print(y_dir)
        x_dir = ti.Vector([0., 1., 0.]).cross(y_dir) * dw * 0.03  # move toward x, R=0.03 is hand crafted...
        x_dir[1] = dy  # direction
        self.rotation[f+1] = qmul3d(
            w2quat3d(ti.Vector([0., -dth, 0.]), self.dtype),
            qmul3d(self.rotation[f], w2quat3d(ti.Vector([0., dw, 0.]), self.dtype))
        )
        #print(self.rotation[f+1], self.rotation[f+1].dot(self.rotation[f+1]))
        self.position[f+1] = max(min(self.position[f] + x_dir, self.xyz_limit[1]), self.xyz_limit[0])


class Chopsticks(Capsule):
    state_dim = 8
    def __init__(self, **kwargs):
        super(Chopsticks, self).__init__(**kwargs)
        self.gap = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.gap_vel = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.h = self.cfg.h
        self.r = self.cfg.r
        self.minimal_gap = self.cfg.minimal_gap
        assert self.action_dim == 7 # 3 linear, 3 angle, 1 for grasp ..

        raise NotImplementedError("This is not suitable now..")

    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        self.gap[f+1] = max(self.gap[f] - self.gap_vel[f], self.minimal_gap)
        self.position[f+1] = max(min(self.position[f] + self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        self.rotation[f+1] = qmul3d(self.rotation[f], w2quat3d(self.w[f], self.dtype))
        #print(self.rotation[f+1])

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps:ti.i32):
        # rewrite set velocity for different
        for j in range(s*n_substeps, (s+1)*n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * self.action_scale[None][k]/n_substeps
            for k in ti.static(range(3)):
                self.w[j][k] = self.action_buffer[s][k+3] * self.action_scale[None][k+3]/n_substeps
            self.gap_vel[j] = self.action_buffer[s][6] * self.action_scale[None][6]/n_substeps

    @ti.func
    def _sdf(self, f, grid_pos):
        delta = ti.Vector([self.gap[f] / 2, 0., 0.])
        p = grid_pos - ti.Vector([0., -self.h/2, 0.])
        a = super(Chopsticks, self)._sdf(f, p-delta) # grid_pos - (mid + delta)
        b = super(Chopsticks, self)._sdf(f, p+delta) # grid_pos - (mid - delta)
        return ti.min(a, b)

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot + (self.cfg.init_gap,)

    def get_state(self, f):
        return np.append(super(Chopsticks, self).get_state(f), self.gap[f])

    @ti.func
    def copy_frame(self, source, target):
        super(Chopsticks, self).copy_frame(source, target)
        self.gap[target] = self.gap[source]

    def set_state(self, f, state):
        assert len(state) == 8
        super(Chopsticks, self).set_state(f, state[:7])
        self.gap[f] = state[7]

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.06
        cfg.r = 0.03
        cfg.minimal_gap = 0.02
        cfg.init_gap = 0.06
        return cfg


class Cylinder(Primitive):
    def __init__(self, **kwargs):
        super(Cylinder, self).__init__(**kwargs)
        self.h = self.cfg.h
        self.r = self.cfg.r
        assert self.dim == 3

    @ti.func
    def _sdf(self, f, grid_pos):
        # convert it to a 2D box .. and then call the sdf of the 2d box..
        d = ti.abs(ti.Vector([length(ti.Vector([grid_pos[0], grid_pos[2]])), grid_pos[1]])) - ti.Vector([self.h, self.r])
        return min(max(d[0], d[1]), 0.0) + length(max(d, 0.0)) # if max(d, 0) < 0 or if max(d, 0) > 0

    @ti.func
    def _normal(self, f, grid_pos):
        p = ti.Vector([grid_pos[0], grid_pos[2]])
        l = length(p)
        d = ti.Vector([l, ti.abs(grid_pos[1])]) - ti.Vector([self.h, self.r])

        # if max(d) > 0, normal direction is just d
        # other wise it's 1 if d[1]>d[0] else -d0
        # return min(max(d[0], d[1]), 0.0) + length(max(d, 0.0))
        f = ti.cast(d[0] > d[1], self.dtype)
        n2 = max(d, 0.0) + ti.cast(max(d[0], d[1]) <= 0., self.dtype) * ti.Vector([f, 1-f]) # normal should be always outside ..
        n2_ = n2/length(n2)
        p2 = p/l
        n3 = ti.Vector([p2[0] * n2_[0], n2_[1] * (ti.cast(grid_pos[1]>=0, self.dtype) * 2 - 1), p2[1] * n2_[0]])
        return normalize(n3)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.2
        cfg.r = 0.1
        return cfg


class Torus(Primitive):
    def __init__(self, **kwargs):
        super(Torus, self).__init__(**kwargs)
        self.tx = self.cfg.tx
        self.ty = self.cfg.ty
        assert self.dim == 3

    @ti.func
    def _sdf(self, f, grid_pos):
        q = ti.Vector([length(ti.Vector([grid_pos[0], grid_pos[2]])) - self.tx, grid_pos[1]])
        return length(q) - self.ty

    @ti.func
    def _normal(self, f, grid_pos):
        x = ti.Vector([grid_pos[0], grid_pos[2]])
        l = length(x)
        q = ti.Vector([length(x) - self.tx, grid_pos[1]])

        n2 = q/length(q)
        x2 = x/l
        n3 = ti.Vector([x2[0] * n2[0], n2[1], x2[1] * n2[0]])
        return normalize(n3)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.tx = 0.2
        cfg.ty = 0.1
        return cfg


class Box(Primitive):
    def __init__(self, **kwargs):
        super(Box, self).__init__(**kwargs)
        self.size = ti.Vector.field(self.dim, self.dtype, shape=())
        self.round = ti.field(self.dtype, shape=())

    def initialize(self):
        super(Box, self).initialize()
        self.size[None] = self.cfg.size
        self.round[None] = self.cfg.round

    def make_inertia(self):
        # https://github.com/hzaskywalker/Robotics/blob/master/robot/torch_robotics/objects/rigid_body.py
        w, d, h = self.cfg.size
        m = self.cfg.mass
        tensor = np.zeros((6, 6), dtype=np.float32)
        tensor[[3,4,5], [3,4,5]] = m
        tensor[0, 0] = 1/12 * (d**2 + h**2) * m
        tensor[1, 1] = 1/12 * (w**2 + h**2) * m
        tensor[2, 2] = 1/12 * (w**2 + d**2) * m
        #self.inertia.from_numpy(tensor)
        self.inv_inertia.from_numpy(np.linalg.inv(tensor))

    @ti.func
    def _sdf(self, f, grid_pos):
        # p: vec3,b: vec3
        q = ti.abs(grid_pos) - self.size[None]
        out = length(max(q, 0.0))
        if ti.static(self.dim == 3):
            out += min(max(q[0], max(q[1], q[2])), 0.0)
        else:
            out += min(max(q[0], q[1]), 0.0)
        return out - self.round[None]

    @ti.func
    def _normal(self, f, grid_pos):
        q = ti.abs(grid_pos) - self.size[None]
        inside = ti.cast(0., self.dtype)
        if ti.static(self.dim == 2):
            inside = max(q[0], q[1])
        else:
            inside = max(q[0], max(q[1], q[2]))

        f = ti.cast(inside <= 0, self.dtype)

        normal = ti.cast(q == inside, self.dtype) * f

        #yy = ti.cast(q == inside, self.dtype)
        #print('q', q)
        #print('xx', inside, normal, yy)

        q2 = max(q, 0.0)
        normal += q2/length(q2) * (1-f)

        mask = (2. * ti.cast(grid_pos > 0, self.dtype) - 1.)
        #print('yy', q2, normal)
        #print('mask', mask)

        normal = normal * mask
        #return normalize(normal)
        return normal/ti.sqrt(normal.dot(normal) + 1e-30)


    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.size = (0.1, 0.1, 0.1)
        cfg.round = 0.
        return cfg

    def get_corners(self):
        assert self.dim == 2
        size = self.cfg.size
        return np.array([
            [-size[0], -size[1]],
            [-size[0], size[1]],
            [size[0], size[1]],
            [size[0], -size[1]],
        ])


class Gripper(Box):
    state_dim = 8
    def __init__(self, **kwargs):
        super(Gripper, self).__init__(**kwargs)
        self.gap = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.gap_vel = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.size = ti.Vector.field(3, self.dtype, shape=())
        self.size[None] = self.cfg.size
        self.minimal_gap = self.cfg.minimal_gap

    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        self.gap[f+1] = max(self.gap[f] - self.gap_vel[f], self.minimal_gap)
        self.position[f+1] = max(min(self.position[f] + self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        self.rotation[f+1] = qmul3d(self.rotation[f], w2quat3d(self.w[f], self.dtype))

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps:ti.i32):
        for j in range(s*n_substeps, (s+1)*n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * self.action_scale[None][k]/n_substeps
            for k in ti.static(range(3)):
                self.w[j][k] = self.action_buffer[s][k+3] * self.action_scale[None][k+3]/n_substeps
            self.gap_vel[j] = self.action_buffer[s][6] * self.action_scale[None][6]/n_substeps

    @ti.func
    def get_pos(self, f, flag):
        return self.position[f] + qrot3d(self.rotation[f], ti.Vector([self.gap[f] / 2 * flag, 0., 0.]))

    @ti.func
    def sdf_2(self, f, grid_pos, flag):
        grid_pos = self.inv_trans(grid_pos, self.get_pos(f, flag), self.rotation[f])
        return Box._sdf(self, f, grid_pos)

    @ti.func
    def normal_2(self, f, grid_pos, flag):
        grid_pos = self.inv_trans(grid_pos, self.get_pos(f, flag), self.rotation[f])
        return self.qrot(self.rotation[f], Box._normal(self, f, grid_pos))

    @ti.func
    def sdf(self, f, grid_pos):
        return ti.min(self.sdf_2(f, grid_pos, -1), self.sdf_2(f, grid_pos, 1))

    @ti.func
    def normal(self, f, grid_pos):
        a = self.sdf_2(f, grid_pos, -1)
        b = self.sdf_2(f, grid_pos, 1)
        a_n = self.normal_2(f, grid_pos, -1)
        b_n = self.normal_2(f, grid_pos, 1)
        m = ti.cast(a <= b, self.dtype)
        return m * a_n + (1-m) * b_n

    @ti.func
    def collider_v(self, f, grid_pos, dt, flag):
        inv_quat = ti.Vector(
            [self.rotation[f][0], -self.rotation[f][1], -self.rotation[f][2], -self.rotation[f][3]]).normalized()
        relative_pos = self.qrot(inv_quat, grid_pos - self.get_pos(f, flag))
        new_pos = self.qrot(self.rotation[f + 1], relative_pos) + self.get_pos(f+1, flag)
        collider_v = (new_pos - grid_pos) / dt  # TODO: revise
        return collider_v, relative_pos

    @ti.func
    def collide(self, f, grid_pos, v_out, dt, mass):
        v_out1 = self.collide2(f, grid_pos, v_out, dt, mass, -1)
        v_out2 = self.collide2(f, grid_pos, v_out1, dt, mass, 1)
        return v_out2

    @ti.func
    def collide2(self, f, grid_pos, v_out, dt, mass, flag):
        dist = self.sdf_2(f, grid_pos, flag)
        influence = min(ti.exp(-dist * self.softness[None]), 1)
        if (self.softness[None] > 0 and influence > 0.1) or dist <= 0:
            D = self.normal_2(f, grid_pos, flag)

            v_in = v_out

            collider_v_at_grid, relative_pos = self.collider_v(f, grid_pos, dt, flag)

            input_v = v_out - collider_v_at_grid
            normal_component = input_v.dot(D)

            grid_v_t = input_v - min(normal_component, 0) * D

            grid_v_t_norm = length(grid_v_t)
            grid_v_t_friction = grid_v_t / grid_v_t_norm * max(0,
                                                               grid_v_t_norm + normal_component * self.friction[None])
            flag = ti.cast(normal_component < 0 and ti.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30, self.dtype)
            grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
            v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence

            if ti.static(self.needs_impact):
                force = -(v_out - v_in) * mass # inverse the force ..
                torque = grid_pos.cross(force)
                self.wrench[f+1] += ti.Vector([torque[0], torque[1], torque[2], force[0], force[1], force[2]]) # wrench is applied for f+1

        return v_out

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot + (self.cfg.init_gap,)

    def get_state(self, f):
        return np.append(super(Gripper, self).get_state(f), self.gap[f])

    @ti.func
    def copy_frame(self, source, target):
        super(Gripper, self).copy_frame(source, target)
        self.gap[target] = self.gap[source]

    def set_state(self, f, state):
        assert len(state) == 8
        super(Gripper, self).set_state(f, state[:7])
        self.gap[f] = state[7]

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        # cfg.h = 0.06
        # cfg.r = 0.03
        cfg.size = (0.03, 0.06, 0.03)
        cfg.minimal_gap = 0.06
        cfg.init_gap = 0.06
        cfg.round = 0
        return cfg



class Prism(Primitive):
    def __init__(self, **kwargs):
        super(Prism, self).__init__(**kwargs)
        self.h = ti.Vector.field(2, self.dtype, shape=())
        self.prot = ti.Vector.field(4, self.dtype, shape=())

    def initialize(self):
        super(Prism, self).initialize()
        self.h[None] = self.cfg.h
        self.prot[None] = self.cfg.prot

    @ti.func
    def _sdf(self, f, grid_pos):
        inv_quat = ti.Vector([self.prot[None][0], -self.prot[None][1],
                              -self.prot[None][2], -self.prot[None][3]]).normalized()
        grid_pos = qrot3d(inv_quat, grid_pos)
        q = ti.abs(grid_pos)
        return max(q[2] - self.h[None][1],
                   max(q[0] * 0.866025 + grid_pos[1] * 0.5, -grid_pos[1]) - self.h[None][0] * 0.5)

    @ti.func
    def _normal(self, f, grid_pos):
        # TODO: replace it with analytical normal later..
        d = ti.cast(1e-4, ti.float64)
        n = ti.Vector.zero(self.dtype, self.dim)
        for i in ti.static(range(self.dim)):
            inc = grid_pos
            dec = grid_pos
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self._sdf(f, inc) - self._sdf(f, dec))
        return n / length(n)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = (0.1, 0.1)
        cfg.prot = (1.0, 0.0, 0.0, 0.0)
        return cfg




class RollingPinExt(Capsule):
    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        vel = self.v[f]
        w = self.w[f]
        dw = vel[0]  # rotate about object y
        dth = vel[1]  # rotate about the world w
        dy = vel[2]  # decrease in y coord...
        y_dir = qrot3d(self.rotation[f], ti.Vector([0., -1., 0.]))
        x_dir = ti.Vector([0., 1., 0.]).cross(y_dir) * (dw * 0.03 + w[0])
        x_dir[1] = dy  # direction
        self.rotation[f + 1] = qmul3d(
            w2quat3d(ti.Vector([0., -dth, 0.]), self.dtype),
            qmul3d(self.rotation[f], w2quat3d(ti.Vector([0., dw, 0.]), self.dtype))
        )
        # print(self.rotation[f+1], self.rotation[f+1].dot(self.rotation[f+1]))
        self.position[f + 1] = max(min(self.position[f] + x_dir, self.xyz_limit[1]), self.xyz_limit[0])



class Knife(Primitive):
    def __init__(self, **kwargs):
        super(Knife, self).__init__(**kwargs)
        self.prism = Prism(h=self.cfg.h, prot=self.cfg.prot)
        self.box = Box(size=self.cfg.size)

    def initialize(self):
        super(Knife, self).initialize()
        self.prism.initialize()
        self.box.initialize()

    @ti.func
    def _sdf(self, f, grid_pos):
        q1 = self.prism._sdf(f, grid_pos)
        q2 = self.box._sdf(f, grid_pos)
        return max(q1, q2)

    @ti.func
    def _sdf_helper(self, f, grid_pos, h):
        q = ti.abs(grid_pos)
        return max(q[2] - h[None][1], max(q[0] * 0.866025 + grid_pos[1] * 0.5, -grid_pos[1]) - h[None][0] * 0.5)

    @ti.func
    def _normal(self, f, grid_pos):
        # TODO: replace it with analytical normal later..
        d = ti.cast(1e-4, ti.float64)
        n = ti.Vector.zero(self.dtype, self.dim)
        for i in ti.static(range(self.dim)):
            inc = grid_pos
            dec = grid_pos
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self._sdf(f, inc) - self._sdf(f, dec))
        return n / length(n)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = (0.1, 0.1)
        cfg.size = (0.1, 0.1, 0.1)
        cfg.prot = (1.0, 0.0, 0.0, 0.0)
        return cfg


class Primitives:
    def __init__(self, cfgs, max_timesteps=1024, dim=3, dtype=ti.f64):
        outs = []
        self.primitives = []
        if cfgs is None:
            cfgs = []

        for i in cfgs:
            if isinstance(i, CN):
                cfg = i
            else:
                cfg = CN(new_allowed=True)
                cfg = cfg._load_cfg_from_yaml_str(yaml.safe_dump(i))
            outs.append(cfg)

        self.action_dims = [0]
        for i in outs:
            primitive = eval(i.shape)(cfg=i, dim=dim, max_timesteps=max_timesteps, dtype=dtype)
            self.primitives.append(primitive)
            self.action_dims.append(self.action_dims[-1] + primitive.action_dim)
        self.n = len(self.primitives)

    @property
    def action_dim(self):
        return self.action_dims[-1]

    @property
    def state_dim(self):
        return sum([i.state_dim for i in self.primitives])

    def set_action(self, s, n_substeps, action):
        action = np.asarray(action).reshape(-1).clip(-1, 1)
        assert len(action) == self.action_dims[-1]
        for i in range(self.n):
            self.primitives[i].set_action(s, n_substeps, action[self.action_dims[i]:self.action_dims[i+1]])

    def get_grad(self, n):
        grads = []
        for i in range(self.n):
            grad = self.primitives[i].get_action_grad(0, n)
            if grad is not None:
                grads.append(grad)
        return np.concatenate(grads, axis=1)

    def set_softness(self, softness=666.):
        for i in self.primitives:
            i.softness[None] = softness

    def get_softness(self):
        if len(self.primitives) == 0:
            return None
        return self.primitives[0].softness[None]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        return self.primitives[item]

    def __len__(self):
        return len(self.primitives)

    def initialize(self):
        for i in self.primitives:
            i.initialize()

    def get_polygons(self):
        polygons = []
        for i in self.primitives:
            corners = i.get_corners()
            x, y, cos, sin = i.get_state(0)
            poly = (np.array([[cos, -sin], [sin, cos]]) @ corners.T).T + np.array([x, y])
            polygons.append((poly, i.cfg.color))

        return polygons




