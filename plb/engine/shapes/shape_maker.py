import numpy as np
import copy

COLORS = [
    (127 << 16) + 127,
    (127 << 8),
    127,
    127 << 16,
]


class Shapes:
    # make shapes from the configuration
    def __init__(self, cfg, dim=3):
        self.objects = []
        self.colors = []
        self.object_id = []

        self.dim = dim

        state = np.random.get_state()
        np.random.seed(0)  # fix seed 0
        for i in cfg:
            kwargs = {key: eval(val) if isinstance(val, str) else val for key, val in i.items() if key != 'shape'}
            print(kwargs)
            if i['shape'] == 'box':
                self.add_box(**kwargs)
            elif i['shape'] == 'sphere':
                self.add_sphere(**kwargs)
            elif i['shape'] == 'torus':
                self.add_torus(**kwargs)
            else:
                raise NotImplementedError(f"Shape {i['shape']} is not supported!")
        np.random.set_state(state)

    def get_n_particles(self, volume):
        return max(int(volume / 0.2 ** 3) * 10000, 1)

    def add_object(self, particles, color=None, init_rot=None, **extras):
        if init_rot is not None:
            import transforms3d
            q = transforms3d.quaternions.quat2mat(init_rot)
            origin = particles.mean(axis=0)
            particles = (particles[:, :self.dim] - origin) @ q.T + origin
        self.objects.append(particles[:, :self.dim])
        if color is None or isinstance(color, int):
            tmp = COLORS[len(self.objects) - 1] if color is None else color
            color = np.zeros(len(particles), np.int32)
            color[:] = tmp
        self.object_id.append([len(self.object_id)] * len(particles))
        self.colors.append(color)

    def add_box(self, init_pos, width, n_particles=10000, color=None, init_rot=None, **extras):
        # pass
        if isinstance(width, float):
            width = np.array([width] * self.dim)
        else:
            width = np.array(width)
        if n_particles is None:
            n_particles = self.get_n_particles(np.prod(width))
        p = (np.random.random((n_particles, self.dim)) * 2 - 1) * (0.5 * width) + np.array(init_pos)
        self.add_object(p, color, init_rot=init_rot)

    def add_sphere(self, init_pos, radius, n_particles=10000, color=None, init_rot=None, **extras):
        if n_particles is None:
            if self.dim == 3:
                volume = (radius ** 3) * 4 * np.pi / 3
            else:
                volume = (radius ** 2) * np.pi
            n_particles = self.get_n_particles(volume)

        p = np.random.normal(size=(n_particles, self.dim))
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        u = np.random.random(size=(n_particles, 1)) ** (1. / self.dim)
        p = p * u * radius + np.array(init_pos)[:self.dim]
        self.add_object(p, color, init_rot=init_rot)

    def add_torus(self, init_pos, tx, ty, n_particles=10000, color=None, init_rot=None, **extras):

        def length(x):
            return np.sqrt(np.einsum('ij,ij->i', x, x) + 1e-14)

        if n_particles is None:
            raise NotImplementedError

        p = np.ones((n_particles, 3)) * 5

        remain_cnt = n_particles  # how many left to sample
        while remain_cnt > 0:
            x = np.random.random((remain_cnt,)) * (2 * ty + 2 * tx) - (ty + tx)
            y = np.random.random((remain_cnt,)) * (4 * ty) - (2 * ty)
            z = np.random.random((remain_cnt,)) * (2 * ty + 2 * tx) - (ty + tx)

            vec1 = np.stack([x, z], axis=-1)
            len1 = length(vec1) - tx
            vec2 = np.stack([len1, y], axis=-1)
            len2 = length(vec2) - ty

            accept_map = len2 <= 0
            accept_cnt = sum(accept_map)
            start = n_particles - remain_cnt
            p[start:start + accept_cnt] = np.stack([x, y, z], axis=-1)[accept_map]

            remain_cnt -= accept_cnt

        assert np.all(p != 5)
        p = p + np.array(init_pos)[:self.dim]
        self.add_object(p, color, init_rot=init_rot)

    def get(self):
        assert len(self.objects) > 0, "please add at least one shape into the scene"
        return np.concatenate(self.objects), np.concatenate(self.colors), np.concatenate(self.object_id)

    def remove_object(self, index):
        self.objects.pop(index)
        self.colors.pop(index)
