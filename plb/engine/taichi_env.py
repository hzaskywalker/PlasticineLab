import numpy as np
import cv2
import taichi as ti
import matplotlib.pyplot as plt
from ..config.utils import CfgNode

# TODO: run on GPU, fast_math will cause error on float64's sqrt; removing it cuases compile error..
ti.init(arch=ti.gpu, debug=False, fast_math=True)


@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg: CfgNode, nn=False, loss=True):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        # primitives are environment specific parameters ..
        # move it inside can improve speed; don't know why..
        from .mpm_simulator import MPMSimulator
        from .primitive import Primitives
        from .renderer import Renderer
        from .renderer.renderer2d import Renderer2D
        from .shapes import Shapes
        from .losses import Loss
        from .nn.mlp import MLP

        dim = cfg.SIMULATOR.dim

        self.cfg = cfg.ENV
        self.shapes = Shapes(cfg.SHAPES, cfg.SIMULATOR.dim)
        self.init_particles, self.particle_colors, self.object_id = self.shapes.get()

        cfg.SIMULATOR.defrost()
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)

        self.primitives = Primitives(cfg.PRIMITIVES, dim=dim, dtype=ti.f64 if cfg.SIMULATOR.dtype == 'float64' else ti.f32)

        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitives)
        for i in self.primitives:
            i.dt = self.simulator.dt #TODO: hack for the two-way coupling

        if dim == 3:
            # use the ray-tracer render
            self.renderer = Renderer(cfg.RENDERER, self.primitives)
        else:
            self.renderer = Renderer2D(cfg.RENDERER, self.primitives)

        if nn:
            self.nn = MLP(self.simulator, self.primitives, (256, 256))

        if loss:
            self.loss = Loss(cfg.ENV.loss, self.simulator)
        else:
            self.loss = None
        self._is_copy = True

    def set_copy(self, is_copy: bool):
        self._is_copy = is_copy

    def initialize(self):
        # initialize all taichi variable according to configurations..
        self.primitives.initialize()
        self.simulator.initialize()

        #for i in self.primitives:
        #    i.set_particles(self.simulator.n_particles[None], self.simulator.x)

        if self.renderer is not None:
            self.renderer.initialize()

        if self.loss:
            self.loss.initialize()
            self.renderer.set_target_density(self.loss.target_density.to_numpy() / self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        self.simulator.object_id.from_numpy(self.object_id)
        if self.loss:
            self.loss.clear()

    def render(self, mode='human', wait_time=1, window_name='x', **kwargs):
        assert self._is_copy, "The environment must be in the copy mode for render ..."
        if self.n_particles > 0:
            x = self.simulator.get_x(0)
            self.renderer.set_particles(x, self.particle_colors)
        if self.simulator.dim == 2 and len(self.primitives) > 0:
            self.renderer.set_primitives(self.primitives.get_polygons())
        img = self.renderer.render_frame(**kwargs)

        if img.shape[-1] >= 3:
            img[:, :, :3] = img[:, :, :3].clip(0, 1) * 255
        if img.shape[-1] == 3:
            img = np.uint8(img)

        if mode == 'human':
            cv2.imshow(window_name, img[..., ::-1])
            cv2.waitKey(wait_time)
        elif mode == 'plt':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        else:
            return img

    def step(self, action=None):
        if action is not None:
            action = np.array(action)
        self.simulator.step(is_copy=self._is_copy, action=action)

    def compute_loss(self):
        assert self.loss is not None
        if self._is_copy:
            self.loss.clear()
            return self.loss.compute_loss(0)
        else:
            return self.loss.compute_loss(self.simulator.cur)

    def get_state(self):
        assert self.simulator.cur == 0
        return {
            'state': self.simulator.get_state(0),
            'softness': self.primitives.get_softness(),
            'is_copy': self._is_copy
        }

    def set_state(self, state, softness=None, is_copy=None):
        if softness is None:
            softness = state['softness']
            is_copy = state['is_copy']
            state = state['state']

        self.simulator.cur = 0
        self.simulator.set_state(0, state)
        self.primitives.set_softness(softness)
        self._is_copy = is_copy
        if self.loss:
            self.loss.reset()
            self.loss.clear()

    def p_pos_to_loss_target(self, p, visualize=False):
        self.simulator.clear_grid()
        self.simulator.reset_kernel(p)
        self.simulator.compute_grid_m_kernel(0)
        target_gmass = self.simulator.grid_m.to_numpy()
        self.loss.load_target_density(grids=target_gmass)
        self.renderer.set_target_density(self.loss.target_density.to_numpy() / self.simulator.p_mass)

        if visualize:
            fig, ax = plt.subplots(1, 2)
            img = self.render(mode='array', render_mode='rgb')
            ax[0].imshow(np.uint8(img))
            ax[1].imshow(target_gmass.mean(1).T)
            plt.show()
