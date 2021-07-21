from yacs.config import CfgNode as CN


_C = CN()

cfg = _C


# ---------------------------------------------------------------------------- #
# Simulator
# ---------------------------------------------------------------------------- #
_C.SIMULATOR = CN()
_C.SIMULATOR.dim = 3
_C.SIMULATOR.quality = 1  # control the number of particles and size of the grids
_C.SIMULATOR.yield_stress = 50.
_C.SIMULATOR.dtype = "float64"
_C.SIMULATOR.max_steps = 1024
_C.SIMULATOR.n_particles = 9000
_C.SIMULATOR.E = 5e3
_C.SIMULATOR.nu = 0.2  # Young's modulus and Poisson's ratio
_C.SIMULATOR.ground_friction = 1.5
_C.SIMULATOR.gravity = (0, -1, 0)


# ---------------------------------------------------------------------------- #
# PRIMITIVES, i.e., Controller
# ---------------------------------------------------------------------------- #
_C.PRIMITIVES = list()

# ---------------------------------------------------------------------------- #
# Controller
# ---------------------------------------------------------------------------- #
_C.SHAPES = list()


# ---------------------------------------------------------------------------- #
# RENDERER
# ---------------------------------------------------------------------------- #
_C.RENDERER = RENDERER = CN()
RENDERER.spp = 50
RENDERER.max_ray_depth = 2
RENDERER.image_res = (256, 256)
RENDERER.voxel_res = (168, 168, 168)
RENDERER.target_res = (64, 64, 64)

RENDERER.dx = 1. / 150
RENDERER.sdf_threshold=0.37 * 0.56
RENDERER.max_ray_depth=2
RENDERER.bake_size=6
RENDERER.use_roulette=False

RENDERER.light_direction = (2., 1., 0.7)
RENDERER.camera_pos = (0.5, 1.2, 4.)
RENDERER.camera_rot = (0.2, 0)
RENDERER.use_directional_light = False
RENDERER.use_roulette = False
RENDERER.max_num_particles = 1000000

# ---------------------------------------------------------------------------- #
# ENV
# ---------------------------------------------------------------------------- #
_C.ENV = ENV = CN()

loss = ENV.loss = CN()
loss.soft_contact = False
loss_weight = loss.weight = CN()
loss_weight.sdf = 10
loss_weight.density = 10
loss_weight.contact = 1
loss.target_path = ''

ENV.n_observed_particles = 200

_C.VARIANTS = list()


def get_cfg_defaults():
    return _C.clone()
