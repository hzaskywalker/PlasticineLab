import taichi as ti
import multiprocessing
import numpy as np
import taichi as ti
from transforms3d.quaternions import axangle2quat


def test_cfg(cls):
    np.random.seed(0)
    xyz = np.zeros((7,))
    xyz[:3] = np.random.random((3,)) - 0.5
    xyz[3:] = axangle2quat(np.random.random((3,)), np.random.random() * np.pi * 2)

    primitive = cls(cfg=cls.default_config())

    grid_pos = ti.Vector.field(3, shape=(), dtype=ti.f64)
    ans = ti.Vector.field(3, shape=(), dtype=ti.f64)

    primitive.set_state(0, xyz)
    print('finish int')

    @ti.kernel
    def gt():
        ans[None] = primitive.normal2(0, grid_pos[None])

    @ti.kernel
    def sdf():
        ans[None] = primitive.normal(0, grid_pos[None])

    def run(func, a):
        grid_pos[None] = a
        func()
        return ans.to_numpy()

    passed = 0
    while True:
        passed += 1
        if passed % 1000 == 0:
            print('passed', passed)
        a = np.random.random((3,)) * 2 - 1
        assert abs(run(gt, a) - run(sdf, a)).max() < 1e-5, f"{cls}, {a}"


ti.init(arch=ti.gpu, fast_math=False, debug=False)
from plb.engine.primitive.primitives import Torus, Sphere, Capsule, RollingPin, Cylinder, Chopsticks
#test_cfg(Torus)
#test_cfg(Sphere)
#test_cfg(Capsule)
#test_cfg(RollingPin)
test_cfg(Cylinder)
#test_cfg(Chopsticks)
