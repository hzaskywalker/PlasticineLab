import numpy as np
from .primitives import Sphere
import taichi as ti

def test_sphere():
    c = Sphere(radius=0.1)
    c.set_state(0, [0.5, 0.5, 0.5, 1, 0, 0, 0])
    print(c.get_state(0))

    @ti.kernel
    def ask():
        print(c.sdf(0, ti.Vector([0., 0., 0.])))
        print('out', c.normal(0, ti.Vector([0., 0., 0.])))

    ask()
    print('should be')
    print(np.linalg.norm([0.5, 0.5, 0.5]) - 0.1)
    print(np.array([-0.5, -0.5, -0.5])/np.linalg.norm([0.5, 0.5, 0.5]))
