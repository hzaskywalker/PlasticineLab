import numpy as np
import cv2
import taichi as ti
from ...config import load
from .renderer import Renderer
from ..primitive.primitives import Sphere


def test_render():
    cfg = load()
    sphere1 = Sphere(radius=0.2)
    sphere2 = Sphere(radius=0.1)
    sphere3 = Sphere(radius=0.1)

    render = Renderer(cfg.RENDERER, (sphere1, sphere2, sphere3))

    sphere1.set_state(0, [0.1, 0.2, 0.2, 1, 0, 0,0])
    sphere2.set_state(0, [0.9, 0.2, 0.2, 1, 0, 0,0])
    sphere3.set_state(0, [0.5, 0.8, 0.6, 1, 0, 0,0])
    sphere1.set_color([0.3, 0.7, 0.3])
    sphere2.set_color([0.3, 0.3, 0.7])
    sphere3.set_color([0.7, 0.3, 0.7])


    x = np.random.random((10000, 3)) * 0.2 + np.array([0.5, 0.5, 0.5])

    render.set_particles(x, np.zeros(10000,) + (255<<8) + 255)

    img = render.render_frame(40, shape=1, primitive=1, target=0)
    cv2.imshow('x', img)
    cv2.waitKey(0)
