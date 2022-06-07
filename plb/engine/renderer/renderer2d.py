import numpy as np
import taichi as ti
import cv2

class Renderer2D:
    def __init__(self, cfg, primitives=(), **kwargs):
        self.gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
        self.background_image = None

    def set_target_density(self, target):
        # self.gui.set_image(cv2.resize(kwargs['background'], (512, 512)))
        pass

    def initialize(self):
        pass

    def set_particles(self, x, color):
        self.x = x
        self.color = color

    def set_primitives(self, polygons):
        self.polygons = polygons

    def render_frame(self, **kwargs):
        self.gui.clear()
        if self.background_image is not None:
            self.gui.set_image(self.background_image)

        if len(self.polygons) > 0:
            img = self.gui.get_image()
            img = img[::-1, :, [2, 1, 0]]
            img = np.uint8(img[:, :, :3].clip(0, 1) * 255).copy()
            for i, c in self.polygons:
                i[:, 1] = 1. - i[:, 1]
                cv2.polylines(img, [np.int32(i * 512)], True, (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))
                cv2.fillPoly(img, [np.int32(i * 512)], (int(c[0] * 255 * 0.5), int(c[1] * 255 * 0.5), int(c[2] * 255 *0.5)))
            img = (np.float32(img)/255)[::-1, :, [2, 1, 0]]
            self.gui.set_image(img)

        self.gui.circles(self.x[..., [1, 0]], radius=1.5, color=self.color)

        # gui.show(path)

        img = self.gui.get_image()
        img = img[::-1, :, [2, 1, 0]]
        return img