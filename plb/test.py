import taichi as ti
import numpy as np
import tina
from engine.mpm_simulator import MPMSimulator
from engine.primitive.primitives import Plane, Sphere
import trimesh

from utils import get_rotation_between


def main():
    n0 = np.array([0, 1., 0.])
    ti.init(arch=ti.cuda)
    scene = tina.Scene(1366, maxfaces=2 ** 18, smoothing=True)
    pars = tina.SimpleParticles(radius=0.003)
    scene.add_object(pars, tina.Classic())
    scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
                                        tina.translate([0, 0.3, 0]) @ tina.scale(0.5)), tina.Lamp(color=64))
    ball_radius = 0.06
    ball = tina.MeshTransform(tina.PrimitiveMesh.sphere())
    metal = tina.PBR(basecolor=[1.0, 0.9, 0.8], metallic=0.8, roughness=0.4)
    scene.add_object(ball, metal)
    # scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
    #                                     tina.translate([2.74432755e+01 / 80.0, 0.3,
    #                                                     (1.09041557e+01 + 2.47605629e+01) / 2 / 80.0]) @
    #                                     tina.quaternion(
    #                                         tuple(get_rotation_between(n0, np.array([0.001242, 0.3266, 0.9451]))))
    #                                     ), tina.Lamp(color=64))
    # scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
    #                                     tina.translate([2.74432755e+01 / 80.0, 0.3, 2.47605629e+01 / 80.0]) @
    #                                     tina.quaternion(
    #                                         tuple(get_rotation_between(n0, -np.array(
    #                                             [0.47275931, -0.32556817, -0.81884307]))))
    #                                     @ tina.scale(0.5)), tina.Lamp(color=64))
    # scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
    #                                     tina.translate([1.54432755e+01 / 80.0, 0.3, 3.16887665e+01 / 80.0]) @
    #                                     tina.quaternion(
    #                                         tuple(get_rotation_between(n0, -np.array(
    #                                             [-0.47275931, -0.32556817, -0.81884307]))))
    #                                     @ tina.scale(0.5)), tina.Lamp(color=64))
    # scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
    #                                     tina.translate([1.54432755e+01 / 80.0, 0.3, 3.97595286e+00 / 80.0]) @
    #                                     tina.quaternion(
    #                                         tuple(get_rotation_between(n0, -np.array([-0.9451, -0.3266, 0.001242]))))
    #                                     @ tina.scale(0.5)), tina.Lamp(color=64))
    # scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
    #                                     tina.translate([3.44327569e+00 / 80.0, 0.3, 1.09041557e+01 / 80.0]) @
    #                                     tina.quaternion(
    #                                         tuple(get_rotation_between(n0, -np.array(
    #                                             [-0.47275931, -0.32556817, 0.81884307]))))
    #                                     @ tina.scale(0.5)), tina.Lamp(color=64))
    # scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
    #                                     tina.translate([3.44327569e+00 / 80.0, 0.3, 2.47605629e+01 / 80.0]) @
    #                                     tina.quaternion(
    #                                         tuple(get_rotation_between(n0, -np.array(
    #                                             [0.47275931, -0.32556817, 0.81884307]))))
    #                                     @ tina.scale(0.1)), tina.Lamp(color=64))
    NUM_PARTICLES = 2 ** 16

    plane_bottom = Plane(cfg=None)
    plane_1 = Plane(cfg=None, init_pos=(2.74432755e+01 / 80.0, 0.3, 1.09041557e+01 / 80.0),
                    init_rot=(tuple(get_rotation_between(n0, -np.array([0.9451, -0.3266, 0.001242])))))
    plane_2 = Plane(cfg=None, init_pos=(2.74432755e+01 / 80.0, 0.3, 2.47605629e+01 / 80.0),
                    init_rot=(tuple(get_rotation_between(n0, -np.array([0.47275931, -0.32556817, -0.81884307])))))
    plane_3 = Plane(cfg=None, init_pos=(1.54432755e+01 / 80.0, 0.3, 3.16887665e+01 / 80.0),
                    init_rot=(tuple(get_rotation_between(n0, -np.array([-0.47275931, -0.32556817, -0.81884307])))))
    plane_4 = Plane(cfg=None, init_pos=(1.54432755e+01 / 80.0, 0.3, 3.97595286e+00 / 80.0),
                    init_rot=(tuple(get_rotation_between(n0, -np.array([-0.9451, -0.3266, 0.001242])))))
    plane_5 = Plane(cfg=None, init_pos=(3.44327569e+00 / 80.0, 0.3, 1.09041557e+01 / 80.0),
                    init_rot=(tuple(get_rotation_between(n0, -np.array([-0.47275931, -0.32556817, 0.81884307])))))
    plane_6 = Plane(cfg=None, init_pos=(3.44327569e+00 / 80.0, 0.3, 2.47605629e+01 / 80.0),
                    init_rot=(tuple(get_rotation_between(n0, -np.array([0.47275931, -0.32556817, 0.81884307])))))
    from yacs.config import CfgNode as CN
    ball_cfg = CN()
    ball_cfg.action = CN()
    ball_cfg.action.dim = 3
    ball_cfg.action.scale = (1, 1, 1)
    ball_prim = Sphere(cfg=ball_cfg, radius=ball_radius,
                       init_pos=(15.44/80.0, 0.6, 17.8/80.0))

    sim = MPMSimulator(
        num_particles=NUM_PARTICLES,
        E=5e6,
        dt_scale=4,
        gravity=(0., -9.8, 0.),
        primitives=(plane_bottom, plane_1, plane_2, plane_3, plane_4, plane_5, plane_6, ball_prim)
    )
    sim.initialize()
    plane_bottom.initialize()
    plane_1.initialize()
    plane_2.initialize()
    plane_3.initialize()
    plane_4.initialize()
    plane_5.initialize()
    plane_6.initialize()
    ball_prim.initialize()
    gel = trimesh.load('gel.STL')
    assert gel.is_watertight
    gel_points = trimesh.sample.volume_mesh(gel, NUM_PARTICLES)
    gel_points = gel_points / 80.0
    gel_points[:, 1] += 0.3
    sim.reset(gel_points)

    gui = ti.GUI(res=1366)
    scene.init_control(gui, blendish=True)

    t = 0
    while gui.running:
        ball_state = ball_prim.get_state(0)
        if ball_state[1] > 0.4625 + 0.5 * ball_radius:
            ball_action = np.array([0, -0.002, 0])
        else:
            ball_action = np.array([-0.002, 0, 0])
        ball_prim.set_action(0, sim.substeps, ball_action)
        # print(ball_prim.get_state(1))
        # ball_prim.set_state(0, ball_state)
        ball.set_transform(tina.translate(ball_state[:3]) @ tina.scale(ball_radius))
        scene.input(gui)
        sim.step(True)
        t += 1
        pts = sim.get_x(0)
        np.savetxt(f'outputs/{t:08d}.xyz', pts, fmt='%.6f')
        v = sim.get_v(0)
        print(pts.shape, pts[0], v[0])

        pars.set_particles(pts)
        # colors = np.array(list(map(ti.hex_to_rgb, [0x068587, 0xED553B, 0xEEEEF0])))
        # pars.set_particle_colors(colors[mpm.material.to_numpy()])
        scene.render()
        gui.set_image(scene.img)
        gui.show()


if __name__ == '__main__':
    main()
