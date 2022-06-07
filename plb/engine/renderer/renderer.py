"""
Adapted from taichi element
"""
import open3d as o3d
import taichi as ti
import numpy as np
import math
import time
from .renderer_utils import ray_aabb_intersection, inf, out_dir

DIFFUSE = 0
SPECULAR = 1

fov = 0.23
dist_limit = 100

exposure = 1.5
light_direction_noise = 0.03
light_color = [1.0, 1.0, 1.0]


@ti.data_oriented
class Renderer:
    def __init__(self, cfg, primitives=(), **kwargs):
        # overwrite configurations
        for i, v in kwargs.items():
            cfg[i] = v
        print("Initialize Renderer")
        print(str(cfg).replace('\n', '  \n'))

        self.dx = cfg.dx
        self.spp = cfg.spp
        self.voxel_res = cfg.voxel_res
        self.max_num_particles = cfg.max_num_particles
        self.bake_size = cfg.bake_size
        self.max_ray_depth = cfg.max_ray_depth
        self.sdf_threshold = cfg.sdf_threshold
        self.use_directional_light = cfg.use_directional_light
        self.light_direction = cfg.light_direction
        self.image_res = cfg.image_res
        self.use_roulette = cfg.use_roulette

        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]

        self.aspect_ratio = self.image_res[0] / self.image_res[1]
        self.inv_dx = 1 / self.dx

        self.camera_pos = cfg.camera_pos #ti.Vector([float(i) for i in cfg.camera_pos])
        self.camera_rot = cfg.camera_rot

        self.camera_pos_multi = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.camera_rot_multi = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.camera_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.mask = None

        # taichi part
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.depth_buffer = ti.field(dtype=ti.f32)
        self.mask_buffer = ti.field(dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)

        self.particle_x = ti.Vector.field(3, dtype=ti.f32)
        self.particle_color = ti.field(dtype=ti.i32)
        self.particle_mask = ti.field(dtype=ti.i32)

        self.num_particles = ti.field(ti.i32, shape=())

        self.volume = ti.field(dtype=ti.int64)
        self.sdf = ti.field(dtype=ti.f32)
        self.sdf_copy = ti.field(dtype=ti.f32)

        self.target_res = cfg.target_res
        self.target_density = ti.field(dtype=ti.f32, shape=self.target_res)
        self.target_density2 = ti.field(dtype=ti.f32, shape=self.target_res)
        self.color_vec = ti.Vector.field(3, dtype=ti.f32)
        self.mask_vec = ti.field(dtype=ti.i32)
        self.target_density_color = ti.Vector([0.1, 0.3, 0.9])

        self.primitives = primitives.primitives #TODO: it seems that new version of taichi only supports list indexing in taichi scope..

        ti.root.dense(ti.ij, self.image_res).place(self.color_buffer)
        ti.root.dense(ti.ij, self.image_res).place(self.depth_buffer, self.mask_buffer)
        ti.root.dense(ti.l, self.max_num_particles).place(self.particle_x, self.particle_color, self.particle_mask)
        ti.root.dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, [i // 4 for i in self.voxel_res]).place(self.volume, self.sdf,
                                                                                               self.sdf_copy,
                                                                                               self.color_vec, self.mask_vec)

        # flags
        self.visualize_rgb = ti.field(dtype=ti.i32, shape=())
        self.visualize_depth = ti.field(dtype=ti.i32, shape=())
        self.visualize_target = ti.field(dtype=ti.i32, shape=())
        self.visualize_primitive = ti.field(dtype=ti.i32, shape=())
        self.visualize_shape = ti.field(dtype=ti.i32, shape=())

    def set_object_mask(self, mask):
        self.mask = mask

    # -----------------------------------------------------
    # build sdf from particles
    # -----------------------------------------------------
    @ti.func
    def smooth(self, volume, volume_out, res: ti.template()):
        a, b, c = ti.static(res)
        for id in ti.grouped(volume):
            if id[0] >= 1 and id[1] >= 1 and id[2] >= 1 and id[0] < a - 1 and id[1] < b - 1 and id[2] < c - 1:
                sum = 0.0
                for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    sum += volume[id + ti.Vector([i, j, k])]
                volume_out[id] = sum / 27.
            else:
                volume_out[id] = 1.

    @ti.kernel
    def build_sdf_from_particles(self):
        # bake
        size = ti.static(self.bake_size)
        resx, resy, resz = ti.static(self.voxel_res)
        for i in ti.grouped(self.volume):
            self.volume[i] = (self.volume[i] + 1) * 2 - 1

        num_p = self.num_particles[None]
        for id, i, j, k in ti.ndrange(num_p, (-size - 1, size + 1), (-size - 1, size + 1), (-size - 1, size + 1)):
            p = (self.particle_x[id] - self.bbox[0]) * self.inv_dx  # 0 is the lower bound
            coord = p.cast(ti.i32)

            idx = coord + ti.Vector([i, j, k])
            if idx[0] >= 0 and idx[1] >= 0 and idx[2] >= 0 and \
                    idx[0] < resx and idx[1] < resy and idx[2] < resz:
                color = self.particle_color[id]
                mask = self.particle_mask[id]

                dist = (idx - p).norm()
                dist = min(max(0, (255 * 0.2 * dist)), 255)
                char = (ti.cast(dist, ti.int64) << 24) + color
                ti.atomic_min(self.volume[idx], char)

                char2 = (ti.cast(dist, ti.int64) << 24) + mask
                ti.atomic_min(self.mask_vec[idx], char2)

        for i in ti.grouped(self.volume):
            c = self.volume[i]
            for j in ti.static(range(2, -1, -1)):
                self.color_vec[i][j] = (c & 255) / 255.
                c = c >> 8
            self.sdf[i] = (c & 255) / 255.
            self.mask_vec[i] = self.mask_vec[i] & ((1<<24) - 1)

        for _ in ti.static(range(1)):
            self.smooth(self.sdf, self.sdf_copy, self.voxel_res)
            self.smooth(self.sdf_copy, self.sdf, self.voxel_res)

    # -----------------------------------------------------
    # sample textures
    # -----------------------------------------------------

    @ti.func
    def sample_mask(self, pos, res: ti.template()):
        # bilinear interpolation to sample a tex from texture
        pos = (pos - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
        out = 0
        if pos.min() >= 0 and pos.max() <= 1:
            a, b, c = ti.static(res)
            pos = pos * ti.Vector([a, b, c])
            base = ti.min(ti.cast(pos, ti.i32), ti.Vector([a, b, c]) - 1)  # clip

            x, y, z = base[0], base[1], base[2]
            x1, y1, z1 = min(x + 1, a - 1), min(y + 1, b - 1), min(z + 1, c - 1)  # clip again..
            out = ti.max(self.mask_vec[x, y, z], self.mask_vec[x1, y1, z1])
        return out

    @ti.func
    def sample_tex(self, tex, pos, res: ti.template()):
        # bilinear interpolation to sample a tex from texture
        a, b, c = ti.static(res)
        pos = pos * ti.Vector([a, b, c])
        base = ti.min(ti.cast(pos, ti.i32), ti.Vector([a, b, c]) - 1)  # clip
        fx = pos - base

        x, y, z = base[0], base[1], base[2]
        x1, y1, z1 = min(x + 1, a - 1), min(y + 1, b - 1), min(z + 1, c - 1)  # clip again..
        c00 = tex[base] * (1 - fx[0]) + tex[x1, y, z] * fx[0]
        c01 = tex[x, y, z1] * (1 - fx[0]) + tex[x1, y, z1] * fx[0]
        c10 = tex[x, y1, z] * (1 - fx[0]) + tex[x1, y1, z] * fx[0]
        c11 = tex[x, y1, z1] * (1 - fx[0]) + tex[x1, y1, z1] * fx[0]

        c0 = c00 * (1 - fx[1]) + c10 * fx[1]
        c1 = c01 * (1 - fx[1]) + c11 * fx[1]

        return c0 * (1 - fx[2]) + c1 * fx[2]

    @ti.func
    def sample_sdf(self, pos):
        pos = (pos - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
        out = 0.0
        if pos.min() >= 0 and pos.max() <= 1:
            out = self.sample_tex(self.sdf, pos, self.voxel_res) - self.sdf_threshold  # 0.35
        return out

    @ti.func
    def sample_color(self, pos):
        pos = (pos - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
        out = ti.Vector([0., 0., 0.])
        if pos.min() >= 0 and pos.max() <= 1:
            out = self.sample_tex(self.color_vec, pos, self.voxel_res)
        return out

    @ti.func
    def sample_normal(self, sample_sdf_func: ti.template(), p):
        d = 1e-3  # this seems to be important ... otherwise it won't be smooth..
        n = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            inc = p
            dec = p
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (sample_sdf_func(inc) - sample_sdf_func(dec))
        return n.normalized()

    @ti.func
    def ground_color(self, p):
        color = ti.Vector([0.3, 0.5, 0.7])
        #if p[0] <= 1 and p[0] >= 0 and p[2] <= 1 and p[2] >= 0 and False:
        #    color *= ((ti.cast(p[0] / 0.25, ti.int32) + ti.cast(p[2] / 0.25, ti.int32)) % 2) * 0.2 + 0.35
        #else:
        color *= 0.4
        return color

    @ti.func
    def sample_target_density(self, p):
        return self.sample_tex(self.target_density, p, self.target_res)

    # -----------------------------------------------------
    # collision handler
    # -----------------------------------------------------
    @ti.func
    def next_hit(self, o, d):
        normal = ti.Vector([0., 0., 0.])
        color = ti.Vector([0.0, 0.0, 0.0])
        closest = inf
        roughness = 0.05
        material = DIFFUSE  # diffuse

        object_mask = 0

        background_color = ti.Vector([0.6, 0.7, 0.7]) # ti.Vector([0.6, 0.7, 0.7])

        """
        # add background
        if d[2] != 0:
            ray_closest = -(o[2] + 5.5) / d[2]
            # ray_closest = (0. - o[2])/d[2]
            if ray_closest > 0 and ray_closest < closest and d[2] < 0:  # and o[1] + d[1] * ray_closest <=1:
                closest = ray_closest
                normal = ti.Vector([0.0, 0.0, 1.0])
                color = background_color
                roughness = 0.0

            ray_closest = -(o[2] - 5.5) / d[2]
            if ray_closest > 0 and ray_closest < closest and d[2] > 0 and False:  # and o[1] + d[1] * ray_closest <=1:
                closest = ray_closest
                normal = ti.Vector([0.0, 0.0, -1.0])
                color = background_color
                roughness = 0.0

        # add background
        if d[0] != 0:
            ray_closest = -(o[0] + 5.5) / d[0]
            if ray_closest > 0 and ray_closest < closest and d[0] < 0:  # and o[1] + d[1] * ray_closest <=1:
                closest = ray_closest
                normal = ti.Vector([1.0, 0.0, 0.0])
                color = background_color
                roughness = 0.0

            ray_closest = -(o[0] - 5.5) / d[0]
            if ray_closest > 0 and ray_closest < closest and d[0] > 0:  # and o[1] + d[1] * ray_closest <=1:
                closest = ray_closest
                normal = ti.Vector([-1.0, 0.0, 0.0])
                color = background_color
                roughness = 0.0
        """

        # add ground...
        if d[1] < 0:
            ground_dist = (o[1] + 0.002) / (-d[1])
            if ground_dist < dist_limit and ground_dist < closest:
                closest = ground_dist
                normal = ti.Vector([0., 1., 0.])
                color = self.ground_color(o + d * closest)
                roughness = 0.0
                material = DIFFUSE  # specular
        # return closest, normal, color, roughness

        if ti.static(len(self.primitives) > 0):
            if self.visualize_primitive[None]:
                # copy from ray match
                j = 0
                dist = 0.0
                sdf_val = inf
                sdf_id = 0

                while j < 200 and dist < dist_limit and sdf_val > 1e-8:
                    pp = o + dist * d

                    sdf_val = inf
                    for i in ti.static(range(len(self.primitives))):
                        dd = ti.cast(self.primitives[i].sdf(0, pp), ti.f32)
                        if dd < sdf_val:
                            sdf_val = dd
                            sdf_id = i
                    dist += sdf_val
                    j += 1

                if dist < closest and dist < dist_limit:
                    closest = dist
                    for i in ti.static(range(len(self.primitives))):
                        if sdf_id == i:
                            normal = ti.cast(self.primitives[i].normal(0, o + dist * d), ti.f32)
                            color = self.primitives[i].color[None]

                            object_mask = - sdf_id - 1
                    roughness = 0.
                    material = DIFFUSE

        # ------------------------ plasticine --------------------------------
        # shoot function
        if self.visualize_shape[None]:
            intersect, tnear, tfar = ray_aabb_intersection(self.bbox[0], self.bbox[1], o, d)

            if intersect:
                tnear = max(tnear, 0.)
                pos = o + d * (tnear + 1e-4)
                step = ti.Vector([0., 0., 0.])

                for j in range(500):
                    s = self.sample_sdf(pos)
                    if s < 0:
                        back_step = step
                        for k in range(20):
                            back_step = back_step * 0.5
                            if self.sample_sdf(pos - back_step) < 0:
                                pos -= back_step

                        dist = (o - pos).norm()
                        if dist < closest:
                            closest = dist
                            normal = self.sample_normal(self.sample_sdf, pos)
                            color = self.sample_color(pos)
                            object_mask = self.sample_mask(pos, self.voxel_res)
                            material = DIFFUSE
                        break
                    else:
                        step = d * max(s * 0.05, 0.01)
                        pos += step

        if self.visualize_target[None]:
            # ------------------------ target density ----------------------------
            intersect, tnear, tfar = ray_aabb_intersection(ti.Vector([0.0, 0.0, 0.0]), ti.Vector([1.0, 1.0, 1.0]), o, d)
            if intersect:
                tnear = max(tnear, 0.)
                pos = o + d * (tnear + 1e-4)
                step = ti.Vector([0., 0., 0.])
                total_forward = 0.0

                for j in range(500):
                    if total_forward + tnear > tfar:
                        break
                    s = self.sample_target_density(pos)
                    if s < 0:
                        back_step = step
                        for k in range(20):
                            back_step = back_step * 0.5
                            if self.sample_target_density(pos - back_step) < 0:
                                pos -= back_step

                        dist = (o - pos).norm()
                        if dist < closest:
                            closest = dist
                            normal = self.sample_normal(self.sample_target_density, pos)
                            color = self.target_density_color
                            material = DIFFUSE
                        break
                    else:
                        step_length = (1.0 / self.target_res[0])
                        step = d * step_length
                        total_forward += step_length
                        pos += step

        return closest, normal, color, roughness, material, object_mask

    # -----------------------------------------------------
    # ray tracing
    # -----------------------------------------------------
    @ti.func
    def sample_sphere(self):
        u = ti.random(ti.f32)
        v = ti.random(ti.f32)
        x = u * 2 - 1
        phi = v * 2 * 3.14159265358979
        yz = ti.sqrt(1 - x * x)
        return ti.Vector([x, yz * ti.cos(phi), yz * ti.sin(phi)])

    @ti.func
    def sky_color(self, direction):
        coeff1 = direction.dot(ti.Vector([0.8, 0.65, 0.15])) * 0.5 + 0.5
        coeff1 = ti.max(ti.min(coeff1, 1.), 0.)
        light = coeff1 * ti.Vector([0.9, 0.9, 0.9]) + (1. - coeff1) * ti.Vector([0.7, 0.7, 0.8])
        #return light * 1.5
        return light * 10.

    @ti.func
    def trace(self, pos, d):
        contrib = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        depth = 0
        hit_sky = 1
        ray_depth = 0

        while depth < self.max_ray_depth:
            closest, normal, c, roughness, material, object_mask = self.next_hit(pos, d)
            hit_pos = pos + closest * d
            depth += 1
            ray_depth = depth
            if normal.norm() != 0:
                out_direction = ti.Vector([0., 0., 0.])
                if material == SPECULAR:
                    out_direction = d - d.dot(normal) * 2 * normal
                else:
                    out_direction = out_dir(normal)
                glossy = self.sample_sphere() * roughness
                d = (out_direction + glossy).normalized()

                pos = hit_pos + 1e-4 * d
                throughput *= c

                # throughput *= absorption

                if ti.static(self.use_directional_light):
                    dir_noise = ti.Vector([
                        ti.random() - 0.5,
                        ti.random() - 0.5,
                        ti.random() - 0.5
                    ]) * light_direction_noise
                    direct = (ti.Vector(self.light_direction) + dir_noise).normalized()
                    dot = direct.dot(normal)
                    if dot > 0:
                        dist = self.next_hit(pos, direct)[0]
                        if dist > dist_limit:
                            contrib += throughput * ti.Vector(
                                light_color) * dot
            else:  # hit sky
                hit_sky = 1
                depth = self.max_ray_depth

            if ti.static(self.use_roulette):
                max_c = throughput.max()
                if ti.random() > max_c:
                    depth = self.max_ray_depth
                    throughput = [0, 0, 0]
                else:
                    throughput /= max_c

        if hit_sky:
            if ray_depth != 1:
                # contrib *= max(d[1], 0.05)
                pass
            else:
                # directly hit sky
                pass
        else:
            throughput *= 0
        out = contrib
        if ti.static(not self.use_directional_light):
            out = throughput * self.sky_color(d)
        return out

    @ti.kernel
    def copy_rgb(self, img: ti.ext_arr(), samples: ti.i32):
        for i, j in self.color_buffer:
            u = 1.0 * i / self.image_res[0]
            v = 1.0 * j / self.image_res[1]

            darken = 1.0 - self.vignette_strength * max((ti.sqrt(
                (u - self.vignette_center[0]) ** 2 +
                (v - self.vignette_center[1]) ** 2) - self.vignette_radius), 0)

            for c in ti.static(range(3)):
                img[i, j, c] = ti.sqrt(self.color_buffer[i, j][c] * darken *
                                       exposure / samples)

    @ti.kernel
    def copy_depth(self, img: ti.ext_arr(), samples: ti.i32):
        for i, j in self.depth_buffer:
            img[i, j, 0] = self.depth_buffer[i, j] / samples
            img[i, j, 1] = self.mask_buffer[i, j]

    @ti.kernel
    def compute_camera_mat(self):
        self.camera_mat[None] = ti.Matrix([
            [ti.cos(self.camera_rot_multi[None][1]), 0.0000000, ti.sin(self.camera_rot_multi[None][1])],
            [0.0000000, 1.0000000, 0.0000000],
            [-ti.sin(self.camera_rot_multi[None][1]), 0.0000000, ti.cos(self.camera_rot_multi[None][1])],
        ]) @ ti.Matrix([
            [1.0000000, 0.0000000, 0.0000000],
            [0.0000000, ti.cos(self.camera_rot_multi[None][0]), ti.sin(self.camera_rot_multi[None][0])],
            [0.0000000, -ti.sin(self.camera_rot_multi[None][0]), ti.cos(self.camera_rot_multi[None][0])],
        ])

    @ti.kernel
    def render(self):
        ti.block_dim(128)
        # print(self.sample_sdf(self.bbox[0] + 0.05))
        # return

        mat = self.camera_mat[None]

        for u, v in self.color_buffer:
            pos = self.camera_pos_multi[None]
            d = ti.Vector([
                - (2 * fov * (u + ti.random(ti.f32)) / self.image_res[1] -
                 fov * self.aspect_ratio - 1e-5),
                - ( 2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5 ), 1.0
            ])
            #print(u, v, d)
            d = mat @ d.normalized()
            if self.visualize_rgb[None]:
                contrib = self.trace(pos, d)
                self.color_buffer[u, v] += contrib
            if self.visualize_depth[None]:
                depth, x1, x2, x3, x4, obj = self.next_hit(pos, d)
                self.depth_buffer[u, v] += (depth * d).dot(mat @ ti.Vector([0, 0, 1]))
                if ti.abs(obj) > ti.abs(self.mask_buffer[u, v]):
                    self.mask_buffer[u, v] = obj

    @ti.kernel
    def initialize_particles_kernel(self, x: ti.ext_arr(), color: ti.ext_arr(), mask: ti.ext_arr()):
        self.bbox[0] = [inf, inf, inf]
        self.bbox[1] = [-inf, -inf, -inf]
        for i in range(self.num_particles[None]):
            for c in ti.static(range(3)):
                self.particle_x[i][c] = x[i, c]
                v = (ti.floor(self.particle_x[i][c] * self.inv_dx) - 6.0) * self.dx
                ti.atomic_min(self.bbox[0][c], v)
                ti.atomic_max(self.bbox[1][c], v)
            self.particle_color[i] = ti.cast(color[i], ti.i32)
            self.particle_mask[i] = ti.cast(mask[i], ti.i32)

    def set_particles(self, x, color, mask=None):
        # assume that num_part and particle_x is all calculated ...
        self.num_particles[None] = len(x)  # set num_particles ...
        if mask is None:
            mask = self.mask if self.mask is not None else np.zeros_like(color)
        self.initialize_particles_kernel(x, color, mask)

        # update box..
        bbox = self.bbox.to_numpy()
        desired_res = (bbox[1] - bbox[0]) / self.dx
        for a, b in zip(desired_res, self.voxel_res):
            assert a < b, f"the sdf {bbox} should be smaller {desired_res} < {self.voxel_res}"
        bbox[1] = bbox[0] + np.array(self.voxel_res) * self.dx
        self.bbox.from_numpy(bbox)

        # reset ..
        self.volume.fill(int((1 << 31) - 1))
        self.build_sdf_from_particles()

    def render_frame(self, spp=None, **kwargs):
        if spp is None:
            spp = self.spp

        last_t = 0
        render_mode = kwargs.get('render_mode', 'rgb')
        self.visualize_rgb[None] = 1 if 'rgb' in render_mode else 0
        self.visualize_depth[None] = 1 if 'd' in render_mode else 0
        visualize_target = kwargs.get('target', 0)
        self.visualize_shape[None] = kwargs.get('shape', 1)
        self.visualize_primitive[None] = kwargs.get('primitive', 1)
        self.color_buffer.fill(0)
        self.depth_buffer.fill(0)
        self.mask_buffer.fill(0)

        for i in range(1, 1 + spp):
            # Opacity=50%
            self.visualize_target[None] = int(i % 2 == 0) * visualize_target
            self.render()

            interval = 20
            if i % interval == 0:
                if last_t != 0:
                    ti.sync()
                last_t = time.time()

        img = []
        if self.visualize_rgb[None]:
            rgb_img = np.zeros((self.image_res[0], self.image_res[1], 3), dtype=np.float32)
            self.copy_rgb(rgb_img, spp)
            img.append(rgb_img)
        if self.visualize_depth[None]:
            depth_img = np.zeros((self.image_res[0], self.image_res[1], 2), dtype=np.float32)
            self.copy_depth(depth_img, spp)
            img.append(depth_img)
        img = np.concatenate(img, axis=-1)

        return img.transpose(1, 0, 2)  # opencv format for render..

    @ti.kernel
    def smooth_target_density(self):
        for I in ti.grouped(self.target_density):
            self.target_density2[I] = -self.target_density2[I] + 3
        self.smooth(self.target_density2, self.target_density, self.target_res)

    @ti.kernel
    def fill_target_density(self, val: ti.f32):
        for I in ti.grouped(self.target_density):
            self.target_density[I] = val
            self.target_density2[I] = val

    def set_target_density(self, target_density=None):
        if target_density is not None:
            self.target_density2.from_numpy(target_density.astype(np.float32))
            self.smooth_target_density()
        else:
            self.fill_target_density(0.0)

    def initialize(self):
        self.camera_pos_multi[None] = self.camera_pos
        self.camera_rot_multi[None] = self.camera_rot
        self.compute_camera_mat()
        print("initialize camera at", self.camera_pos, self.camera_rot)

    def set_camera_pose(self, *args, camera_pos=None, camera_rot=None):
        if camera_rot is not None:
            self.camera_rot_multi[None] = camera_rot
        if camera_pos is not None:
            self.camera_pos_multi[None] = camera_pos
        self.compute_camera_mat()

    def set_camera_matrix(self, mat):
        self.camera_mat[None] = mat


    def get_int(self, image_res=None):
        image_res = image_res or self.image_res
        int = np.array([
            - np.array([2 * fov / image_res[1], 0, -fov - 1e-5,]),
            - np.array([0, 2 * fov / image_res[1], -fov - 1e-5,]),
            [0, 0, 1]
        ])
        return np.linalg.inv(int)

    def get_ext(self):
        T = np.zeros((4, 4))
        T[:3, :3] = self.camera_mat.to_numpy()
        T[:3, 3] = self.camera_pos_multi.to_numpy()
        T[3, 3] = 1
        return np.linalg.inv(T)

    def project_camera_xyz_to_uv(self, xyz):
        xyz = self.get_int() @ xyz
        return xyz[:2]/xyz[2]

    def projection_matrix(self):
        # from world to int
        import numpy as np
        int = np.zeros((3, 4))
        int[:3,:3] = self.get_int()
        return int @ self.get_ext()

    def normalized_projection_matrix(self):
        # from world to int
        import numpy as np
        int = np.zeros((3, 4))
        int[:3,:3] = self.get_int()
        a = 2.0/(self.image_res[0]-1)
        b = 2.0/(self.image_res[1]-1)
        int = np.array([[a, 0, -1], -np.array([0, b, -1]), [0, 0, 1]])@int
        return int @ self.get_ext()[:,[0,2,1,3]]

    def getRT(self):
        ext = np.linalg.inv(self.get_ext())
        return ext[:3, :3].copy(), ext[:3, 3].copy()

    def get_o3d_camera(self):
        int = self.get_int()
        fx, fy = int[0, 0], int[1, 1]
        cx, cy = int[0, 2], int[1, 2]
        w, h = self.image_res[1], self.image_res[0]
        cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        return cam

    def rgbd2pcd(self, rgb, depth):
        # rgb = rgb[:, ::-1] # notice that we have reversed the image
        # depth = depth[:, ::-1]
        cam_param = self.get_o3d_camera()
        extrinsic = self.get_ext()
        rgb = o3d.geometry.Image(np.ascontiguousarray(np.rot90(rgb, 0, (0, 1))).astype(np.uint8))
        depth = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1., depth_trunc=np.inf,
                                                                  convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_param)
        pcd.transform(np.linalg.inv(extrinsic))
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        return pcd # pcd.points, pcd.colors

    def draw_geometries(self, objects, mode='human'):
        if mode == 'human':
            o3d.visualization.draw_geometries(objects)
        else:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=512, height=512, visible=False)
            if isinstance(objects, tuple) or isinstance(objects, list):
                for geom in objects:
                    vis.add_geometry(geom)
            else:
                vis.add_geometry(objects)
            ctr = vis.get_view_control()
            cam_param = self.get_o3d_camera()

            o3d_cam = o3d.camera.PinholeCameraParameters()
            o3d_cam.intrinsic = cam_param
            o3d_cam.extrinsic = self.get_ext()

            ctr.convert_from_pinhole_camera_parameters(o3d_cam, allow_arbitrary=True)
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            vis.destroy_window()
            return np.uint8(np.asarray(image) * 255)

    def visualize_pcd(self, pcd, mode='rgb'):
        return self.draw_geometries([pcd], mode)

    def setRT(self, R, trans):
        self.camera_mat.from_numpy(R)
        self.camera_pos_multi.from_numpy(trans)
