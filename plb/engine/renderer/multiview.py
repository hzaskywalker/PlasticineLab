import numpy as np
import matplotlib
import os
import sklearn.cluster
from ..taichi_env import TaichiEnv
import transforms3d

def colormap_depth(img):
    if img.shape[-1] == 1:
        img = img[..., 0]

    minima = img.min()
    maxima = img.max()
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
    img = np.uint8(mapper.to_rgba(img)[..., :3] * 255)
    return img


def lookat(center, theta, phi, radius):
    R = transforms3d.euler.euler2mat(theta, phi, 0., 'sxyz')
    b = np.array([0, 0, radius], dtype=float)
    back = R[0:3, 0:3].dot(b)
    return R, center - back

def visualize_mask(masks):
    image = None
    for idx, i in enumerate(masks):
        if image is None:
            image = np.uint8(i)
        else:
            image = image + np.uint8(i) * (idx + 1)
    return image

def compute_mask_and_bbox(x):
    pass

def render_everything(R, T,
                      env: TaichiEnv,
                      object_mask=None, **kwargs):
    renderer = env.renderer
    renderer.set_object_mask(object_mask)
    renderer.setRT(R, T)

    out = env.render('rgb_array', render_mode='rgbd', **kwargs)
    rgb = out[..., :3]
    depth = out[..., 3]
    mask = np.round(out[..., 4]).astype(np.int32)

    object_masks = []
    for i in range(1, mask.max() + 1):
        object_masks.append(mask==i)

    return {
        'I': np.uint8(rgb),
        'd': depth,
        'M': np.stack(object_masks) if len(object_masks) > 0 else [],
        'nM': len(object_masks),
        'K': renderer.normalized_projection_matrix(),
        'R': R,
        't': T[:, None]
    }

def render_multiview(env, object_mask=None,
                     center=(0.5, 0.5, 0.5),
                     theta=0.8,
                     phis=None,
                     radius=1.1, **kwargs):
    if phis is None:
        assert 'N_view' in kwargs, "please provide N_view if phis is not defined"
        N_view = kwargs.pop('N_view')
        phis = [np.pi * 2 /N_view * i for i in range(N_view)]
    if isinstance(theta, float):
        theta = (theta,) * len(phis)

    particles = env.simulator.get_x(0)[:, [0, 2, 1]]

    if object_mask is None:
        cluster = sklearn.cluster.DBSCAN(eps=1./64, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto',
                               leaf_size=30, p=None, n_jobs=None)
        cluster.fit(particles)
        object_mask = np.float32(cluster.labels_) + 1.
        # print('num objects:', object_mask.max() + 1)
        # print(object_mask.shape)

    outs = []
    for theta, phi in zip(theta, phis):
        cam = lookat(center, theta, phi, radius)
        outs.append(render_everything(*cam, env, object_mask, **kwargs))
        K = outs[-1]['K'] # (3x4) turns a world to -1, 1 depth

        kinv = np.eye(4)
        kinv[:3, :4] = K
        kinv = np.linalg.pinv(kinv)[:3, :4]
        origin = kinv[:3, 3]
        dists = np.linalg.norm(particles - origin, axis=1)
        outs[-1]['KBounds'] = np.array([dists.min(), dists.max()])
        # xyz = np.concatenate((particles, np.ones_like(particles[..., :1])), 1) @ P.T
        # x = xyz[:, 0] / xyz[:, 2]
        # y = xyz[:, 1] / xyz[:, 2]
        # print(x.min(), x.max())
        # print(y.min(), y.max())
    return outs

def build_hdf5(fileName, B, V, nA, maxM, C=3, H=512, W=512, override=False):
    import h5py
    fileName += '.hdf5'
    if os.path.exists(fileName) and override:
        os.system(f"rm -rf {fileName}")
    hdf5File = h5py.File(fileName, mode='w')
    hdf5File.create_dataset("I", (B, V, C, H, W), np.uint8)    # B, V, C, H, W images
    hdf5File.create_dataset("M", (B, V, maxM, H, W), np.uint8)    # B, V, M, H, W masks with M maximum number of objects in dataset
    hdf5File.create_dataset("nM", (B, V, 1), np.int64)         # B, V, 1 number of objects in each view
    hdf5File.create_dataset("K", (B, V, 3, 4), np.float32)     # B, V, 3, 4 camera projection matrix, including intrinsics and extrinsics
    # such that x_c = P\bar{x}_w
    hdf5File.create_dataset("R", (B, V, 3, 3), np.float32)     # B, V, 3, 3 camera frame to world rotation matrix
    hdf5File.create_dataset("t", (B, V, 3, 1), np.float32)     # B, V, 3, 1 camera frame to world translation
    # such that x_c = Rx_w + t, with x_c in camera coord sys and x_w in world
    hdf5File.create_dataset("KBounds", (B, V, 2), np.float32)  # B, V, 2
    hdf5File.create_dataset("a", (B, nA, 1), np.float32)       # B, nA, 1 action
    return hdf5File



def write_h5(hdf5File, idx, data, action=None):
    def extract(key): return [i[key] for i in data]
    hdf5File["I"][idx] = np.stack([i.transpose(2, 0, 1) for i in extract('I')])
    for v, d in enumerate(data):
        hdf5File['M'][idx, v, :len(d['M'])] = d['M']

    hdf5File["nM"][idx] = np.array(extract('nM'))[:, None]
    hdf5File["K"][idx] = np.array(extract('K')) # V, 3, 4
    hdf5File["R"][idx] = np.array(extract('R'))
    hdf5File["t"][idx] = np.array(extract('t'))
    hdf5File["KBounds"][idx] = np.array(extract('KBounds'))#np.array(extract('KBounds'))

    if action is not None:
        hdf5File["a"][idx] = np.array(action)[:, None]