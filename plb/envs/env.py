import gym
from gym.spaces import Box
import os
import yaml
import numpy as np
from ..config import load
from yacs.config import CfgNode
from .utils import merge_lists

PATH = os.path.dirname(os.path.abspath(__file__))


class PlasticineEnv(gym.Env):
    def __init__(self, cfg_path, version, nn=False):
        from ..engine.taichi_env import TaichiEnv
        self.cfg_path = cfg_path
        cfg = self.load_varaints(cfg_path, version)
        self.taichi_env = TaichiEnv(cfg, nn)  # build taichi environment
        self.taichi_env.initialize()
        self.cfg = cfg.ENV
        self.taichi_env.set_copy(True)
        self._init_state = self.taichi_env.get_state()
        self._n_observed_particles = self.cfg.n_observed_particles
        self.taichi_env.simulator.set_obs_num(self._n_observed_particles)

        obs = self.reset()
        self.observation_space = Box(-np.inf, np.inf, obs.shape)
        self.action_space = Box(-1, 1,
                                (self.taichi_env.primitives.action_dim,))

    def reset(self):
        self.taichi_env.set_state(**self._init_state)
        self._recorded_actions = []
        return self._get_obs()

    def _get_obs(self):
        # TODO: check if _n_observed_particles should be passed as argument
        # step_size = self.simulator.n_particles // self._n_observed_particles
        obs = self.taichi_env.get_obs(self._n_observed_particles)
        return obs

    def step(self, action):
        self.taichi_env.step(action)
        loss_info = self.taichi_env.compute_loss()

        self._recorded_actions.append(action)
        obs = self._get_obs()
        r = loss_info['reward']
        if np.isnan(obs).any() or np.isnan(r):
            if np.isnan(r):
                print('nan in r')
            import pickle
            import datetime
            with open(f'{self.cfg_path}_nan_action_{str(datetime.datetime.now())}', 'wb') as f:
                pickle.dump(self._recorded_actions, f)
            raise Exception("NaN..")
        return obs, r, False, loss_info

    def render(self, mode='human'):
        return self.taichi_env.render(mode)

    @classmethod
    def load_varaints(self, cfg_path, version):
        assert version >= 1
        cfg_path = os.path.join(PATH, cfg_path)
        cfg = load(cfg_path)
        variants = cfg.VARIANTS[version - 1]

        new_cfg = CfgNode(new_allowed=True)
        new_cfg = new_cfg._load_cfg_from_yaml_str(yaml.safe_dump(variants))
        new_cfg.defrost()
        if 'PRIMITIVES' in new_cfg:
            new_cfg.PRIMITIVES = merge_lists(
                cfg.PRIMITIVES, new_cfg.PRIMITIVES)
        if 'SHAPES' in new_cfg:
            new_cfg.SHAPES = merge_lists(cfg.SHAPES, new_cfg.SHAPES)
        cfg.merge_from_other_cfg(new_cfg)

        cfg.defrost()
        # set target path id according to version
        name = list(cfg.ENV.loss.target_path)
        name[-5] = str(version)
        cfg.ENV.loss.target_path = os.path.join(PATH, '../', ''.join(name))
        cfg.VARIANTS = None
        cfg.freeze()

        return cfg
