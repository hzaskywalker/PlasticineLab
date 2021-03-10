import gym
from .env import PlasticineEnv
from gym import register

ENVS = []
for env_name in ['Move', 'Torus', 'Rope', 'Writer', "Pinch", "Rollingpin", "Chopsticks", "Table", 'TripleMove', 'Assembly']:
    for id in range(5):
        register(
            id = f'{env_name}-v{id+1}',
            entry_point=f"plb.envs.env:PlasticineEnv",
            kwargs={'cfg_path': f"{env_name.lower()}.yml", "version": id+1},
            max_episode_steps=50
        )


def make(env_name, nn=False, sdf_loss=10, density_loss=10, contact_loss=1, soft_contact_loss=False):
    env: PlasticineEnv = gym.make(env_name, nn=nn)
    env.taichi_env.loss.set_weights(sdf=sdf_loss, density=density_loss,
                                    contact=contact_loss, is_soft_contact=soft_contact_loss)
    return env