import numpy as np
import torch

from plb.algorithms.ppo.ppo import utils
from plb.algorithms.ppo.ppo.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, eval_envs, seed, num_processes, eval_log_dir,
             device):
    #eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
    #                          None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    eval_episode_ious = []
    eval_episode_last_ious = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)
    
    episodes = 0
    episodes_step = 0

    total_reward = 0
    total_iou = 0
    total_last_iou = 0
    while episodes < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
        
        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)
        total_reward += infos[0]['reward']
        total_iou += infos[0]['iou']

        if done[0]:
            total_last_iou = infos[0]['iou']
            episodes += 1

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

    #eval_envs.close()
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(episodes, total_reward/episodes))
    return total_reward/10, total_iou/10, total_last_iou/10
