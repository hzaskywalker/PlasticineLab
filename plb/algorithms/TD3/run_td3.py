import taichi
import numpy as np
import torch
import gym
from plb import envs
import argparse
import os

from plb.algorithms.TD3 import utils
from plb.algorithms.TD3 import TD3


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
tot_eval_episodes = 0
log_path = None

def eval_policy(policy, eval_env, seed, eval_episodes=10):
    global tot_eval_episodes
    #eval_env = gym.make(env_name)
    #eval_env.seed(seed + 100)
    avg_reward = 0.

    ep_reward = 0
    ep_iou = 0
    ep_last_iou = 0

    for _ in range(eval_episodes):
        tot_eval_episodes += 1
        state, done = eval_env.reset(), False
        episode_steps = 0
        while not done:
            episode_steps+=1
            action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            ep_reward += reward
            ep_iou += info['iou']
            avg_reward += reward
        ep_last_iou += info['iou']

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, ep_reward/eval_episodes, ep_iou/eval_episodes, ep_last_iou/eval_episodes


def train_td3(env, path, logger, old_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=2500, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=200, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=500000, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--gamma", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    max_timesteps = old_args.num_steps

    args, _ = parser.parse_known_args()
    args.max_timesteps = max_timesteps

    args.discount = float(args.gamma)

    log_path = path
    os.makedirs(log_path, exist_ok=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    else:
        raise NotImplementedError

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0


    ep_reward = 0
    ep_iou = 0
    ep_last_iou = 0

    logger.reset()
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, info = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        ep_reward += reward
        ep_iou += info['iou']
        ep_last_iou = info['iou']

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        logger.step(state, action, reward, next_state, done, info)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            logger.reset()

            ep_reward=0
            ep_iou = 0
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # Evaluate episode
            if episode_num % args.eval_freq == 0:
                #evaluations.append(
                r1, r2, iou, _last_iou = eval_policy(policy, env, args.seed)
                output = f"Test Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {r1:.3f}" + f" reward: {r2},  iou: {iou},  last_iou: {_last_iou}"
                print(output)

                #np.save(f"./results/{file_name}", evaluations)
                policy.save(log_path)
