import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .replay_buffer import ReplayBuffer
from .utils import RunningMeanStats

class Agent:

    def __init__(self, env, test_env, algo, log_dir, device, num_steps=3000000,
                 batch_size=256, memory_size=1000000,
                 update_interval=1, start_steps=10000, log_interval=10,
                 eval_interval=200, num_eval_episodes=5, seed=0, logger=None):

        # Environment.
        self._env = env
        self._test_env = test_env
        self.logger = logger

        self._env.seed(seed)
        self._test_env.seed(2**31-1-seed)
        
        # Algorithm.
        self._algo = algo

        # Replay buffer with n-step return.
        self._replay_buffer = ReplayBuffer(
            memory_size=memory_size,
            state_shape=self._env.observation_space.shape,
            action_shape=self._env.action_space.shape,
            gamma=self._algo.gamma, nstep=self._algo.nstep)

        # Directory to log.
        self._log_dir = log_dir
        self._model_dir = os.path.join(log_dir, 'model')
        self._summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        if not os.path.exists(self._summary_dir):
            os.makedirs(self._summary_dir)

        self._steps = 0
        self._episodes = 0
        self._train_return = RunningMeanStats(log_interval)

        self._writer = SummaryWriter(log_dir=self._summary_dir)
        self._best_eval_score = -np.inf

        self._device = device
        self._num_steps = num_steps
        self._batch_size = batch_size
        self._update_interval = update_interval
        self._start_steps = start_steps
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._num_eval_episodes = num_eval_episodes

    def run(self):
        self.start_time = time.time()
        while True:
            self.train_episode()
            if self._steps > self._num_steps:
                break
        self._writer.close()
        print('break')

    def train_episode(self):
        self._episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self._env.reset()
        self.logger.reset()

        while (not done):

            if self._start_steps > self._steps:
                action = self._env.action_space.sample()
            else:
                action = self._algo.explore(state)

            next_state, reward, done, info = self._env.step(action)
            self.logger.step(state, action, reward, next_state, done, info)

            # Set done=True only when the agent fails, ignoring done signal
            # if the agent reach time horizons.
            if episode_steps + 1 >= self._env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            self._replay_buffer.append(
                state, action, reward, next_state, masked_done,
                episode_done=done)

            self._steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if self._steps >= self._start_steps:
                # Update online networks.
                if self._steps % self._update_interval == 0:
                    batch = self._replay_buffer.sample(
                        self._batch_size, self._device)
                    self._algo.update_online_networks(batch, self._writer)

                # Update target networks.
                self._algo.update_target_networks()

        # Evaluate.
        if self._episodes % self._eval_interval == 0:
            self.evaluate()
            self._algo.save_models(
                os.path.join(self._model_dir, 'final'))

        # We log running mean of training rewards.
        self._train_return.append(episode_return)

        if self._episodes % self._log_interval == 0:
            self._writer.add_scalar(
                'reward/train', self._train_return.get(), self._steps)

    def evaluate(self):
        total_return = 0.0

        for _ in range(self._num_eval_episodes):
            state = self._test_env.reset()
            episode_return = 0.0
            done = False
            episode_steps = 0

            while (not done):
                action = self._algo.exploit(state)
                next_state, reward, done, info = self._test_env.step(action)
                episode_return += reward
                state = next_state
                episode_steps += 1

            total_return += episode_return

        mean_return = total_return / self._num_eval_episodes
        if mean_return > self._best_eval_score:
            self._best_eval_score = mean_return
            self._algo.save_models(os.path.join(self._model_dir, 'best'))

        self._writer.add_scalar(
            'reward/test', mean_return, self._steps)

        print('-' * 60)
        print(f'Num steps: {self._steps:<5}  '
              f'return: {mean_return:<5.1f}   final: {info["incremental_iou"]}')
        print('-' * 60)

    def __del__(self):
        self._env.close()
        self._test_env.close()
        self._writer.close()
