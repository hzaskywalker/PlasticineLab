import os
from datetime import datetime
from gym import make
from fluid.alchemy.rl.discor.algorithm import SAC
from fluid.alchemy.rl.discor.agent import Agent

def test():
    env_id = 'HumanoidStandup-v2'
    env = make(env_id)
    test_env = make(env_id)

    device='cuda:0'
    algo = 'SAC'
    seed = 0

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', env_id, f'{algo}-seed{seed}-{time}')

    algo = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device, seed=seed,
        gamma=0.99,
        nstep=1,
        policy_lr=0.0003,
        q_lr=0.0003,
        entropy_lr=0.0003,
        policy_hidden_units=[256, 256],
        q_hidden_units=[256, 256],
        target_update_coef=0.005,
        log_interval=10)

        
    agent = Agent(
        env=env, test_env=test_env, algo=algo, log_dir=log_dir,
        device=device, seed=seed,
        num_steps=3000000,
        batch_size=256,
        memory_size=1000000,
        update_interval=1,
        start_steps=10000,
        log_interval=10,
        eval_interval=5000,
        num_eval_episodes=5
    )
    agent.run()

if __name__ == '__main__':
    test()
