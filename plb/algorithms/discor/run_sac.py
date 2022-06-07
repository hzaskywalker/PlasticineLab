from plb.algorithms.discor.algorithm import SAC
from plb.algorithms.discor.agent import Agent
from plb.algorithms.discor.network import GaussianPolicy, TwinnedStateActionFunction


def train(env, path, logger, args):
    device = 'cuda:0'
    test_env = env

    algo = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
        gamma=0.99,
        nstep=1,
        policy_lr=0.0003,
        q_lr=0.0003,
        entropy_lr=0.0003,
        policy_hidden_units=[256, 256],
        q_hidden_units=[256, 256],
        target_update_coef=0.005,
        log_interval=10,
        GaussianPolicy=GaussianPolicy,
        TwinnedStateActionFunction=TwinnedStateActionFunction)

    agent = Agent(
        env=env, test_env=test_env, algo=algo, log_dir=path,
        device=device,
        num_steps=args.num_steps,
        batch_size=256,
        memory_size=1000000,
        update_interval=1,
        start_steps=2500,
        log_interval=10,
        eval_interval=200,
        num_eval_episodes=5,
        logger=logger,
    )
    agent.run()
