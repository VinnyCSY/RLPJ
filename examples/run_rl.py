''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )
    eval_env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        if args.load_checkpoint_path != "":
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                pattern_shape=env.pattern_shape[0],
                use_pattern=args.pattern,
                mlp_layers=[64,64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        if args.load_checkpoint_path != "":
            agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                pattern_shape=env.pattern_shape[0],
                use_pattern=args.pattern,
                hidden_layers_sizes=[64,64],
                q_mlp_layers=[64,64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    elif args.algorithm == 'a2c':
        from rlcard.agents import A2CAgent
        if args.load_checkpoint_path != "":
            agent = A2CAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = A2CAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                pattern_shape=env.pattern_shape[0],
                use_pattern=args.pattern,
                actor_mlp_layers=[64,64],
                critic_mlp_layers=[64,64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every,
            )

    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    if args.self_play:
        env.set_agents([agent] * env.num_players)
    else:
        env.set_agents(agents)
    eval_env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            if args.self_play:
                for traj in trajectories:
                    for ts in traj:
                        agent.feed(ts)
            else:
                for ts in trajectories[0]:
                    agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'indianpoker',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )

    parser.add_argument(
        '--self_play',
        default=False,
        action='store_true',
    )

    parser.add_argument(
        '--pattern',
        default=False,
        action='store_true',
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
            'a2c',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_dqn_result/',
    )
    
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

