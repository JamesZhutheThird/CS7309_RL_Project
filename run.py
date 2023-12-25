import os
import argparse
import random
import numpy as np
import torch
import json

atari_env = ["BoxingNoFrameskip-v4", "BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "VideoPinball-ramNoFrameskip-v4"]
mujoco_env = ["Ant-v2", "HalfCheetah-v2","Hopper-v2", "Humanoid-v2" ]

def main():
    parser = argparse.ArgumentParser(description="Train rl model on gym environments")
    parser.add_argument("--env_name", type=str, choices=atari_env+mujoco_env, help="The environment to train your rl models.")
    parser.add_argument('--model', type=str, choices=["DQN","PPO","TD3"], help='The reinforcement learning model.')
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=1e7)
    parser.add_argument("--epoch", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="The device to train the agent. By default, we use gpu if available.")
    parser.add_argument("--n_envs", type=int, default=8)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Using device: ", args.device)
    print("Using environment: ", args.env_name)
    print("Using model: ", args.model)

    args.output_dir = f"results/{args.model}/{args.env_name}/"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+"checkpoints", exist_ok=True)

    
    if args.model == "DQN" and args.env_name in atari_env:
        from modules.dqn.trainer import train_dqn
        train_dqn(args)
    elif args.model == "PPO" and args.env_name in mujoco_env:
        from modules.ppo.trainer import train_ppo
        train_ppo(args)
    elif args.model == "TD3" and args.env_name in mujoco_env:
        from modules.td3.trainer import train_td3
        train_td3(args)

    print('Finish running.')

if __name__ == '__main__':
    main()
