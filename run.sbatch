#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --partition=64c512g
#SBATCH --gres=gpu:0
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -a 0,4
#SBATCH --output=./slurm_logs/rl-%A-%a.out
#SBATCH --error=./slurm_logs/rl-%A-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=JamesZhutheThird@sjtu.edu.cn

# run each job using case
case $SLURM_ARRAY_TASK_ID in
0)
    python run.py --model DQN --env_name VideoPinball-ramNoFrameskip-v4 --num_steps 10000000 --n_envs 16
    ;;
1)
    python run.py --model DQN --env_name BreakoutNoFrameskip-v4 --num_steps 10000000 --n_envs 16
    ;;
2)
    python run.py --model DQN --env_name PongNoFrameskip-v4 --num_steps 10000000 --n_envs 16
    ;;
3)
    python run.py --model DQN --env_name BoxingNoFrameskip-v4 --num_steps 10000000 --n_envs 16
    ;;
4)
    python run.py --model PPO --env_name Hopper-v2 --epoch 5000 --n_envs 16
    ;;
5)
    python run.py --model PPO --env_name Humanoid-v2 --epoch 5000 --n_envs 16
    ;;
6)
    python run.py --model PPO --env_name HalfCheetah-v2 --epoch 5000 --n_envs 16
    ;;
7)
    python run.py --model PPO --env_name Ant-v2 --epoch 5000 --n_envs 16
    ;;
8)
    python run.py --model TD3 --env_name Hopper-v2 --num_steps 2000000 --n_envs 16
    ;;
9)
    python run.py --model TD3 --env_name Humanoid-v2 --num_steps 2000000 --n_envs 16
    ;;
10)
    python run.py --model TD3 --env_name HalfCheetah-v2 --num_steps 2000000 --n_envs 16
    ;;
11)
    python run.py --model TD3 --env_name Ant-v2 --num_steps 2000000 --n_envs 16
    ;;
esac
