salloc -p debuga100 -n 1 -N 1 --gres=gpu:1 --cpus-per-task=16 --mem=120G
salloc -p debug64c512g -n 1 -N 1 --gres=gpu:0 --cpus-per-task=16 --mem=120G

cd zcz72/RL
conda activate RL_3.9

python run.py --model DQN --env_name VideoPinball-ramNoFrameskip-v4 --num_steps 10000000 --n_envs 16
python run.py --model DQN --env_name BreakoutNoFrameskip-v4 --num_steps 10000000 --n_envs 16
python run.py --model DQN --env_name PongNoFrameskip-v4 --num_steps 10000000 --n_envs 16
python run.py --model DQN --env_name BoxingNoFrameskip-v4 --num_steps 10000000 --n_envs 16

python run.py --model PPO --env_name Hopper-v2 --epoch 5000 --n_envs 16
python run.py --model PPO --env_name Humanoid-v2 --epoch 5000 --n_envs 16
python run.py --model PPO --env_name HalfCheetah-v2 --epoch 5000 --n_envs 16
python run.py --model PPO --env_name Ant-v2 --epoch 5000 --n_envs 16

python run.py --model TD3 --env_name Hopper-v2 --num_steps 2000000 --n_envs 16
python run.py --model TD3 --env_name Humanoid-v2 --num_steps 2000000 --n_envs 16
python run.py --model TD3 --env_name HalfCheetah-v2 --num_steps 2000000 --n_envs 16
python run.py --model TD3 --env_name Ant-v2 --num_steps 2000000 --n_envs 16