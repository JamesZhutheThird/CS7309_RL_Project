salloc -p debuga100 -n 1 -N 1 --gres=gpu:1 --cpus-per-task=16 --mem=120G
ssh gpu23
cd zcz72/RL
conda activate RL_3.9
python run.py --model DQN --env_name BoxingNoFrameskip-v4 --num_steps 100000 --n_envs 16

python run.py --model PPO --env_name Hopper-v2 --epoch 100 --n_envs 16
