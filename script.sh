cd /dssg/home/acct-csyk/csyk-lab/zcz72/RL_debug

conda activate /dssg/home/acct-csyk/csyk-lab/.conda/envs/RL_3.9

python /dssg/home/acct-csyk/csyk-lab/zcz72/RL_debug/run.py --model DQN --env_name VideoPinball-ramNoFrameskip-v4 --num_steps 10000000 --n_envs 16
