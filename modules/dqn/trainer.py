import pdb

from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gym
import numpy as np
import torch
from collections import deque
import time
import json
import os

from modules.dqn.agent import DQNAgent, ReplayBuffer
from modules.dqn.wrapper.atari_wrapper import SubprocVecEnv, wrap_cover

hyper_params = {
    "seed": 1024,  # which seed to use
    "env": "Pong",  # name of the game
    "n_envs": 16,  # number of environments
    "replay-buffer-size": int(1e5),  # replay buffer size
    "learning-rate": 1e-4,  # learning rate for Adam optimizer
    "discount-factor": 0.99,  # discount factor
    "num-steps": int(1e7),  # total number of steps to run the environment
    "batch-size": 32,  # number of transitions to optimize at the same time
    "learning-starts": 1000,  # number of steps before learning starts
    "learning-freq": 4,  # number of iterations between every optimization step
    "use-double-dqn": False,  # use double deep Q-learning
    "target-update-freq": 1,  # number of iterations between every target network update
    "eps-start": 1,  # e-greedy start threshold
    "eps-end": 0.01,  # e-greedy end threshold
    "eps-fraction": 0.1,  # fraction of num-steps
}


class Trainer(object):
    def __init__(self, env, agent, params) :
        self.env = env
        self.agent = agent
        self.params = params
        self.reward_list = []
        self.epinfobuf = deque(maxlen=100)

        self.train_steps = params["num-steps"]
        self.n_envs = params["n_envs"]
        self.eps_timesteps = params["eps-fraction"] * float(params["num-steps"])
        self.eps_end = params["eps-end"]
        self.eps_start = params["eps-start"]
        self.learning_start = params["learning-starts"]
        self.learning_freq = params["learning-freq"]

        self.log_path = os.path.join(hyper_params["output-dir"], f"log.txt")
        log_file= open(self.log_path,"w")
        log_file.write(f"Model: DQN\nEnvironment: {hyper_params['env']}\n")
        log_file.write(f"{'='*40}\n")
        log_file.close()

    def train(self):
        env = self.env
        agent = self.agent
        state = env.reset()
        # episode_reward = 0
        self.start_time = time.time()
        pbar = tqdm(range(self.train_steps))
        for t in range(1, self.train_steps // self.n_envs + 1):
            fraction = min(1.0, float(agent.step) / self.eps_timesteps)
            eps_threshold = self.eps_start + fraction * (self.eps_end - self.eps_start)

            action = agent.act(state, eps_threshold)

            next_state, reward, done, infos = env.step(action)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: self.epinfobuf.append(maybeepinfo)

            for i in range(self.n_envs):
                pbar.update(1)
                agent.step += 1
                agent.memory.add(state[i], action[i], reward[i], next_state[i], float(done[i]))
            state = next_state

            if agent.step > self.learning_start and agent.step % hyper_params["learning-freq"] == 0:
                agent.optimize_td_loss()

            if t % 1000 == 0:
                time_interval = time.time() - self.start_time
                avg_reward = np.mean([epinfo['r'] for epinfo in self.epinfobuf])
                self.reward_list.append(avg_reward)
                print(f"Step: {agent.step} | Average Reward: {avg_reward:.3f} | Time: {time_interval:.2f}")
                log_file = open(self.log_path, "a")
                log_file.write(f"Step: {agent.step} | Average Reward: {avg_reward:.3f} | Time: {time_interval:.2f}\n")
                log_file.close()
            if t % 50000 == 0:
                self.agent.save_model(os.path.join(hyper_params["output-dir"],"checkpoints", f"checkpoint_{t*self.n_envs}.pt"))

        self.agent.save_model(os.path.join(hyper_params["output-dir"], "checkpoints", f"checkpoint_last.pt"))

    def plot(self):
        env_name = self.params["env"]
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, self.train_steps, self.n_envs * 1000)[1:], self.reward_list)
        plt.xlabel("Training Steps")
        plt.ylabel("Average Reward")
        plt.title(f"Reward vs. Steps for DQN on {env_name}")
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:,.0f}".format(x/1000) + "k"))
        plt.savefig(os.path.join(hyper_params["output-dir"], f"Rewards_DQN_{env_name}.pdf"), bbox_inches="tight")
        print(f'Save results to {os.path.join(hyper_params["output-dir"], f"Rewards_DQN_{env_name}.pdf")}')
        plt.close()


def train_dqn(args):
    hyper_params["seed"] = args.seed
    hyper_params["env"] = args.env_name.removesuffix("NoFrameskip-v4")
    hyper_params["n_envs"] = args.n_envs
    hyper_params["batch-size"] = args.batch_size
    hyper_params["num-steps"] = int(args.num_steps)
    hyper_params["learning-rate"] = args.lr
    hyper_params["device"] = args.device
    hyper_params["output-dir"] = args.output_dir

    # save hyper_params
    with open(args.output_dir + "hyper_params.json", "w") as f:
        json.dump(hyper_params, f, indent=4)

    env = SubprocVecEnv([wrap_cover(args.env_name) for i in range(args.n_envs)])

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    agent = DQNAgent(env.observation_space, env.action_space, replay_buffer, use_double_dqn=hyper_params["use-double-dqn"], lr=args.lr, batch_size=args.batch_size, target_update_freq=hyper_params["target-update-freq"],
        gamma=hyper_params['discount-factor'], device=args.device, )

    trainer = Trainer(env, agent, hyper_params)
    trainer.train()
    trainer.plot()
