import pdb
import time

import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import json
import os

import matplotlib.ticker as ticker
from tqdm import tqdm

from modules.td3.agent import TD3Agent, ReplayBuffer

hyper_params = {
    "seed": 1024,  # which seed to use
    "env": "Ant",  # name of the game
    "batch-size": 64,  # number of transitions to optimize at the same time
    "discount-factor": 0.99,  # discount factor
    "num-steps": int(1e6),  # total number of steps to run the environment
    "start_timesteps": int(25000),  # Time steps initial random policy is used
    "expl_noise": 0.1,  # Std of Gaussian exploration noise
    "eval_every": int(1e3),  # How often (time steps) we evaluate
    "replay-buffer-size": int(1e6),  # size of replay buffer
    "actor_lr": 3e-4,  # learning rate for actor
    "critic_lr": 3e-4,  # learning rate for critic
    "tau": 0.005,  # tau factor
    "policy_noise": 0.2,  # policy noise
    "noise_clip": 0.5,  # noise clip
    "policy_freq": 2,  # policy frequency
}


class Trainer(object):
    def __init__(self, env, agent, params):
        self.env = env
        self.agent = agent
        self.params = params

        self.train_steps = int(params["num-steps"])
        self.start_timesteps = params["start_timesteps"]
        self.expl_noise = params["expl_noise"]
        self.eval_every = params["eval_every"]
        self.evaluations = []

        self.log_path = os.path.join(hyper_params["output-dir"], f"log.txt")
        log_file = open(self.log_path, "w")
        log_file.write(f"Model: TD3\nEnvironment: {hyper_params['env']}\n")
        log_file.write(f"{'=' * 40}\n")
        log_file.close()

    def train(self):
        env = self.env
        agent = self.agent
        self.start_time = time.time()

        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        state, done = env.reset(), False

        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in tqdm(range(self.train_steps)):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < self.start_timesteps:
                action = env.action_space.sample()
            else:
                action = (agent.select_action(np.array(state)) + np.random.normal(0, max_action * self.expl_noise, size=action_dim)).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            agent.memory.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= self.start_timesteps:
                agent.train()

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t + 1} | Episode Num: {episode_num + 1} | Episode T: {episode_timesteps} | Reward: {episode_reward:.3f}")
                log_file = open(self.log_path, "a")
                log_file.write(f"Total T: {t + 1} | Episode Num: {episode_num + 1} | Episode T: {episode_timesteps} | Reward: {episode_reward:.3f}\n")
                log_file.close()

                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                # Evaluate episode
            if (t+1) % self.eval_every == 0:
                self.evaluations.append(self.eval())

            if (t + 1) % (self.eval_every*100) == 0:
                agent.save(os.path.join(hyper_params["output-dir"], "checkpoints", f"checkpoint_{t + 1}.pt"))

        agent.save(os.path.join(hyper_params["output-dir"], "checkpoints", f"checkpoint_last.pt"))

    def eval(self, eval_episodes=10):
        eval_env = gym.make(self.params["env"] + "-v2")

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = self.agent.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        log_file = open(self.log_path, "a")
        log_file.write(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}\n")
        log_file.close()
        return avg_reward

    def plot(self):
        env_name = self.params["env"]
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(0, self.train_steps+1, self.eval_every)[1:], self.evaluations)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title('Reward vs. Steps for TD3 on {}'.format(env_name))
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:,.0f}".format(x/1000) + "k"))
        plt.savefig(os.path.join(hyper_params["output-dir"], f"Rewards_TD3_{env_name}.pdf"), bbox_inches="tight")
        print(f'Save results to {os.path.join(hyper_params["output-dir"], f"Rewards_TD3_{env_name}.pdf")}')
        plt.close()

def train_td3(args):
    hyper_params["env"] = args.env_name.removesuffix("-v2")
    hyper_params["num-steps"] = args.num_steps
    hyper_params["batch-size"] = args.batch_size
    hyper_params["device"] = args.device
    hyper_params["output-dir"] = args.output_dir

    env = gym.make(args.env_name)
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    agent = TD3Agent(env, replay_buffer, batch_size=args.batch_size, actor_lr=hyper_params["actor_lr"], critic_lr=hyper_params["critic_lr"], discount=hyper_params["discount-factor"], tau=hyper_params["tau"],
        policy_noise=hyper_params["policy_noise"], noise_clip=hyper_params["noise_clip"], policy_freq=hyper_params["policy_freq"])

    trainer = Trainer(env, agent, hyper_params)
    trainer.train()
    trainer.plot()
