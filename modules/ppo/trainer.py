import os
import time

from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
from collections import deque, namedtuple

from modules.ppo.agent import PPOAgent, Normalize

Transition = namedtuple('Transition', ['state', 'aciton', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])

hyper_params = {
    "seed": 1024,  # which seed to use
    "env": "Ant",  # name of the game
    "batch-size": 64,  # number of transitions to optimize at the same time
    "discount-factor": 0.99,  # discount factor
    "lambda": 0.95,  # lambda factor
    "actor_lr": 3e-4,  # learning rate for actor
    "critic_lr": 3e-4,  # learning rate for critic
    "clip_eps": 0.2,  # clip epsilon
    "update_freq": 2048,  # update frequency
    "epoch": 4000,  # total number of epochs to train the agent
    "eval_every": 10,  # evaluate every eval_every epochs
}


class Trainer(object):
    def __init__(self, env, agent, normal, params):
        self.env = env
        self.agent = agent
        self.params = params
        self.normal = normal
        self.epoch = params["epoch"]
        self.update_freq = params["update_freq"]
        self.eval_every = params["eval_every"]

        self.reward_list = []
        self.evaluation = []

        self.log_path = os.path.join(hyper_params["output-dir"], f"log.txt")
        log_file = open(self.log_path, "w")
        log_file.write(f"Model: PPO\nEnvironment: {hyper_params['env']}\n")
        log_file.write(f"{'=' * 40}\n")
        log_file.close()

    def train(self):
        env = self.env
        agent = self.agent
        episodes = 0
        self.start_time = time.time()

        for iter in tqdm(range(self.epoch)):
            scores = []
            steps = 0
            while steps < self.update_freq:
                state = self.normal(env.reset())
                score = 0
                for _ in range(10000):
                    steps += 1
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = self.normal(next_state)

                    agent.memory.add(state, action, reward, done)
                    score += reward
                    state = next_state
                    if done:
                        break
                episodes += 1
                scores.append(score)
                self.reward_list.append(score)
                if episodes % 500 == 0:
                    torch.save({
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict(),
                        'norm': [self.normal.mean, self.normal.std, self.normal.stdd, self.normal.n]
                    }, os.path.join(hyper_params["output-dir"], "checkpoints", f"checkpoint_{episodes}.pt"))
            score_avg = np.mean(scores)
            time_interval = time.time() - self.start_time
            print(f'Episode: {episodes} | Average Reward: {score_avg:.3f} | Time: {time_interval:.2f}')
            log_file = open(self.log_path, "a")
            log_file.write(f'Episode: {episodes} | Average Reward: {score_avg:.3f} | Time: {time_interval:.2f}\n')
            log_file.close()
            agent.learn()
            if (iter + 1) % self.eval_every == 0:
                self.evaluation.append(self.eval())

        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'norm': [self.normal.mean, self.normal.std, self.normal.stdd, self.normal.n]
        }, os.path.join(hyper_params["output-dir"], "checkpoints", f"checkpoint_last.pt"))

    def eval(self, eval_episodes=10):
        eval_env = gym.make(self.params["env"] + "-v2")

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = self.normal(eval_env.reset()), False
            while not done:
                action = self.agent.act(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                state = self.normal(state)
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
        plt.plot(range(0, self.epoch + 1, 10)[1:], self.evaluation)
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title('Reward vs. Epoch for PPO on {}'.format(env_name))
        plt.savefig(os.path.join(hyper_params["output-dir"], f"Rewards_PPO_{env_name}.pdf"), bbox_inches="tight")
        print(f'Save results to {os.path.join(hyper_params["output-dir"], f"Rewards_PPO_{env_name}.pdf")}')
        plt.close()


def train_ppo(args):
    hyper_params["env"] = args.env_name.removesuffix("-v2")
    hyper_params["epoch"] = args.epoch
    hyper_params["batch-size"] = args.batch_size
    hyper_params["device"] = args.device
    hyper_params["output-dir"] = args.output_dir

    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = PPOAgent(state_size=state_size, action_size=action_size, batch_size=hyper_params["batch-size"], actor_lr=hyper_params["actor_lr"], critic_lr=hyper_params["critic_lr"], gamma=hyper_params["discount-factor"],
        _lambda=hyper_params["lambda"], clip_eps=hyper_params["clip_eps"], device=args.device, )
    normal = Normalize(state_size)

    trainer = Trainer(env, agent, normal, hyper_params)
    trainer.train()
    trainer.plot()
