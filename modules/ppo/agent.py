from gym import spaces
import numpy as np
import random

from modules.ppo.model import Actor, Critic

import torch

from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReplayBuffer(object):
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param done: whether the episode terminated
        """
        data = (state, action, reward, done)

        self._storage.append(data)

    def _encode_sample(self, indices):
        states, actions, rewards, dones = [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones)

    def sample(self):
        """
        Randomly sample a batch of transitions from the buffer.
        :return: a mini-batch of sampled transitions
        """
        indices = np.arange(0, len(self._storage))
        return self._encode_sample(indices)

    def clear(self):
        del self._storage[:]


class PPOAgent(object):
    def __init__(
        self, 
        state_size, 
        action_size,
        batch_size,
        actor_lr,
        critic_lr,
        gamma,
        _lambda,
        clip_eps,
        device,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self._lambda = _lambda
        self.clip_eps = clip_eps

        self.memory = ReplayBuffer()
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=0.001)
        self.device = device
    
    def get_gae(self, rewards, masks, values):
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + self.gamma * running_returns * masks[t]
            running_tderror = rewards[t] + self.gamma * previous_value * masks[t] - values.data[t]
            running_advants = running_tderror + self.gamma * self._lambda * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def learn(self):
        device = self.device
        
        states, actions, rewards, dones = self.memory.sample()

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        # memory = np.array(memory)
        # actions = np.array(memory[:, 1])
        # print(type(actions))
        # states = torch.Tensor(np.vstack(memory[:, 0])).to(device)
        # actions = torch.from_numpy(np.vstack(memory[:, 1])).to(device)
        # rewards = torch.from_numpy(np.vstack(memory[:, 2])).to(device)
        # masks = torch.from_numpy(np.array(memory[:, 3])).to(device)

        values = self.critic(states)

        returns, advants = self.get_gae(rewards, 1 - dones, values)
        old_mu, old_std = self.actor(states)
        pi = torch.distributions.Normal(old_mu, old_std)
        old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n//self.batch_size):
                b_index = arr[self.batch_size*i:self.batch_size*(i+1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                mu, std = self.actor(b_states)
                pi = torch.distributions.Normal(mu, std)
                new_prob = pi.log_prob(b_actions).sum(1, keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                ratio = torch.exp(new_prob-old_prob)

                surrogate_loss = ratio * b_advants
                values = self.critic(b_states)
                critic_loss = criterion(values, b_returns)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                clipped_loss = ratio * b_advants
                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
        self.memory.clear()
        
    def act(self, state):
        device = self.device
        state = torch.Tensor(state).unsqueeze(0).to(device)
        mu, sigma = self.actor(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample().cpu().numpy()
        action = action[0]
        return action


class Normalize:
    def __init__(self, state_size):
        self.mean = np.zeros((state_size,))
        self.std = np.zeros((state_size, ))
        self.stdd = np.zeros((state_size, ))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean
        x = x - self.mean
        x = x / (self.std + 1e-8)
        x = np.clip(x, -5, +5)
        return x
