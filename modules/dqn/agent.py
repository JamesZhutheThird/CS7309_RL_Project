from gym import spaces
import random
import numpy as np

from modules.dqn.model import DQN
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(object):
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)


class DQNAgent(object):
    """
    DQN running agent
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        target_update_freq,
        gamma,
        device,
    ):
        """
        Initialise the DQN algorithm using the Adam optimizer
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.memory = replay_buffer
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.n_actions = action_space.n

        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network(1.0)
        self.target_network.eval()

        # self.optim = torch.optim.RMSprop(self.policy_network.parameters(), lr=lr)
        self.optim = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        # self.loss_function = nn.SmoothL1Loss()
        self.loss_function = nn.MSELoss()

        self.device = device
        self.step = 0
        self.learn_step_counter = 0

    def optimize_td_loss(self):
        """
        Optimize the TD-error over a single minibatch of transitions
        :return: the loss
        """
        
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network(1e-2)
        device = self.device

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            if self.use_double_dqn:
                _, max_next_action = self.policy_network(next_states).max(1)
                max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(next_states)
                max_next_q_values = torch.max(next_q_values, -1)[0]
                # max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_function(input_q_values, target_q_values)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.learn_step_counter += 1
        return loss.item()

    def update_target_network(self, update_rate):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # self.target_network.load_state_dict(self.policy_network.state_dict())
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                * target_param.data + update_rate*policy_param.data)
        # self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray, eps_threshold: float):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        device = self.device
        if random.random() > eps_threshold:
            state = np.array(state) / 255.0
            # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            state = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                q_values = self.policy_network(state)
                action = torch.argmax(q_values, dim=1).data.cpu().numpy()
                # _, action = q_values.max(1)
        else:
            action = np.random.randint(0, self.n_actions, (state.shape[0]))
        return action
    
    def save_model(self, path):
        torch.save(self.policy_network.state_dict(), path)
    
    def load_model(self, path):
        self.policy_network.load_state_dict(torch.load(path))
        self.update_target_network(1.0)

