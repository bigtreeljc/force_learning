import gym
import numpy as np
from itertools import count
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

class test(unittest.TestCase):
  def test1(self):
    test1("CartPole-v0")

  def test2(self):
    test1("CartPole-v0")


def test1(env_name):
  default_gamma = 0.99
  default_seed=543
  default_render=True
  default_log_interval=10
  seed = 0
  env = gym.make(env_name)
  env.seed(seed)

  n_states = env.observation_space.shape[0]
  n_actions = env.action_space.n

  running_reward = 10
  pg = reinforce(n_states, n_actions)

  for i_episode in count(1):
    state = env.reset()
    for t in range(10000):  # Don't infinite loop while learning
      action = pg.select_action(state)
      state, reward, done, _ = env.step(action)
      if default_render:
        env.render()
      pg.rewards.append(reward)
      if done:
        break

    running_reward = running_reward * 0.99 + t * 0.01
    pg.finish_episode()
    if i_episode % pg.log_interval == 0:
      print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))
    if running_reward > env.spec.reward_threshold:
      print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
      break

class policy_net(nn.Module):
  def __init__(self, n_states, n_actions, n_hidden=128):
    super(policy_net, self).__init__()
    self.affine1 = nn.Linear(n_states, n_hidden)
    self.affine2 = nn.Linear(n_hidden, n_actions)

  def forward(self, x):
    x = F.relu(self.affine1(x))
    action_scores = self.affine2(x)
    return F.softmax(action_scores, dim=1)

# it is actually use self-defined function
class reinforce:
  def __init__(self, n_states, n_actions, 
          seed=543,
          log_interval=0.1,
          optimizer_func=optim.Adam,
          gamma=0.99):
    self.policy = policy_net(n_states, n_actions)
    self.optimizer = optimizer_func(self.policy.parameters(), lr=1e-2)
    self.eps = np.finfo(np.float32).eps.item()
    self.gamma = gamma
    self.log_interval = log_interval
    self.saved_log_probs = []
    self.rewards = []

  def select_action(self, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = self.policy(state)
    m = Categorical(probs)
    action = m.sample()
    self.saved_log_probs.append(m.log_prob(action))
    return action.item()

  def finish_episode(self):
    R = 0
    policy_loss = []
    rewards = []
    for r in self.rewards[::-1]:
        R = r + self.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
    for log_prob, reward in zip(self.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    self.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    self.optimizer.step()
    del self.rewards[:]
    del self.saved_log_probs[:]

# TODO
class reinforce_with_baseline:
  def __init__(self, n_states, n_actions, 
          seed=543,
          log_interval=0.1,
          optimizer_func=optim.Adam,
          gamma=0.99):
    self.policy = policy_net(n_states, n_actions, n_hidden=128)
    self.s_value_func = Policy(n_states, 1, n_hidden=128)

    self.optimizer = optimizer_func(self.policy.parameters(), lr=1e-2)
    self.eps = np.finfo(np.float32).eps.item()
    self.gamma = gamma
    self.log_interval = log_interval
    self.saved_log_probs = []
    self.rewards = []
    self.Gt = []
    self.sigma = []

  def select_action(self, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = self.policy(state)
    m = Categorical(probs)
    action = m.sample()
    self.saved_log_probs.append(m.log_prob(action))
    return action.item()

  def finish_episode(self):
    R = 0
    policy_loss = []
    rewards = []
    for r in self.rewards[::-1]:
        R = r + self.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
    for log_prob, reward in zip(self.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    self.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    self.optimizer.step()
    del self.rewards[:]
    del self.saved_log_probs[:]


