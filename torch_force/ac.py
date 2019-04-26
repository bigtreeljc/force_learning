import matplotlib as mpl
mpl.use('Agg')
import gym, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class test(unittest.TestCase):
  def test1(self):
    test1("CartPole-v0")

  def test2(self):
    test1("CartPole-v0")


def test1(env_name):
  env = gym.make(env_name)
  env = env.unwrapped
  
  env.seed(1)
  torch.manual_seed(1)
  
  state_space = env.observation_space.shape[0]
  action_space = env.action_space.n
  
  #Hyperparameters
  learning_rate = 0.01
  gamma = 0.99
  episodes = 20000
  render = True
  eps = np.finfo(np.float32).eps.item()
  SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
  ac = actor_critic(state_space, action_space)

  running_reward = 10
  live_time = []
  for i_episode in count(episodes):
    state = env.reset()
    for t in count():
      action = ac.select_action(state)
      state, reward, done, info = env.step(action)
      if render: env.render()
      ac.rewards.append(reward)

      if done or t >= 1000:
        break

    running_reward = running_reward * 0.99 + t * 0.01
    live_time.append(t)
    # plot(live_time)
    # if i_episode % 100 == 0:
    #   modelPath = './AC_CartPole_Model/ModelTraing'+str(i_episode)+'Times.pkl'
    #   torch.save(model, modelPath)
    ac.finish_episode()

def test2(env_name):
  max_frames   = 20000
  frame_idx    = 0
  test_rewards = []
  n_states  = envs.observation_space.shape[0]
  n_actions = envs.action_space.n
  
  #Hyper params:
  hidden_size = 256
  lr          = 1e-3
  num_steps   = 5

  env = gym.make(env_name) 
  model = a2c(n_states, n_actions)

  state = envs.reset()
  while frame_idx < max_frames:
    # rollout trajectory
    for _ in range(num_steps):
      dist, value = model.select_action(state)
      action = dist.sample()
      next_state, reward, done, _ = envs.step(action.cpu().numpy())
      model.collect_info(reward, action, done)
      
      state = next_state
      frame_idx += 1
      
      if frame_idx % 100 == 0:
        test_rewards.append(np.mean([test_env() for _ in range(10)]))
        plot(frame_idx, test_rewards)
            
    model.finish_episode(next_state)


class ac_net(nn.Module):
  def __init__(self, n_states, n_actions, n_hidden=32):
    super(ac_net, self).__init__()
    self.fc1 = nn.Linear(n_states, n_hidden)

    self.action_head = nn.Linear(n_hidden, n_actions)
    self.value_head = nn.Linear(n_hidden, 1) # Scalar Value

    # os.makedirs('./AC_CartPole-v0', exist_ok=True)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    action_score = self.action_head(x)
    state_value = self.value_head(x)

    return F.softmax(action_score, dim=-1), state_value

class actor_critic:
  def __init__(self, n_states, n_actions, n_hidden=32,
          learning_rate=0.01,
          gamma=0.99):
    model = ac_net()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    self.gamma = gamma

    self.save_actions = []
    self.rewards = []

  def select_action(self, state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    self.save_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

  def finish_episode(self):
    R = 0
    save_actions = self.save_actions
    policy_loss = []
    value_loss = []
    rewards = []

    for r in self.rewards[::-1]:
      R = r + gamma * R
      rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob , value), r in zip(save_actions, rewards):
      reward = r - value.item()
      policy_loss.append(-log_prob * reward)
      value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del self.rewards[:]
    del self.save_actions[:]

class ac_net1(nn.Module):
  def __init__(self, n_states, n_actions, hidden_size=128, std=0.0):
    super(ac_net1, self).__init__()
    
    self.critic = nn.Sequential(
        nn.Linear(n_states, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )
    
    self.actor = nn.Sequential(
        nn.Linear(n_states, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions),
        nn.Softmax(dim=1),
    )
      
  def forward(self, x):
    value = self.critic(x)
    probs = self.actor(x)
    dist  = Categorical(probs)
    return dist, value

class a2c:
  def __init__(self, n_states, n_actions, hidden_size=256,
          lr=1e-3,
          gamma=0.99,
          optimizer_cls=optim.Adam):
    self.model = ac_net1(n_states, n_actions, hidden_size)
    self.optimizer = optimizer_cls(self.model.parameters, lr=lr)
    self.gamma = gamma
    self.test_rewards = []
    self.log_probs = []
    self.values    = []
    self.rewards   = []
    self.masks     = []
    self.entropy = 0
    
  def select_action(self, state):
    state = torch.FloatTensor(state).to(device)
    dist, value = model(state)
    return dist, value

  def collect_info(self, reward, action, done):
    self.log_prob = dist.log_prob(action)
    self.entropy += dist.entropy().mean()
    self.log_probs.append(log_prob)
    self.values.append(value)
    self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
    self.masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

  def _compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

  def finish_episode(self, next_state):
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = self._compute_returns(next_value, rewards, masks)
    
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)

    advantage = returns - values

    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
