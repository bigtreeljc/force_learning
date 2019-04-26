import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import unittest
import pdb

import numpy as np
import matplotlib.pyplot as plt
import copy

from time import sleep

class test(unittest.TestCase):
  def test1(self):
    test1("CartPole-v0", reward_function=reward_func)

  def test2(self):
    test1("MountainCar-v0")

def test1(env_name, reward_function=None):
  memory_capacity = 2000

  env = gym.make(env_name)
  n_actions = env.action_space.n
  n_states = env.observation_space.shape[0]
  # pdb.set_trace()

  dqn = deep_q_network(n_states, n_actions)
  episodes = 400
  print("Collecting Experience....")
  reward_list = []

  plt.ion()
  fig, ax = plt.subplots()

  for i in range(episodes):
    state = env.reset()
    ep_reward = 0
    while True:
      env.render()
      action = dqn.choose_action(state)
      next_state, reward, done, info = env.step(action)

      if reward_function is not None:
        x, x_dot, theta, theta_dot = next_state
        reward = reward_function(env, x, x_dot, theta, theta_dot)

      dqn.store_transition(state, action, reward, next_state)
      ep_reward += reward

      if dqn.memory_counter >= memory_capacity:
        dqn.learn()
        if done:
          print(f"episode: {i} , the episode reward is {round(ep_reward, 3)}")

      if done:
        # sleep(1)
        break

      state = next_state 

    r = copy.copy(reward)
    reward_list.append(r)
    ax.set_xlim(0,300)
    #ax.cla()
    ax.plot(reward_list, 'g-', label='total_loss')
    plt.pause(0.001)

# disable video rendering
# env = wrappers.Monitor(env, aigym_path, video_callable=False ,force=True)
def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

class dqn_net(nn.Module):
  def __init__(self, n_states, n_actions):
    super(dqn_net, self).__init__()
    self.fc1 = nn.Linear(n_states, 50)
    self.fc1.weight.data.normal_(0, 0.1)
    self.fc2 = nn.Linear(50, 30)
    self.fc2.weight.data.normal_(0, 0.1)
    self.out = nn.Linear(30, n_actions)
    self.out.weight.data.normal_(0, 0.1)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    action_prob = self.out(x)
    return action_prob

class deep_q_network:
  """docstring for DQN"""
  def __init__(self, n_states, n_actions, 
          net_cls=dqn_net, 
          memory_capacity=2000, 
          batch_size=64,
          learning_rate=1e-2,
          loss_func=nn.MSELoss(),
          exploration_ratio=0.9,
          reward_decay = 0.90,
          update_every=100):
    self.eval_net = net_cls(n_states, n_actions)
    self.target_net = net_cls(n_states, n_actions)
    self.learn_step_counter = 0
    self.memory_counter = 0
    self.memory = np.zeros((memory_capacity, n_states*2+2))
    # why the NUM_STATE*2 +2
    # When we store the memory, we put the state, action, reward and next_state in the memory
    # here reward and action is a number, state is a ndarray
    self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
    self.loss = loss_func
    self.batch_size = batch_size
    self.update_every = update_every
    self.memory_capacity = memory_capacity
    self.n_states = n_states
    self.n_actions = n_actions
    self.exploration_ratio = exploration_ratio
    self.gamma = reward_decay

  def store_transition(self, state, action, reward, next_state):
    transition = np.hstack((state, [action, reward], next_state))
    index = self.memory_counter % self.memory_capacity
    self.memory[index, :] = transition
    self.memory_counter += 1

  def choose_action(self, state, play=False):
    state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
    if np.random.randn() <= self.exploration_ratio or play:# greedy policy
        action_value = self.eval_net.forward(state)
        action = torch.max(action_value, 1)[1].data.numpy()
        action = action[0] # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
    else: # random policy
        action = np.random.randint(0, self.n_actions)
        action = action #if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
    return action

  # TODO
  def play(self):
    pass

  # TODO
  def persist(self):
    pass

  def learn(self):
    #update the parameters
    if self.learn_step_counter % self.update_every == 0:
        self.target_net.load_state_dict(
                self.eval_net.state_dict())
    self.learn_step_counter+=1

    #sample batch from memory
    sample_index = np.random.choice(self.memory_capacity, self.batch_size)
    batch_memory = self.memory[sample_index, :]
    batch_state = torch.FloatTensor(
            batch_memory[:, :self.n_states])
    batch_action = torch.LongTensor(
            batch_memory[:, self.n_states:self.n_states+1].astype(int))
    batch_reward = torch.FloatTensor(
            batch_memory[:, self.n_states+1:self.n_states+2])
    batch_next_state = torch.FloatTensor(
            batch_memory[:,-self.n_states:])

    #q_eval
    q_eval = self.eval_net(batch_state).gather(1, batch_action)
    q_next = self.target_net(batch_next_state).detach()
    q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
    loss = self.loss(q_eval, q_target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

