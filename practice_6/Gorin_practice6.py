# -*- coding: utf-8 -*-
"""ddpg-dce-hometask.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KSvIdKgfZpaawm_JX3emi2Hm55ZwaiUO
"""

# Commented out IPython magic to ensure Python compatibility.
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from copy import deepcopy
from random import sample
import math

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Orstein-Uhlenbeck process


class OUNoise():
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class Net(nn.Module):
    def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim, output_tanh):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, layer1_dim)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim)
        self.layer3 = nn.Linear(layer2_dim, output_dim)
        self.output_tanh = output_tanh
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        hidden = self.layer1(input)
        hidden = self.relu(hidden)
        hidden = self.layer2(hidden)
        hidden = self.relu(hidden)
        output = self.layer3(hidden)

        if self.output_tanh:
            return self.tanh(output)
        else:
            return output


"""# DDPG

Задаем структуру аппроксимаций $\pi^\eta(s)$, $Q^\theta(s,a)$ и начальные вектора параметров $\eta$, $\theta$.

Для каждого эпизода делаем:

   Пока эпизод не закончен делаем:

- Находясь в состоянии $S_t$ совершаем действие

    $$
    A_t = \pi^\eta(S_t) + Noise,
    $$

    получаем награду $R_t$  переходим в состояние $S_{t+1}$. Сохраняем 
    $(S_t,A_t,R_t,D_t,S_{t+1}) \Rightarrow Memory$


- Берем $\{(s_i,a_i,r_i,d_i,s'_i)\}_{i=1}^{n} \leftarrow Memory$, определяем значения

    $$
    y_i = r_i + (1 - d_i) \gamma Q^\theta(s'_i,\pi^\eta(s'_i))
    $$
    функции потерь

    $$
    Loss_1(\theta) = \frac{1}{n}\sum\limits_{i=1}^n \big(y_i - Q^\theta(s_i,a_i)\big)^2,\quad Loss_2(\eta) = -\frac{1}{n}\sum\limits_{i=1}^n Q^\theta(s_i,\pi^\eta(s_i))
    $$

    и обновляем вектор параметров

    $$
    \theta \leftarrow \theta - \alpha \nabla_\theta Loss_1(\theta),\quad \eta \leftarrow \eta - \beta \nabla_\eta Loss_2(\eta),\quad \alpha,\beta > 0
    $$

- Уменьшаем $Noise$

"""


class DDPG():
    def __init__(self, state_dim, action_dim, action_scale, noise_decrease,
                 gamma=0.99, batch_size=64, q_lr=1e-3, pi_lr=1e-4, tau=1e-2, memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.pi_model = Net(self.state_dim, 400, 300,
                            self.action_dim, output_tanh=True)
        self.q_model = Net(self.state_dim + self.action_dim,
                           400, 300, 1, output_tanh=False)
        self.pi_target_model = deepcopy(self.pi_model)
        self.q_target_model = deepcopy(self.q_model)
        self.noise = OUNoise(self.action_dim)
        self.noise_threshold = 1
        self.noise_decrease = noise_decrease
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.q_optimazer = torch.optim.Adam(self.q_model.parameters(), lr=q_lr)
        self.pi_optimazer = torch.optim.Adam(
            self.pi_model.parameters(), lr=pi_lr)
        self.memory = deque(maxlen=memory_size)

    def get_action(self, state):
        pred_action = self.pi_model(torch.FloatTensor(state)).detach().numpy()
        action = self.action_scale * \
            (pred_action + self.noise_threshold * self.noise.sample())
        return np.clip(action, -self.action_scale, self.action_scale)

    def update_target_model(self, target_model, model, optimazer, loss):
        optimazer.zero_grad()
        loss.backward()
        optimazer.step()
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * param.data)

    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) > self.batch_size:
            batch = sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(
                torch.FloatTensor, zip(*batch))
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            pred_next_actions = self.action_scale * \
                self.pi_target_model(next_states)
            next_states_and_pred_next_actions = torch.cat(
                (next_states, pred_next_actions), dim=1)
            targets = rewards + self.gamma * \
                (1 - dones) * self.q_target_model(next_states_and_pred_next_actions)

            states_and_actions = torch.cat((states, actions), dim=1)
            temp = (self.q_model(states_and_actions) - targets.detach())
            q_loss = torch.mean(
                (targets.detach() - self.q_model(states_and_actions)) ** 2)
            self.update_target_model(
                self.q_target_model, self.q_model, self.q_optimazer, q_loss)

            pred_actions = self.action_scale * self.pi_model(states)
            states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
            pi_loss = - torch.mean(self.q_model(states_and_pred_actions))
            self.update_target_model(
                self.pi_target_model, self.pi_model, self.pi_optimazer, pi_loss)

        if self.noise_threshold > 0:
            self.noise_threshold = max(
                0, self.noise_threshold - self.noise_decrease)


def train_ddpg(epochs, env, agent, trajectory_len):
    rewards = []
    for epoch in range(epochs):

        total_reward = 0
        state = env.reset()
        for _ in range(trajectory_len):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.fit(state, action, reward, done, next_state)

            if done:
                break

            state = next_state
        rewards.append(total_reward)
        print(f'episode: {epoch}\t total_reward: {total_reward}')
    return rewards


class Net_dce(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net_dce, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))

        return x


class DCE():
    def __init__(self, state_dim, action_dim, action_scale):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h_size = 64
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_scale = action_scale

        self.model = Net_dce(self.state_dim, self.action_dim)

    def generate_weights(self, sigma):

        fc1_W = torch.randn_like(self.model.fc1.weight) * sigma
        fc1_b = torch.randn_like(self.model.fc1.bias) * sigma
        fc2_W = torch.randn_like(self.model.fc2.weight) * sigma
        fc2_b = torch.randn_like(self.model.fc2.bias) * sigma
        fc3_W = torch.randn_like(self.model.fc3.weight) * sigma
        fc3_b = torch.randn_like(self.model.fc3.bias) * sigma
        fc4_W = torch.randn_like(self.model.fc4.weight) * sigma
        fc4_b = torch.randn_like(self.model.fc4.bias) * sigma

        return np.array([fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b, fc4_W, fc4_b])

    def set_weights(self, weights):

        # set the weights for each layer
        self.model.fc1.weight.data.copy_(weights[0])
        self.model.fc1.bias.data.copy_(weights[1])
        self.model.fc2.weight.data.copy_(weights[2])
        self.model.fc2.bias.data.copy_(weights[3])
        self.model.fc3.weight.data.copy_(weights[4])
        self.model.fc3.bias.data.copy_(weights[5])
        self.model.fc4.weight.data.copy_(weights[6])
        self.model.fc4.bias.data.copy_(weights[7])

    def get_action(self, state):
        action = self.model(torch.FloatTensor(state))
        return self.action_scale * action.detach().numpy()

    def evaluate(self, env, weights, gamma=1.0, trajectory_len=200, use_target=False):
        self.set_weights(weights)
        episode_return = 0.0
        state = env.reset()
        for t in range(trajectory_len):
            action = self.get_action(torch.FloatTensor(state))
            state, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                break
        return episode_return


def train_dce(env, agent, epochs=501, trajectory_len=200, gamma=1.0, print_every=10, pop_size=100, elite_frac=0.25, sigma=2):

    n_elite = int(pop_size*elite_frac)

    sigma_decrease = 1 / (epochs * 2 // 3)

    scores_deque = deque(maxlen=150)
    scores = []
    # Initialize the weight with random noise
    best_weight = agent.generate_weights(sigma)

    for epoch in range(epochs):
        if epoch > 150:
            sigma = 1
        elif epoch > 300:
            sigma = 0.5
        # Define the cadidates and get the reward of each candidate
        weights_pop = [best_weight +
                       agent.generate_weights(sigma) for i in range(pop_size)]
        rewards = np.array(
            [agent.evaluate(env, weights, gamma, trajectory_len) for weights in weights_pop])

        # Select best candidates from collected rewards
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        agent.update_target_model(best_weight)
        reward = agent.evaluate(env, best_weight, gamma=1.0, use_target=True)

        scores_deque.append(reward)
        scores.append(reward)

        if epoch % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                epoch, np.mean(scores_deque)))
    return scores


env_pendulum = gym.make('Pendulum-v1')
state_dim = 3
action_dim = 1
action_scale = 2


epochs = 50
traj_per_epoch = 500
trajectory_len = 200

q_param = 0.8

agent_dce = DCE(state_dim, action_dim, action_scale)

history_dce = train_dce(env_pendulum, agent_dce)

env_pendulum = gym.make('Pendulum-v1')
state_dim = 3
action_dim = 1
action_scale = 2
epochs = 500
trajectory_len = 200

agent_ddpg = DDPG(state_dim=3, action_dim=1, action_scale=2,
                  noise_decrease=1 / (epochs * trajectory_len))
history_ddpg = train_ddpg(epochs, env_pendulum, agent_ddpg, trajectory_len)

with plt.style.context('ggplot'):

    plt.figure(figsize=(20, 8))
    plt.plot(history_dce)
    plt.title('DCE rewards for pendulum')

with plt.style.context('ggplot'):

    plt.figure(figsize=(20, 8))
    plt.plot(history_ddpg)
    plt.title('DDPG rewards for pendulum')

env_car = gym.make('MountainCarContinuous-v0')

action_scale = 1
state_dim_car = env_car.observation_space.shape[0]
action_dim_car = env_car.action_space.shape[0]

epochs = 500
trajectory_len_car = 1000

agent_ddpg_car = DDPG(state_dim_car, action_dim_car,
                      action_scale, noise_decrease=1 / (epochs * trajectory_len))
histroy_ddpg_car = train_ddpg(
    epochs, env_car, agent_ddpg_car, trajectory_len_car)

with plt.style.context('ggplot'):

    plt.figure(figsize=(20, 8))
    plt.plot(histroy_ddpg_car)
    plt.title('DDPG rewards for mountain car')


class Net_dce_car(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net_dce_car, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))

        return x


class DCE_car():
    def __init__(self, state_dim, action_dim, action_scale):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_scale = action_scale

        self.model = Net_dce_car(self.state_dim, self.action_dim)

    def generate_weights(self, sigma):

        fc1_W = torch.randn_like(self.model.fc1.weight) * sigma
        fc1_b = torch.randn_like(self.model.fc1.bias) * sigma
        fc2_W = torch.randn_like(self.model.fc2.weight) * sigma
        fc2_b = torch.randn_like(self.model.fc2.bias) * sigma
        return np.array([fc1_W, fc1_b, fc2_W, fc2_b])

    def set_weights(self, weights):

        # set the weights for each layer
        self.model.fc1.weight.data.copy_(weights[0])
        self.model.fc1.bias.data.copy_(weights[1])
        self.model.fc2.weight.data.copy_(weights[2])
        self.model.fc2.bias.data.copy_(weights[3])

    def get_action(self, state):
        action = self.model(torch.FloatTensor(state))
        return self.action_scale * action.detach().numpy()

    def evaluate(self, env, weights, gamma=1.0, trajectory_len=200, use_target=False):
        self.set_weights(weights)
        episode_return = 0.0
        state = env.reset()
        for t in range(trajectory_len):
            action = self.get_action(torch.FloatTensor(state))
            state, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                break
        return episode_return


agent_dce_car = DCE_car(state_dim, action_dim, action_scale)

history_dce = train_dce(env_pendulum, agent_dce,
                        trajectory_len=1000, pop_size=50, sigma=1)

"""# car copypaste"""


class Agent(nn.Module):
    def __init__(self, env, h_size=16):
        super(Agent, self).__init__()
        self.env = env
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = env.action_space.shape[0]
        # define layers (we used 2 layers)
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_W = torch.from_numpy(
            weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(
            weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.data

    def act(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            action = self.forward(state)
        return action

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float()
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return


def cem(agent, n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
    """PyTorch implementation of the cross-entropy method.

    Params
    ======
        Agent (object): agent instance
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    n_elite = int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []
    # Initialize the weight with random noise
    best_weight = sigma * np.random.randn(agent.get_weights_dim())

    for i_iteration in range(1, n_iterations+1):
        # Define the cadidates and get the reward of each candidate
        weights_pop = [
            best_weight + (sigma * np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t)
                           for weights in weights_pop])

        # Select best candidates from collected rewards
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)

        torch.save(agent.state_dict(), 'checkpoint.pth')

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque) >= 90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(
                i_iteration-100, np.mean(scores_deque)))
            break
    return scores


agent_dce_car = Agent(env_car)
scores = cem(agent_dce_car)