import gym

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import random


class NN(nn.Module):
    def __init__(self, state_dim, action_n):
        super(NN, self).__init__()

        self.linear1 = nn.Linear(state_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, action_n)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DQN():
    def __init__(self, action_n, batch_size, gamma, model, lr, trajectory_n):

        self.action_n = action_n
        self.model = model
        self.epsilon = 1
        self.memory = []
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon_decrease = 1 / trajectory_n
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def get_action(self, state):

        q_values = self.model(torch.FloatTensor(state)).detach().numpy()

        prob = np.ones(self.action_n) * self.epsilon / self.action_n
        agrmax_action = np.argmax(q_values)
        prob[agrmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_n), p=prob)

        return action

    def get_batch(self):
        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, dones, next_states = [], [], [], [], []
        for i in range(len(batch)):
            states.append(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            dones.append(batch[i][3])
            next_states.append(batch[i][4])

        return torch.FloatTensor(states), actions, rewards, dones, torch.FloatTensor(next_states)

    def fit(self, state, action, reward, done, next_state):

        self.memory.append([state, action, reward, done, next_state])
        if len(self.memory) > self.batch_size:

            states, actions, rewards, dones, next_states = self.get_batch()

            q_values = self.model(states)
            next_q_values = self.model(next_states)

            targets = q_values.clone()
            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + \
                    (1 - dones[i]) * self.gamma * torch.max(next_q_values[i])

            loss = torch.mean((targets.detach() - q_values) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.epsilon = max(0, self.epsilon - self.epsilon_decrease)


def main():

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n

    trajectory_n = 100
    trajectory_len = 500
    batch_size = 64
    gamma = 0.99
    lr = 1e-2
    model = NN(state_dim, action_n)
    agent = DQN(action_n, batch_size, gamma, model, lr, trajectory_n)

    for traj_i in range(trajectory_n):
        total_rewards = 0
        state = env.reset()
        for i in range(trajectory_len):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            total_rewards += reward
            agent.fit(state, action, reward, done, next_state)

            state = next_state

            env.render()

            if done:
                break

        print(f'{traj_i = } \t {total_rewards = }')


if __name__ == '__main__':
    main()
