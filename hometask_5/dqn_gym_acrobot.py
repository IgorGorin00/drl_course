import gym

import torch
import torch.nn as nn

import numpy as np
from random import sample
import matplotlib.pyplot as plt


class NN(nn.Module):
    def __init__(self, state_dim, action_n):
        super(NN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, action_n))

    def forward(self, x):
        x = torch.FloatTensor(x)
        return self.net(x)


class DQN():
    def __init__(self, model, action_n, batch_size, gamma, trajectory_n):
        self.model = model
        self.action_n = action_n
        self.epsilon = 1
        self.epsilon_decrease = 1 / trajectory_n
        self.memory = []
        self.batch_size = batch_size
        self.gamma = gamma

    def get_action(self, state):

        qvalues = self.model(state).detach().numpy()
        probs = np.ones(self.action_n) * self.epsilon / self.action_n
        agrmax_action = np.argmax(qvalues)
        probs[agrmax_action] += 1 - self.epsilon
        action = np.random.choice(self.action_n, p=probs)

        return action

    def get_batch(self):
        batch = sample(population=self.memory, k=self.batch_size)

        states, actions, rewards, dones, next_states = [], [], [], [], []

        for i in range(self.batch_size):
            states.append(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            dones.append(batch[i][3])
            next_states.append(batch[i][4])

        return states, actions, rewards, dones, next_states

    def training_step(self, state, action, reward, done, next_state):

        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) > self.batch_size * 10:

            states, actions, rewards, dones, next_states = self.get_batch()

            q = self.model(states)
            q_next = self.model(next_states)

            targets = q.clone()

            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + \
                    (1 - dones[i]) * self.gamma * torch.max(q_next[i])

            loss = torch.mean((targets.detach() - q) ** 2)
            self.epsilon = max(0, self.epsilon - self.epsilon_decrease)
            return loss


def train(epochs, agent, model, env, trajectory_len, lr, opt_f=torch.optim.SGD):

    opt = opt_f(model.parameters(), lr=lr)
    history = {'rewards': [], 'losses': []}

    for epoch in range(epochs):

        trajectory_reward = 0
        state = env.reset()
        for _ in range(trajectory_len):

            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)

            trajectory_reward += reward
            loss = agent.training_step(state, action, reward, done, next_state)

            if loss is not None:
                loss.backward()
                opt.step()
                opt.zero_grad()

            state = next_state

            # env.render()

            if done:
                break

        history['losses'].append(loss)
        print(f'{epoch = } \t {trajectory_reward = } \t {loss = }')

    return history


def main():
    env = gym.make("Acrobot-v1")

    trajectory_len = 500
    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    batch_size = 64
    gamma = 0.99
    lr = 1e-2
    opt_f = torch.optim.Adam
    epochs = 100

    model = NN(state_dim=state_dim, action_n=action_n)
    agent = DQN(model=model, action_n=action_n,
                batch_size=batch_size, gamma=gamma, trajectory_n=epochs)

    history = train(epochs, agent, model, env, trajectory_len, lr, opt_f=opt_f)


if __name__ == '__main__':
    main()
