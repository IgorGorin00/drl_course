import os

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import time
import numpy as np


class cemAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(cemAgent, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, 512)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc2 = nn.Linear(512, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(1024, self.action_dim)
        self.tanh = nn.Tanh()
        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, x):

        x = self.lrelu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.out(x))

        return x

    def extract_weights_and_biases(self):
        weights = {
            'fc1': self.fc1.weight,
            'fc2': self.fc2.weight,
            'out': self.out.weight
        }
        biases = {
            'fc1': self.fc1.bias,
            'fc2': self.fc2.bias,
            'out': self.out.bias
        }

        return weights, biases

    def get_action(self, state):
        state = torch.FloatTensor(state)
        out = self.forward(state)
        return out.detach().numpy()

    def training_step(self, elite_trajectories):

        fc1_weights = [t['weights']['fc1'] for t in elite_trajectories]
        fc1_biases = [t['biases']['fc1'] for t in elite_trajectories]

        fc2_weights = [t['weights']['fc2'] for t in elite_trajectories]
        fc2_biases = [t['biases']['fc2'] for t in elite_trajectories]

        out_weights = [t['weights']['out'] for t in elite_trajectories]
        out_biases = [t['biases']['out'] for t in elite_trajectories]

        fc1_weigts_mean = sum(fc1_weights) / len(fc1_weights)
        fc1_biases_mean = sum(fc1_biases) / len(fc1_biases)

        fc2_weigts_mean = sum(fc2_weights) / len(fc2_weights)
        fc2_biases_mean = sum(fc2_biases) / len(fc2_biases)

        out_weigts_mean = sum(out_weights) / len(out_weights)
        out_biases_mean = sum(out_biases) / len(out_biases)

        self.fc1.weight = nn.Parameter(fc1_weigts_mean)
        self.fc1.bias = nn.Parameter(fc1_biases_mean)

        self.fc2.weight = nn.Parameter(fc2_weigts_mean)
        self.fc2.bias = nn.Parameter(fc2_biases_mean)

        self.out.weight = nn.Parameter(out_weigts_mean)
        self.out.bias = nn.Parameter(out_biases_mean)


def get_trajectory(env, agent, trajectory_len, viz=False):
    trajectory = {
        'states': [],
        'actions': [],
        'reward': 0,
        'weights': None,
        'biases': None
    }
    state = env.reset()
    for _ in range(trajectory_len):

        trajectory['states'].append(state)
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        trajectory['actions'].append(action)
        trajectory['reward'] += reward
        if viz:
            env.render()
        if done:
            break
    if len(trajectory['states']) != len(trajectory['actions']):
        raise RuntimeError(
            f'len of states {len(trajectory["states"])} \
                    does not equal to len of actions\
                    {len(trajectory["actions"])}')
    weights, biases = agent.extract_weights_and_biases()
    trajectory['weights'] = weights
    trajectory['biases'] = biases

    return trajectory


def get_elite_trajectories(trajectories, q_param):
    rewards = [t['reward'] for t in trajectories]
    q_value = np.quantile(rewards, q_param)
    return np.mean(rewards), [t for i, t in enumerate(trajectories) if t['reward'] > q_value]


'''
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]
'''


def train(epochs, env, agent, traj_per_epoch, q_param_start, trajectory_len=999):
    history = {'reward': [], 'q_param': [], 'etn': []}

    def q_sched(epoch): return max(q_param_start, (epoch / epochs)*0.9)

    start = time.perf_counter()

    for epoch in range(epochs):
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in tqdm(
            range(traj_per_epoch), leave=True, colour='blue')]
        el_tr_n = 0
        q_param = round(q_sched(epoch), 5)
        mean_reward, elite_trajectories = get_elite_trajectories(
            trajectories, q_param)

        if len(elite_trajectories) > 0:
            el_tr_n = len(elite_trajectories)
            agent.training_step(elite_trajectories)

        history['reward'].append(mean_reward)
        history['q_param'].append(q_param)
        history['etn'].append(el_tr_n)

        print(
            f'Epoch [{epoch}] Mean reward [{mean_reward}] Q param [{round(q_param, 3)}] Elite traj n [{el_tr_n}]')
    end = time.perf_counter()
    print(f'Training took {round(end-start, 4)} secs')
    return history


def main():
    env = gym.make("MountainCarContinuous-v0")

    print('env created')

    state_dim = 2
    action_n = 1
    agent = cemAgent(state_dim, action_n)

    epochs = 30
    traj_per_epoch = 5000
    q_param_start = 0.7
    lr = 0.3

    history = train(epochs, env, agent, traj_per_epoch, q_param_start)
    last_trajectory = get_trajectory(env, agent, trajectory_len=500, viz=True)
    print(last_trajectory)


if __name__ == '__main__':
    main()
