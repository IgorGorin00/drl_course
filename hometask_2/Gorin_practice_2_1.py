import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
import gym

import torch
import torch.nn as nn

from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt


class CEMAgent(nn.Module):
    def __init__(self, state_dim, action_n, loss_f=nn.CrossEntropyLoss()):
        super(CEMAgent, self).__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_n),
            nn.Softmax(dim=0))
        self.loss_f = loss_f

    def forward(self, x):
        return self.net(x)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self(state)
        action = np.random.choice(
            self.action_n, p=action_probs.detach().numpy())
        return action

    def training_step(self, elite_trajectories):
        elite_states, elite_actions = [], []
        for t in elite_trajectories:
            elite_states.extend(t['states'])
            elite_actions.extend(t['actions'])
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.LongTensor(np.array(elite_actions))
        out = self(elite_states)
        loss = self.loss_f(out, elite_actions)
        return loss * 10000


def get_trajectory(env, agent, trajectory_len, viz=False):
    trajectory = {
        'states': [],
        'actions': [],
        'reward': 0
    }
    state = env.reset()
    trajectory['states'].append(state)
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        trajectory['actions'].append(action)
        trajectory['reward'] += reward
        if viz:
            env.render()
        if done:
            break
        trajectory['states'].append(state)
    return trajectory


def get_elite_trajectories(trajectories, q_param):
    rewards = [t['reward'] for t in trajectories]
    q_value = np.quantile(rewards, q_param)
    return round(np.mean(rewards), 2), [t for t in trajectories if t['reward'] > q_value]


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


def train(epochs, env, agent, traj_per_epoch, lr, trajectory_len=500, opt_f=torch.optim.SGD):
    history = {'loss': [], 'reward': [], 'q_param': [], 'lr': [], 'etn': []}

    opt = opt_f(agent.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=1)

    def q_sched(epoch): return max(0.2, (epoch / epochs)*0.9)

    start = time.perf_counter()
    loss = 0

    for epoch in range(epochs):
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in tqdm(
            range(traj_per_epoch), leave=True, colour='blue')]
        el_tr_n = 0
        loss = 0
        q_param = round(q_sched(epoch), 5)
        mean_reward, elite_trajectories = get_elite_trajectories(
            trajectories, q_param)

        if len(elite_trajectories) > 0:
            el_tr_n = len(elite_trajectories)
            loss = agent.training_step(elite_trajectories)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss = round(loss.item(), 2)
        last_lr = round(get_lr(opt), 5)
        history['loss'].append(loss)
        history['reward'].append(mean_reward)
        history['q_param'].append(q_param)
        history['lr'].append(last_lr)
        history['etn'].append(el_tr_n)

        print(f'Epoch [{epoch}] Mean reward [{mean_reward}] Loss [{round(loss, 2)}] Q param [{round(q_param, 3)}] Last lr[{round(last_lr, 4)}] Elite traj n [{el_tr_n}]')
        lr_scheduler.step()
    end = time.perf_counter()
    print(f'Training took {round(end-start, 4)} secs')
    return history


def main():
    env = gym.make("Acrobot-v1")
    print('env created')

    state_dim = 6
    action_n = 3
    agent = CEMAgent(state_dim, action_n)

    epochs = 31
    traj_per_epoch = 800
    lr = 0.3
    opt_f = torch.optim.Adam

    history = train(epochs, env, agent, traj_per_epoch, lr)

    last_trajectory = get_trajectory(env, agent, trajectory_len=500)
    print(last_trajectory['reward'])


if __name__ == '__main__':
    main()


#############################################################################
#                                                                           #
#              EVERYTHING BELOW RELATED TO THE VISUALIZATION                #
#                                                                           #
#############################################################################
def viz(history):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
        ax[0].plot(history['reward'])
        ax[0].set_title('Reward x Epoch')
        ax[0].set_ylabel('Reward')
        ax[1].plot(history['loss'][1:])
        ax[1].set_title('Loss x Epoch')
        ax[1].set_ylabel('Cross-Entropy score * 10000')
        ax[1].set_xlabel('Epoch')

    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
        ax[0].plot(history['q_param'])
        ax[0].set_title('Q x Epoch')
        ax[0].set_ylabel('Q parameter')
        ax[0].set_ylim(0, 1)
        ax[1].plot(history['etn'])
        ax[1].set_title('Elite trajectories x Epoch')
        ax[1].set_ylabel('Number of elite trjaectories')
        ax[1].set_xlabel('Epoch')

    def normalize(x: list):
        x = np.array(x)
        print(f'min={x.min()}, max={x.max()}, mean={x.mean()}, std={x.std()}')
        x = (x - x.mean()) / x.std()
        print(f'min={x.min()}, max={x.max()}, mean={x.mean()}, std={x.std()}')
        return x

    print('Q_params')
    q_params = normalize(history['q_param'])
    print()

    print('Elite trajectories numbers')
    etns = normalize(history['etn'])
    print()

    print('Pearson correlation for q_params and elite traj numbers: test statistic and p-value')
    print(pearsonr(x=q_params, y=etns))

    print('Rewards')
    rewards = normalize(history['reward'])

    print('Pearson correlation for q_params and rewards: test statistic and p-value')
    print(pearsonr(x=q_params, y=rewards))

    history_df = pd.DataFrame(history)

    history_df = (history_df - history_df.mean()) / history_df.std()

    corr_mat = history_df.corr()

    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_mat, annot=True)
    plt.show()
