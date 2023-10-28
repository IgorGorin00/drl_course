import gym

import torch
import torch.nn as nn

import time
from tqdm.notebook import tqdm
import numpy as np
from random import sample
import matplotlib.pyplot as plt
# # Deep Q-learning


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
    def __init__(self, model, action_n, batch_size, trajectory_n):
        self.model = model
        self.action_n = action_n
        self.epsilon = 1
        self.epsilon_decrease = 1 / trajectory_n
        self.memory = []
        self.batch_size = batch_size

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

    def training_step(self, state, action, reward, done, next_state, gamma):

        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) > self.batch_size * 10:

            states, actions, rewards, dones, next_states = self.get_batch()

            q = self.model(states)
            q_next = self.model(next_states)

            targets = q.clone()

            for i in range(self.batch_size):
                targets[i][actions[i]] = rewards[i] + \
                    (1 - dones[i]) * gamma * torch.max(q_next[i])

            loss = torch.mean((targets.detach() - q) ** 2)
            self.epsilon = max(0, self.epsilon - self.epsilon_decrease)
            return loss


def train_dqn(traj_n, agent, model, env, trajectory_len, gamma_start, gamma_end, lr, opt_f=torch.optim.SGD):

    opt = opt_f(model.parameters(), lr=lr)
    history = {'rewards': [], 'losses': []}

    slope = (gamma_end - gamma_start) / n
    def gamma_sched(traj_i): return traj_i * slope + gamma_start

    for traj_i in tqdm(range(traj_n)):

        trajectory_reward = 0
        trajectory_loss = 0

        if traj_i <= traj_n * 0.2:
            gamma = gamma_start
        else:
            gamma = gamma_sched(traj_i)

        state = env.reset()

        for _ in range(trajectory_len):

            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)

            trajectory_reward += reward
            loss = agent.training_step(
                state, action, reward, done, next_state, gamma)

            if loss is not None:
                loss.backward()
                opt.step()
                opt.zero_grad()
                trajectory_loss += loss.item()
            state = next_state

            if done:
                break

        history['rewards'].append(trajectory_reward)
        history['losses'].append(trajectory_loss)
        if traj_i % 10 == 0:
            print(f'{traj_i = } \t {trajectory_reward = } \t {trajectory_loss = }')

    return history


# # Deep Cross-Entropy

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


def train_dce(epochs, env, agent, traj_per_epoch, lr, trajectory_len=500, opt_f=torch.optim.SGD):
    history = {'loss': [], 'reward': [], 'q_param': [], 'lr': [], 'etn': []}

    opt = opt_f(agent.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=1)

    def q_sched(epoch): return max(0.7, (epoch / epochs)*0.9)

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


def plot_rewards(history_dce, history_dqn_sched)
   with plt.style.context('ggplot'):
        plt.figure(figsize=(15, 6))
        plt.plot(history_dqn_sched['rewards'],
                 label=f'sched from {gamma_start} to {gamma_end}')
        # plt.plot(history_dqn_099['rewards'], label='0.99')
        plt.axvline(x=traj_n * 0.2, color='gray', label='scheduling start')
        plt.title('DQN rewards')
        plt.legend()
        plt.show()

    with plt.style.context('ggplot'):
        plt.figure(figsize=(15, 6))
        plt.plot(history_dce['reward'])
        plt.title('DCE rewards')
        plt.show()


def main():
    env = gym.make("Acrobot-v1")

    trajectory_len = 500
    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    batch_size = 64
    gamma = 0.995
    lr = 1e-2
    opt_f = torch.optim.Adam

    traj_n = 200

    model_dqn = NN(state_dim=state_dim, action_n=action_n)
    agent_dqn = DQN(model=model_dqn, action_n=action_n,
                    batch_size=batch_size, trajectory_n=traj_n)


    gamma_start = 0.99
    gamma_end = 0.9999
    history_dqn_sched = train_dqn(
        traj_n, agent_dqn, model_dqn, env, trajectory_len, gamma_start, gamma_end, lr, opt_f=opt_f)


    agent_dce = CEMAgent(state_dim, action_n)

    epochs = 31
    traj_per_epoch = 800
    lr_dce = 0.3
    opt_f = torch.optim.Adam
    history_dce = train_dce(epochs, env, agent_dce, traj_per_epoch, lr_dce)

    plot_rewards(history_dce, history_dqn_sched)


if __name__ == '__main__':
    main()
