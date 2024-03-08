# Implement PPO without returns, using 
# A(s, a) = r + \gamma * V(s') - V(s), where s' - next state
# Compare with default PPO on Pendulum-v1

import gym
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm.auto import tqdm



class proximalPolicyOptimization(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.9, batch_size=128,
                 epsilon=0.2, epoch_n=30, pi_lr=1e-4, v_lr=5e-4):

        super().__init__()

        self.pi_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU(),
                                      nn.Linear(128, 2 * action_dim), nn.Tanh())

        self.v_model = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, 1))

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

    def get_action(self, state):
        mean, log_std = self.pi_model(torch.FloatTensor(state))
        dist = Normal(mean, torch.exp(log_std))
        action = dist.sample()
        return action.numpy().reshape(1)

    def fit(self, states, actions, rewards, dones, next_states, with_returns: bool = False):

        states, actions, rewards, dones, next_states = map(np.array, [states, actions, rewards, dones, next_states])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        if with_returns:
            returns = np.zeros(rewards.shape)
            returns[-1] = rewards[-1]
            for t in range(returns.shape[0] - 2, -1, -1):
                returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]
            returns = torch.FloatTensor(returns)
        states, actions, rewards, next_states = map(torch.FloatTensor, [states, actions, rewards, next_states])

        mean, log_std = self.pi_model(states).T
        mean, log_std = mean.unsqueeze(1), log_std.unsqueeze(1)
        dist = Normal(mean, torch.exp(log_std))
        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):

            idxs = np.random.permutation(states.shape[0])
            for i in range(0, states.shape[0], self.batch_size):
                b_idxs = idxs[i: i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]
                b_next_states = next_states[b_idxs]

                if with_returns:
                    b_returns = returns[b_idxs]
                    b_advantage = b_returns.detach() - self.v_model(b_states)
                else:
                    b_advantage = b_rewards + self.gamma * self.v_model(b_next_states) - self.v_model(b_states)


                b_mean, b_log_std = self.pi_model(b_states).T
                b_mean, b_log_std = b_mean.unsqueeze(1), b_log_std.unsqueeze(1)
                b_dist = Normal(b_mean, torch.exp(b_log_std))
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1. - self.epsilon,  1. + self.epsilon) * b_advantage.detach()
                pi_loss = - torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage ** 2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()


def train(env, agent, episode_n=50, trajectory_n=20, with_returns: bool = False):
    total_rewards = []

    for episode in tqdm(range(episode_n), colour="blue"):

        states, actions, rewards, dones, next_states = [], [], [], [], []

        for _ in range(trajectory_n):
            total_reward = 0

            state = env.reset()
            for t in range(200):
                states.append(state)

                action = agent.get_action(state)
                actions.append(action)

                next_state, reward, done, _ = env.step(2 * action)
                rewards.append(reward)
                dones.append(done)
                next_states.append(next_state)

                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

        agent.fit(states, actions, rewards, dones, next_states, with_returns)
    return total_rewards


def main():
    env = gym.make("Pendulum-v1")
#    state_dim = env.observation_space.shape[0]
#    action_dim = env.action_space.shape[0]
    state_dim, action_dim = 3, 1
    env = gym.make('Pendulum-v1')
    agent_1 = proximalPolicyOptimization(state_dim, action_dim)
    history_1 = train(env, agent_1, with_returns=True)

    env = gym.make("Pendulum-v1")
    agent_2 = proximalPolicyOptimization(state_dim, action_dim)
    history_2 = train(env, agent_2, with_returns=False)

    means_1 = []
    for i in range(0, len(history_1), 20):
        means_1.append(np.mean(history_1[i : i + 20]))

    means_2 = []
    for i in range(0, len(history_2), 20):
        means_2.append(np.mean(history_2[i : i + 20]))

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
        ax1.plot(history_1, label="with_returns")
        ax1.plot(history_2, label="with next_states")
        ax1.set_title("method comparison")
        ax1.legend()

        ax2.plot(means_1, label="with_returns")
        ax2.plot(means_2, label="with next_states")
        ax2.set_title("smoothec rewards (mean of 20)")
        ax2.legend()

        ax1.set_xlabel("trajectory")
        ax1.set_ylabel("reward")
        ax2.set_xlabel("trajectory")
        ax2.set_ylabel("reward")

        plt.tight_layout()
        plt.legend()
        plt.show()



if __name__ == "__main__":
    main()
    
