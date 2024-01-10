from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import gym
from random import sample
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm.auto import tqdm
import time


class qFunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(state_dim, 64)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, action_dim)
        self.activation = nn.ReLU()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.linear_1(states)
        hidden = self.activation(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.activation(hidden)
        actions = self.linear_3(hidden)
        return actions

class deepQNetwork():
    def __init__(self, state_dim: int, action_dim: int,
                 gamma: float = 0.99, batch_size: int = 64) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = qFunction(self.state_dim, self.action_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.
        self.memory = []


    def get_action(self, state: npt.NDArray[np.float64]) -> int:
        with torch.no_grad():
            q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def fit(self, state: npt.NDArray[np.float64], action: int, reward: float,
            done: bool, next_state: npt.NDArray[np.float64]) -> torch.Tensor | None:
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) < self.batch_size:
            return
        batch = sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = map(
                torch.tensor, list(zip(*batch)))

        next_actions = torch.max(self.q_function(next_states), dim=1).values
        targets = rewards + self.gamma * (1 - dones) * next_actions 
        q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

        loss = torch.mean((q_values - targets.detach()) ** 2)
        return loss


class deepQNetworkHardUpdate(deepQNetwork):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(deepQNetworkHardUpdate, self).__init__(state_dim, action_dim)
        self.target_network = deepcopy(self.q_function)
    
    def update(self) -> None:
        self.target_network.load_state_dict(self.q_function.state_dict())
    
    def fit(self, state: npt.NDArray[np.float64], action: int, reward: float,
            done: bool, next_state: npt.NDArray[np.float64]) -> torch.Tensor | None:
        
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) < self.batch_size:
            return
        batch = sample(self.memory, self.batch_size)
        states, actions, rewards, _, next_states = map(
                torch.tensor, list(zip(*batch)))

        # target network preds
        next_actions = torch.max(self.target_network(next_states), dim=1).values
        targets = rewards + self.gamma * next_actions 
        q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

        loss = torch.mean((q_values - targets.detach()) ** 2)
        return loss


def train_dqn_hard(env, agent, epoch_n: int, episode_n: int, lr: float,
                   epsilons: npt.NDArray[np.float64], t_max: int = 500
                   ) -> npt.NDArray[np.int64]:
    opt = torch.optim.Adam(agent.q_function.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=epoch_n, steps_per_epoch=episode_n)

    episode_rewards = []
    for epoch in range(epoch_n):
        for episode in tqdm(range(episode_n), colour="blue"):
            try:
                agent.epsilon = epsilons[epoch * episode_n + episode]
            except IndexError:
                agent.epsilon = epsilons[-1]
            episode_reward = 0
            state = env.reset()
            for _ in range(t_max):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                loss = agent.fit(state, action, reward, done, next_state)
                if loss is not None:
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                if done:
                    break
                state = next_state
            episode_rewards.append(episode_reward)
            lr_sched.step()
        agent.update()
    return np.array(episode_rewards)


class deepQNetworkSoftUpdate(deepQNetworkHardUpdate):
    def __init__(self, state_dim: int, action_dim: int, tau: float) -> None:
        super(deepQNetworkSoftUpdate, self).__init__(state_dim, action_dim)
        self.tau = tau
    
    def update(self) -> None:
        target_weights = self.target_network.state_dict()
        trained_weights = self.q_function.state_dict()
        weights_for_update = deepcopy(target_weights)
        for param_group_name in weights_for_update.keys():
            target_param = target_weights[param_group_name]
            trained_param = trained_weights[param_group_name]
            weights_for_update[param_group_name] = self.tau * trained_param + (1 - self.tau) * target_param
        self.target_network.load_state_dict(weights_for_update)


def train_dqn_soft(env, agent, episode_n: int, lr: float,
                   epsilons: npt.NDArray[np.float64], t_max: int = 500
                   ) -> npt.NDArray[np.int64]:
    opt = torch.optim.Adam(agent.q_function.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=episode_n)

    episode_rewards = []
    for episode in tqdm(range(episode_n), colour="blue"):
        try:
            agent.epsilon = epsilons[episode]
        except IndexError:
            agent.epsilon = epsilons[-1]
        episode_reward = 0
        state = env.reset()
        for _ in range(t_max):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            loss = agent.fit(state, action, reward, done, next_state)
            if loss is not None:
                loss.backward()
                opt.step()
                opt.zero_grad()
                agent.update()
            if done:
                break
            state = next_state
        episode_rewards.append(episode_reward)
        lr_sched.step()
        agent.update()
    return np.array(episode_rewards)


class doubleDeepQNetwork(deepQNetworkSoftUpdate):
    def __init__(self, state_dim: int, action_dim: int, tau: float) -> None:
        super(doubleDeepQNetwork, self).__init__(state_dim, action_dim, tau)
    
    def fit(self, state: npt.NDArray[np.float64], action: int, reward: float,
            done: bool, next_state: npt.NDArray[np.float64]) -> torch.Tensor | None:
        
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) < self.batch_size:
            return
        batch = sample(self.memory, self.batch_size)
        states, actions, rewards, _, next_states = map(
                torch.tensor, list(zip(*batch)))
        
        target_network_preds = torch.max(self.target_network(next_states), dim=1).indices
        next_actions = self.q_function(
                next_states)[torch.arange(self.batch_size), target_network_preds]
        targets = rewards + self.gamma * next_actions 
        q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

        loss = torch.mean((q_values - targets.detach()) ** 2)
        return loss


def train_double_dqn(env, agent, episode_n: int, lr: float,
                     epsilons: npt.NDArray[np.float64], t_max: int = 500
                     ) -> npt.NDArray[np.int64]:
    opt = torch.optim.Adam(agent.q_function.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=episode_n)

    episode_rewards = []
    for episode in tqdm(range(episode_n), colour="blue"):
        try:
            agent.epsilon = epsilons[episode]
        except IndexError:
            agent.epsilon = epsilons[-1]
        episode_reward = 0
        state = env.reset()
        for _ in range(t_max):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            loss = agent.fit(state, action, reward, done, next_state)
            if loss is not None:
                loss.backward()
                opt.step()
                opt.zero_grad()
                agent.update()
            if done:
                break
            state = next_state
        episode_rewards.append(episode_reward)
        lr_sched.step()
        agent.update()
    return np.array(episode_rewards)


def median_of_segments(array, segment_size=70):
    reshaped_array = array[:len(array) - len(array) % segment_size].reshape(-1, segment_size)
    medians = np.median(reshaped_array, axis=1)
    return medians


def main():

    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    dqn_hard = deepQNetworkHardUpdate(state_dim, action_dim)
    epoch_n = 3
    episode_n = 2000
    epsilons = np.exp(np.linspace(0, -5, epoch_n * episode_n))
    lr = 1e-3
    print("training dqn with hard update")
    dqn_hard_episode_rewards = train_dqn_hard(
            env, dqn_hard, epoch_n, episode_n, lr, epsilons)

    episode_n = 7000
    tau = .2
    epsilons = np.exp(np.linspace(0, -5, episode_n))
    lr = 1e-3

    soft_dqn = deepQNetworkSoftUpdate(state_dim, action_dim, tau)
    print("training dqn with soft update")
    dqn_soft_episode_rewards = train_dqn_soft(
            env, soft_dqn, episode_n, lr, epsilons)

    tau = 0.5
    double_dqn = doubleDeepQNetwork(state_dim, action_dim, tau)
    episode_n = 7000
    epsilons = np.exp(np.linspace(0, -5, episode_n))
    lr = 1e-3

    print("training double dqn")
    dqn_double_episode_rewards = train_double_dqn(
            env, double_dqn, episode_n, lr, epsilons)

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 16))
        ax1.plot(dqn_hard_episode_rewards, label="DQN Hard update")
        ax1.plot(dqn_soft_episode_rewards, label="DQN Soft update")
        ax1.plot(dqn_double_episode_rewards, label="Double DQN")
        ax1.set_xlabel("N of trajectory (episode)")
        ax1.set_ylabel("Reward")
        ax1.set_title("Episode rewards for DQN with hard update")
        ax1.axhline(-100, color="red", label="-100 reward treshold")
        ax1.legend()
        
        ax2.plot(median_of_segments(dqn_hard_episode_rewards), label="Smoothed DQN Hard update")
        ax2.plot(median_of_segments(dqn_soft_episode_rewards), label="Smoothed DQN Soft update")
        ax2.plot(median_of_segments(dqn_double_episode_rewards), label="Smoothed Double DQN")
        ax2.set_xlabel("N of evaluation")
        ax2.set_ylabel("Reward")
        ax2.set_title("Validation trajectory rewards (target network) for DQN with hard update")
        ax2.axhline(-100, color="red", label="-100 reward treshold")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

