# Implementation of Deep Cross-Entropy method, Monte-Carlo, SARSA and Q-learning
# for environment with continuous observation space - CartPole
#
# Continuous observation space is discretized and then its just default
# method
#
#
#

from typing import Tuple
import numpy as np
import numpy.typing as npt
import gym
import torch
import torch.nn as nn
from typing import List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    total_reward: int


class crossEntropyMethod(nn.Module):
    def __init__(self, env, lr):
        super(crossEntropyMethod, self).__init__()
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_n = self.env.action_space.n
        self.t_max = 500

        self.possible_actions = np.arange(self.action_n)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, self.action_n)
        )

        self.loss_f = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=0)

        self.lr = lr
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, state: npt.NDArray[np.float32]) -> int:
        logits = self.forward(torch.FloatTensor(state))
        probs = self.softmax(logits).detach().numpy()
        return np.random.choice(self.possible_actions, p=probs)

    def training_step(self, elite_states: torch.FloatTensor,
                            elite_actions: torch.LongTensor) -> None:
        preds = self.forward(elite_states)
        loss = self.loss_f(preds, elite_actions)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

    def get_trajectory(self, trajectory_len: int) -> Trajectory:
        states, actions, total_reward = [], [], 0

        state = self.env.reset()
        for _ in range(trajectory_len):
            states.append(state)
            action = self.get_action(state)
            actions.append(action)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return Trajectory(np.array(states), np.array(actions), total_reward)

    def get_elite_states_and_actions(self, trajectories, epoch_rewards, q_param):
        q_value = np.quantile(epoch_rewards, q_param)
        elite_states, elite_actions = [], []
        for t in trajectories:
            if t.total_reward > q_value:
                elite_states.append(t.states)
                elite_actions.append(t.actions)

        if len(elite_states) == 0:
            return None, None
        return torch.FloatTensor(np.concatenate(elite_states)),\
               torch.LongTensor(np.concatenate(elite_actions))



    def train(self, epoch_n: int, traj_per_epoch: int, q_param) -> List[int]:
        all_rewards = []
        for _ in tqdm(range(epoch_n), colour="blue"):
            trajectories = [self.get_trajectory(self.t_max) for _\
                            in range(traj_per_epoch)]
            epoch_rewards = [t.total_reward for t in trajectories]
            all_rewards.extend(epoch_rewards)
            elite_states, elite_actions = self.get_elite_states_and_actions(
                    trajectories, epoch_rewards, q_param)
            if elite_states is not None and elite_actions is not None:
                self.training_step(elite_states, elite_actions)
        return all_rewards 

def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)

def round_state(state: npt.NDArray[np.float32]
                ) -> Tuple[float, float, float, float]:
    p0 = round(state[0], 1)
    p1 = round(state[1], 1)
    p2 = round(state[2], 3)
    p3 = round(state[3], 1)
    return (p0, p1, p2, p3)


def MonteCarlo(env, episode_n: int, t_max: int = 500, gamma: float = 0.99
               ) -> npt.NDArray[np.float64]:
    action_n = env.action_space.n
    q = {}
    n = {}
    all_rewards = np.zeros(episode_n) 
    for episode in tqdm(range(episode_n), colour="blue"):
        epsilon = 1 - episode / episode_n
        states, actions, rewards = [], [], []

        state = env.reset()
        state = round_state(state)
        if state not in q.keys():
            q[state] = np.zeros(action_n)

        for _ in range(t_max):

            states.append(state)
            action = get_epsilon_greedy_action(q[state], epsilon, action_n)
            actions.append(action)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = round_state(state)
            if state not in q.keys():
                q[state] = np.zeros(action_n)

            if done:
                break
        all_rewards[episode] = sum(rewards)

        real_t_len = len(rewards)
        returns = np.zeros(real_t_len + 1)
        for t in range(real_t_len - 1, -1, -1):
            returns[t] = rewards[t] + gamma * returns[t + 1]

        for t in range(real_t_len):
            state = states[t]
            action = actions[t]

            if state not in q.keys():
                q[state] = np.zeros(action_n)
            if state not in n.keys():
                n[state] = np.zeros(action_n)

            q[state][action] +=\
                    (returns[t] - q[state][action]) / (1 + n[state][action])
            n[state][action] += 1
    return all_rewards


#   SARSA Algorithm
#   Let Q(s, a) = 0 and epsilon = 1
#   For each episode k do:
#       while episode is not over:
#       1.  being in the state S_t making action A_t ~ policy(S_t) where policy is epsilon-greedy (Q)
#           getting reward R_t moving to the state S_t+1, making action A_t+1 ~ policy(S_t+1)
#       2.  According to (S_t, A_t, R_t, S_t+1, A_t+1) updating Q:
#               Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (R_t + gamma * Q(S_t+1, A_t+a) - Q(S_t, A_t))
#       decrease epsilon


def SARSA(env, episode_n: int, t_max: int = 500, gamma: float = 0.99,
          alpha: float = 0.5) -> npt.NDArray[np.float64]:

    action_n = env.action_space.n

    q_values = {}
    epsilon = 1

    total_rewards = np.zeros(episode_n) 
    for episode in tqdm(range(episode_n), colour="blue"):
        
        trajectory_reward = 0

        state = env.reset()
        state = round_state(state)
        if state not in q_values.keys():
            q_values[state] = np.zeros(action_n)
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        for _ in range(t_max):
            next_state, reward, done, _ = env.step(action)
            next_state = round_state(next_state)
            if next_state not in q_values.keys():
                q_values[next_state] = np.zeros(action_n)
            trajectory_reward += reward
            next_action = get_epsilon_greedy_action(q_values[next_state],\
                                                    epsilon, action_n)
            q_values[state][action] +=\
                    alpha * (reward + gamma * q_values[next_state][next_action]\
                    - q_values[state][action])
            state = next_state
            action = next_action
            if done:
                break
        total_rewards[episode] = trajectory_reward
        epsilon -= 1 / episode_n
    return total_rewards


def QLearning(env, episode_n: int, t_max: int = 500, gamma: float = 0.99,
              alpha: float = 0.5) -> npt.NDArray[np.float64]:
    action_n = env.action_space.n

    q = {}
    epsilon = 1
    rewards_all = np.zeros(episode_n)
    for episode in tqdm(range(episode_n), colour="blue"):
        
        state = env.reset()
        state = round_state(state)
        if state not in q.keys():
            q[state] = np.zeros(action_n)
        
        t_reward = 0
        for _ in range(t_max):
            
            action = get_epsilon_greedy_action(q[state], epsilon, action_n)
            next_state, reward, done, _ = env.step(action)
            
            t_reward += reward
            if done:
                break
            
            next_state = round_state(next_state)
            if next_state not in q.keys():
                q[next_state] = np.zeros(action_n)


            argmax_next_action = np.argmax(q[next_state])
            q[state][action] +=\
                    alpha * (reward + gamma * q[next_state][argmax_next_action]\
                    - q[state][action])

            state = next_state
        rewards_all[episode] = t_reward
        epsilon -= 1 / episode_n
    
    return rewards_all

def smooth(rewards):
    # Set the window size
    window_size = 500

    # Calculate the number of windows
    num_windows = len(rewards) // window_size

    # Create an rewards to store the median values
    median_values = np.zeros(num_windows)

    # Iterate through the rewards and calculate the median for each window
    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size
        window = rewards[start_index:end_index]
        median_values[i] = np.median(window)
    return median_values


def main():
    env = gym.make("CartPole-v1")

    CEM = crossEntropyMethod(env, lr=0.01)
    epoch_n = 100
    traj_per_epoch = 20
    q_param = 0.8
    print("Training cross-entropy")
    ce_rewards = CEM.train(epoch_n, traj_per_epoch, q_param)
    
    episode_n = 200001
    print("Training monte-carlo")
    mc_rewards = MonteCarlo(env, episode_n)

    print("Training sarsa")
    sarsa_rewards = SARSA(env, episode_n, gamma=0.9999, alpha=0.3)


    print("Training qlearning")
    qlearning_rewards = QLearning(env, episode_n, gamma=0.9999, alpha=0.3)

    smoothed_mc_rewards = smooth(mc_rewards)
    smoothed_sarsa_rewards = smooth(sarsa_rewards)
    smoothed_qlearning_rewards = smooth(qlearning_rewards)
    smoothed_ce_rewards = smooth(ce_rewards)

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
        ax1.plot(ce_rewards, label="Cross-Entropy")
        ax1.plot(mc_rewards, label="Monte-Carlo")
        ax1.plot(sarsa_rewards, label="SARSA")
        ax1.plot(qlearning_rewards, label="Q-Learning")
        ax1.legend()
        ax1.set_xlabel("trajectory n")
        ax1.set_ylabel("reward")
        ax1.set_title("CartPole rewards comparison")


        ax2.plot(smoothed_ce_rewards, label="Cross-Entropy")
        ax2.plot(smoothed_mc_rewards, label="Monte-Carlo")
        ax2.plot(smoothed_sarsa_rewards, label="SARSA")
        ax2.plot(smoothed_qlearning_rewards, label="Q-Learning")
        ax2.legend()
        ax2.set_xlabel("trajectory n")
        ax2.set_ylabel("reward")
        ax2.set_title("CartPole smoothed rewards comparison (meadian of 500)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
