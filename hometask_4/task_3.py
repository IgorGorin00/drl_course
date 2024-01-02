# find best way of choosing epsilon for monte-carlo for taxi-v3

from typing import List
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm.auto import tqdm

def get_epsilon_greedy_action(q_values: np.ndarray, epsilon: float, action_n: int) -> int:
    argmax_action = np.argmax(q_values)
    probs = epsilon * np.ones(action_n) / action_n
    probs[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=probs)
    return action


def MonteCarlo(env,
               episode_n: int, epsilons, t_max=500, gamma=0.99) -> List[int]:
    state_n = env.observation_space.n
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))
    n_values = np.zeros((state_n, action_n))

    total_rewards = []
    for episode in tqdm(range(episode_n), colour="blue"):
        states, actions, rewards = [], [], []

        try:
            epsilon = epsilons[episode]
        except IndexError:
            epsilon = epsilons[-1]
        state = env.reset()
        for _ in range(t_max):
            states.append(state)
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break


        total_rewards.append(sum(rewards))
        g_values = np.zeros(len(rewards))
        g_values[-1] = rewards[-1]
        for t in range(len(rewards) - 2, -1, -1):
            g_values[t] = rewards[t] + gamma * g_values[t + 1]

        for t in range(len(rewards)):
            q_values[states[t]][actions[t]] += (g_values[t] - q_values[states[t]][actions[t]]) / (n_values[states[t]][actions[t]] + 1)
            n_values[states[t]][actions[t]] += 1

        epsilon -= 1 / episode_n
    
    return total_rewards



def smooth(rewards: List[int]):
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
    env = gym.make("Taxi-v3")

    
    episode_n = 20000
    epsilones_default = np.linspace(1, 0.01, episode_n)
    epsilones_1 = np.sin(np.linspace(np.pi / 2, np.pi * 0.9, episode_n))
    epsilones_2 = list(map(lambda x: max(0.1, min(1., 1. - np.log10((x + 1) / (episode_n / 10)))), np.arange(episode_n)))
    epsilones_3 = np.exp(np.linspace(0, -3, episode_n))
    mc_rewards_default = MonteCarlo(env, episode_n, epsilones_default)
    mc_rewards_1 = MonteCarlo(env, episode_n, epsilones_1)
    mc_rewards_2 = MonteCarlo(env, episode_n, epsilones_2)
    mc_rewards_3 = MonteCarlo(env, episode_n, epsilones_3)

    
    smoothed_mc_rewards_default = smooth(mc_rewards_default)
    smoothed_mc_rewards_1 = smooth(mc_rewards_1)
    smoothed_mc_rewards_2 = smooth(mc_rewards_2)
    smoothed_mc_rewards_3 = smooth(mc_rewards_3)
    
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(25, 16))

        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1]) 
        ax0 = plt.subplot(gs[0, :]) # This creates a subplot that spans the top row.
        ax1 = plt.subplot(gs[1, 0]) # This creates a subplot in the bottom left.
        ax2 = plt.subplot(gs[1, 1]) # This creates a subplot in the bottom right.

        # Plot smoothed rewards on ax1
        ax0.plot(smoothed_mc_rewards_default, label="default (linear)")
        ax0.plot(smoothed_mc_rewards_1, label="method1")
        ax0.plot(smoothed_mc_rewards_2, label="method2")
        ax0.plot(smoothed_mc_rewards_3, label="method3")
        ax0.set_xlabel("trajectory n")
        ax0.set_ylabel("reward")
        ax0.set_title("Taxi-v3 smoothed rewards for Moncte-Carlo (meadian of 500)")
        ax0.legend()

        # Plot on ax0
        ax1.plot(mc_rewards_default, label="default (linear)")
        ax1.plot(mc_rewards_1, label="method1")
        ax1.plot(mc_rewards_2, label="method2")
        ax1.plot(mc_rewards_3, label="method3")
        ax1.set_xlabel("trajectory n")
        ax1.set_ylabel("reward")
        ax1.set_title("Taxi-v3 rewards for Monte-Carlo")
        ax1.legend()

        # Plot epsilones on ax2
        ax2.plot(epsilones_default, label="default (linear)")
        ax2.plot(epsilones_1, label="method1")
        ax2.plot(epsilones_2, label="method2")
        ax2.plot(epsilones_3, label="method3")
        ax2.set_xlabel("trajectory n")
        ax2.set_ylabel("epsilon")
        ax2.set_title("Epsilon values over trajectories")
        ax2.legend()

        plt.show()

if __name__ == "__main__":
    main()
