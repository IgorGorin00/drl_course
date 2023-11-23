from typing import Tuple, List
import Frozen_Lake
import numpy  as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def init_policy(env: Frozen_Lake.FrozenLakeEnv):
    policy = {} 
    for state in env.get_all_states():
        policy[state] = {}
        possible_actions = env.get_possible_actions(state)
        for action in possible_actions:
            policy[state][action] = 1 / len(possible_actions)
    return policy


def init_v_values(env: Frozen_Lake.FrozenLakeEnv):
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0
    return v_values


def get_q_values(env: Frozen_Lake.FrozenLakeEnv, v_values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                trans_prob = env.get_transition_prob(state, action, next_state)
                q_values[state][action] += trans_prob * env.get_reward(state, action, next_state)
                q_values[state][action] += trans_prob * gamma * v_values[next_state]
    return q_values


def policy_evaluation_step(env: Frozen_Lake.FrozenLakeEnv,
                           v_values, policy, gamma):
    q_values = get_q_values(env, v_values, gamma)
    new_v_values = init_v_values(env)
    for state in env.get_all_states():
        new_v_values[state] = 0
        for action in env.get_possible_actions(state):
            new_v_values[state] += policy[state][action] * q_values[state][action]
    return new_v_values


def policy_evaluation(env: Frozen_Lake.FrozenLakeEnv,
                      steps_per_epoch: int, gamma: float, policy):
    v_values = init_v_values(env)
    for _ in range(steps_per_epoch):
        v_values = policy_evaluation_step(env, v_values, policy, gamma)
    q_values = get_q_values(env, v_values, gamma)
    return q_values


def policy_improvement(env: Frozen_Lake.FrozenLakeEnv, q_values):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        argmax_action = None
        max_q_value = float("-inf")
        
        for action in env.get_possible_actions(state):
            policy[state][action] = 0
            if q_values[state][action] > max_q_value:
                argmax_action = action
                max_q_value = q_values[state][action]
        policy[state][argmax_action] = 1
    return policy


def train(env: Frozen_Lake.FrozenLakeEnv,
          n_epochs: int,
          steps_per_epoch: int,
          gamma: float):

    policy = init_policy(env) 
    for _ in tqdm(range(n_epochs), colour="blue", leave=False):
        q_values = policy_evaluation(env, steps_per_epoch, gamma, policy)
        policy = policy_improvement(env, q_values)
    return policy


def evaluate(env: Frozen_Lake.FrozenLakeEnv,
             n_runs: int, policy) -> Tuple[float, List[int]]:
    
    lens_of_trajectories = []
    sum_of_all_rewards = 0
    for _ in range(n_runs):
        state = env.reset()
        total_reward = 0
        trajectory_len = 0
        for _ in range(999):
            trajectory_len += 1
            action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        lens_of_trajectories.append(trajectory_len)
        sum_of_all_rewards += total_reward
    return sum_of_all_rewards / n_runs, lens_of_trajectories



def main():
    env = Frozen_Lake.FrozenLakeEnv()
    n_epochs = 100  # L = n_epochs
    steps_per_epoch = 100  # K = steps_per_epoch
    gammas = [0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
    mean_rewards = []
    lens_of_trajectories_all = []
    for gamma in gammas:
        policy = train(env, n_epochs, steps_per_epoch, gamma)
        n_runs = 1000
        mean_reward, lens_of_trajectories = evaluate(env, n_runs, policy)
        mean_rewards.append(mean_reward)
        lens_of_trajectories_all.append(lens_of_trajectories)
        print(f"{env.counter = }")
        print(f"{gamma = }\t{mean_reward = }")
    df_lens = pd.DataFrame({g: l for g, l in zip(gammas, lens_of_trajectories_all)})
    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))
        ax1.bar(x=[str(g) for g in gammas], height=mean_rewards)
        for i in range(len(gammas)):
            ax1.text(i, mean_rewards[i], mean_rewards[i], ha="center", fontsize=18)
        ax1.set_title("Rewards with different gammas", fontsize=25)
        ax1.set_xlabel("Gamma value", fontsize=15)
        ax1.set_ylabel("Mean reward of 1000 iterations", fontsize=15)
        plt.xticks(fontsize=18)

        sns.boxplot(df_lens, ax=ax2)
        ax2.set_title("Lens of trajectories on evaluation", fontsize=25)
        ax2.set_xlabel("Gamma value", fontsize=15)
        ax2.set_ylabel("Distribution of trajectory lengths", fontsize=15)
        plt.show()


if __name__ == "__main__":
    main()
