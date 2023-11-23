import Frozen_Lake
from typing import Tuple, Dict, List 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

State = Tuple[int, int]

def init_v_values(env: Frozen_Lake.FrozenLakeEnv) -> Dict[State, float]:
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0.0
    return v_values


def value_iteration(env: Frozen_Lake.FrozenLakeEnv,
                    v_values: Dict[State, float], gamma: float):
    policy = {} 
    new_v_values = init_v_values(env)
    for state in env.get_all_states():
        next_v_vals = {}
        for action in env.get_possible_actions(state):
            next_val = 0
            for next_state in env.get_next_states(state, action):
                transition_prob = env.get_transition_prob(state, action, next_state)
                reward = env.get_reward(state, action, next_state)
                next_val += transition_prob * reward
                next_val += gamma * transition_prob * v_values[next_state]
            next_v_vals[action] = next_val
        if next_v_vals:
            max_action = max(next_v_vals, key=next_v_vals.get)
            new_v_values[state] = next_v_vals[max_action]
            policy[state] = max_action
        else:
            new_v_values[state] = 0
            policy[state] = "down"
    return new_v_values, policy


def get_trajectory(env: Frozen_Lake.FrozenLakeEnv, policy):
    state = env.reset()
    total_reward = 0
    trajectory_len = 0
    for _ in range(999):
        trajectory_len += 1
        action = policy[state]
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward, trajectory_len


def evaluate(env: Frozen_Lake.FrozenLakeEnv, n_runs: int,
             policy: Dict[State, str]) -> Tuple[float, List[int]]:
    sum_of_rewards = 0
    trajectory_lens = []
    for _ in range(n_runs):
        reward, trajectory_len = get_trajectory(env, policy)
        sum_of_rewards += reward
        trajectory_lens.append(trajectory_len)
    return sum_of_rewards / n_runs, trajectory_lens



def main():
    env = Frozen_Lake.FrozenLakeEnv()
    v_values = init_v_values(env)
    gammas = [0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1]
    n_runs = 1000
    mean_rewards = []
    trajectory_lens_all = []

    for gamma in gammas:
        policy = {}
        for _ in range(10000):
            v_values, policy = value_iteration(env, v_values, gamma)
        print(f"{env.counter = }")
        mean_reward, trajectory_lens = evaluate(env, n_runs, policy)
        print(f"{gamma = }\t{mean_reward = }")
        mean_rewards.append(mean_reward)
        trajectory_lens_all.append(trajectory_lens)

    df_lens = pd.DataFrame({g: l for g, l in zip(gammas, trajectory_lens_all)})
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
