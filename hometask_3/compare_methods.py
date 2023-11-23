import numpy as np
import time
from Frozen_Lake import FrozenLakeEnv
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def get_q_values(v_values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(state, action, next_state) * env.get_reward(state, action, next_state)
                q_values[state][action] += gamma * env.get_transition_prob(state, action, next_state) * v_values[next_state]
    return q_values

def init_policy():
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy


def init_v_values():
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0
    return v_values


def policy_evaluation_step(v_values, policy, gamma):
    q_values = get_q_values(v_values, gamma)
    new_v_values = {}
    for state in env.get_all_states():
        new_v_values[state] = 0
        for action in env.get_possible_actions(state):
            new_v_values[state] += policy[state][action] * q_values[state][action]
    return new_v_values

def policy_evaluation(policy, gamma, eval_iter_n, v_values):
    if v_values is None:
        v_values = init_v_values()
    for _ in range(eval_iter_n):
        v_values = policy_evaluation_step(v_values, policy, gamma)
    q_values = get_q_values(v_values, gamma)
    return q_values


def policy_improvement(q_values):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        argmax_action = None
        max_q_value = float('-inf')
        for action in env.get_possible_actions(state):
            policy[state][action] = 0
            if q_values[state][action] > max_q_value:
                argmax_action = action
                max_q_value = q_values[state][action]
        policy[state][argmax_action] = 1
    return policy


def value_iteration(v_values, gamma):
    new_v_values = {}
    policy = {}
    for state in env.get_all_states():
        v_values_state = {}
        for action in env.get_possible_actions(state):
            next_val = 0
            for next_state in env.get_next_states(state, action):
                transition_prob = env.get_transition_prob(state, action, next_state)
                reward = env.get_reward(state, action, next_state)
                next_val += transition_prob * reward
                next_val += transition_prob * gamma * v_values[next_state]
            v_values_state[action] = next_val

        if v_values_state:
            max_action = max(v_values_state, key=v_values_state.get)
            max_v_val = v_values_state[max_action]
            policy[state] = {}

            for action, v_val in v_values_state.items():
                if v_val == max_v_val:
                    new_v_values[state] = v_val
                    policy[state][action] = 1
                else:
                    policy[state][action] = 0
            new_v_values[state] = max_v_val
        else:
            policy[state] = {"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25}
            new_v_values[state] = 0
    return new_v_values, policy

def get_trajectory(policy):
    total_reward = 0
    state = env.reset()
    for _ in range(1000):
        action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break
    return total_reward

def evaluate(n_runs, policy):
    sum_rewards = 0
    for _ in range(n_runs):
        sum_rewards += get_trajectory(policy)
    return sum_rewards / n_runs

env_calls = []
rewards = []

iter_n = 100
eval_iter_n = 100
gamma = .9999

env = FrozenLakeEnv()
policy = init_policy()
train_rewards = []
for _ in tqdm(range(iter_n), colour="blue", leave=False):
    q_values = policy_evaluation(policy, gamma, eval_iter_n, v_values = None)
    policy = policy_improvement(q_values)
reward = evaluate(1000, policy)
env_calls.append(env.counter)
rewards.append(reward)
print(f"{env.counter = }")
print(f"{reward = }")

iter_n = 70
eval_iter_n = 100
gamma = .9999

env = FrozenLakeEnv()
policy = init_policy()
train_rewards = []
v_values = init_v_values()
for _ in tqdm(range(iter_n), colour="blue", leave=False):
    q_values = policy_evaluation(policy, gamma, eval_iter_n, v_values)
    policy = policy_improvement(q_values)
reward = evaluate(1000, policy)
env_calls.append(env.counter)
rewards.append(reward)
print(f"{env.counter = }")
print(f"{reward = }")

env = FrozenLakeEnv()
v_vals = init_v_values()
gamma = 0.9999
for _ in range(150):
    v_vals, policy = value_iteration(v_vals, gamma)
reward = evaluate(1000, policy)
env_calls.append(env.counter)
rewards.append(reward)
print(f"{env.counter = }")
print(f"{reward = }")


x = ["task 1", "task 2", "task3"]
with plt.style.context("dark_background"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    ax1.bar(x=x, height=env_calls)
    for i in range(3):
        ax1.text(i, env_calls[i], env_calls[i], ha="center", fontsize=18)
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True)
    plt.xticks(fontsize=14)
    ax1.set_title("N of env calls by method", fontsize=20)
    ax1.set_ylabel("N of env calls with best hyperparameters", fontsize=14) 

    ax2.bar(x=x, height=rewards)
    for i in range(3):
        ax2.text(i, rewards[i], rewards[i], ha="center", fontsize=18)
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = True, bottom = True)
    plt.xticks(fontsize=14)
    ax2.set_title("Mean reward with 1000 runs", fontsize=20)
    ax2.set_ylabel("Mean reward", fontsize=14) 
    plt.show()
