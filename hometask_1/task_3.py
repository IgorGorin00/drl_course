# -*- coding: utf-8 -*-
"""gym-taxi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1irDIBtwVsXeaDMnfU0t03Q1a1Qspk8HE
"""

# Commented out IPython magic to ensure Python compatibility.
import gym
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

@dataclass
class Trajectory():
    states: np.ndarray
    actions: np.ndarray
    total_reward: int

    def print(self):
        print("States:\n", self.states)
        print("\nActions:\n", self.actions)
        print("\nTotal reward:\n", self.total_reward)

class Agent():
    def __init__(self, state_n: int, action_n: int) -> None:
        self.state_n = state_n
        self.action_n = action_n
        self.policy = np.ones((state_n, action_n)) / action_n
        self.possible_actions = np.arange(action_n)

    def get_action(self, state: int) -> int:
        probs = self.policy[state]
        return np.random.choice(self.possible_actions, p=probs)

    def update_policy(self, elite_trajectories: list[Trajectory]) -> None:
        new_policy = np.zeros((self.state_n, self.action_n))
        for t in elite_trajectories:
            for state, action in zip(t.states, t.actions):
                new_policy[state][action] += 1
        for state in range(self.state_n):
            s = np.sum(new_policy[state])
            if s:
                new_policy[state] /= s
            else:
                new_policy[state] = self.policy[state]
        self.policy = new_policy

    def update_policy_by_policies(self, elite_policies: list[list[int]]) -> None:
        new_policy = np.zeros((self.state_n, self.action_n))
        for p in elite_policies:
            for state, action in enumerate(p):
                new_policy[state][action] += 1
        for state in range(self.state_n):
            s = np.sum(new_policy[state])
            if s:
                new_policy[state] /= s
            else:
                new_policy[state] = self.policy[state]
        self.policy = new_policy

def sample_det_policies(agent: Agent, n_samples: int) -> list[list[int]]:
    det_policies = []
    for _ in range(n_samples):
        det_policy = []
        for state in range(agent.state_n):
            action = agent.get_action(state)
            det_policy.append(action)
        det_policies.append(det_policy)
    return det_policies

def get_trajectory_with_det_policy(env: gym.wrappers.time_limit.TimeLimit,
                                   policy: list[int], traj_len: int) -> Trajectory:
    states = []
    actions = []
    total_reward = 0
    state = env.reset()
    for _ in range(traj_len):
        states.append(state)
        action = policy[state]
        actions.append(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return Trajectory(
        states=np.array(states),
        actions=np.array(actions),
        total_reward=total_reward
    )

def get_trajectory(env: gym.wrappers.time_limit.TimeLimit, agent: Agent, traj_len: int) -> Trajectory:
    states = []
    actions = []
    total_reward = 0

    state = env.reset()
    for _ in range(traj_len):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return Trajectory(
        states=np.array(states),
        actions=np.array(actions),
        total_reward=total_reward
    )

def get_elite_trajectories(trajecotries: list[Trajectory], epoch_rewards: list[int], q_param: float) -> list[Trajectory]:
    q_value = np.quantile(epoch_rewards, q=q_param)
    return [t for t in trajecotries if t.total_reward > q_value]

def get_elite_t_chunks(trajectories: list[list[Trajectory]], policy_mean_rewards: list[int], q_param: float) -> list[Trajectory]:
    q_value = np.quantile(policy_mean_rewards, q=q_param)
    elite_trajectories = []
    for t_chunk, pol_reward in zip(trajectories, policy_mean_rewards):
        if pol_reward > q_value:
            elite_trajectories.extend(t_chunk)
    return elite_trajectories

def get_elite_policies(policies: list[list[int]], policies_rewards: list[int], q_param: float):
    elite_policies = []
    q_value = np.quantile(policies_rewards, q=q_param)
    for policy, reward in zip(policies, policies_rewards):
        if reward > q_value:
            elite_policies.append(policy)
    return elite_policies

@dataclass
class History():
    epoch_rewards: np.ndarray
    epoch_mean_rewards: np.ndarray
    elite_policies_ns: np.ndarray

    def show(self):
        x = np.arange(len(self.epoch_mean_rewards))
        with plt.style.context("dark_background"):
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
            ax1.plot(self.epoch_mean_rewards, zorder=2)
            ax1.scatter(x=x, y=self.epoch_mean_rewards, zorder=3)
            sns.boxplot(data=self.epoch_rewards, zorder=1, ax=ax1)
            ax1.set_title("rewards")

            ax2.plot(self.elite_policies_ns, zorder=1)
            ax2.scatter(x=x, y=self.elite_policies_ns, zorder=2)
            ax2.set_title("elite policies ns")

def train(env: gym.wrappers.time_limit.TimeLimit, agent: Agent, n_epochs: int, n_policies: int, traj_per_policy: int, traj_len: int, q_param: float):
    epoch_rewards = []
    epoch_mean_rewards = []
    elite_policies_ns = []
    for epoch in range(n_epochs):
        det_policies = sample_det_policies(agent, n_policies)
        det_policies_rewards = []
        trajectories = []
        for det_policy in det_policies:
            policy_rewards = []
            policy_trajectories = []
            for _ in range(traj_per_policy):
                traj = get_trajectory_with_det_policy(env, det_policy, traj_len)
                policy_rewards.append(traj.total_reward)
                policy_trajectories.append(traj)
            trajectories.append(policy_trajectories)
            policy_mean_reward = np.mean(np.array(policy_rewards))
            det_policies_rewards.append(policy_mean_reward)
        # elite_trajectories = get_elite_t_chunks(trajectories, det_policies_rewards, q_param)
        # agent.update_policy(elite_trajectories)
        # elite_traj_n = len(elite_trajectories)
        elite_policies = get_elite_policies(det_policies, det_policies_rewards, q_param)
        elite_policies_n = len(elite_policies)
        agent.update_policy_by_policies(elite_policies)

        val_traj = get_trajectory(env, agent, traj_len)
        epoch_mean_reward = np.mean(np.array(det_policies_rewards))
        epoch_mean_rewards.append(epoch_mean_reward)
        epoch_rewards.append(det_policies_rewards)
        elite_policies_ns.append(elite_policies_n) #elite_traj_n)#
        val_traj_reward = val_traj.total_reward
        print(f"{epoch = }, {epoch_mean_reward = }, {elite_policies_n = }, {val_traj_reward = }")
    return History(
        epoch_rewards=np.array(epoch_rewards),
        epoch_mean_rewards=np.array(epoch_mean_rewards),
        elite_policies_ns=np.array(elite_policies_ns)
        )

env = gym.make("Taxi-v3")

STATE_N = 500
ACTION_N = 6

agent = Agent(STATE_N, ACTION_N)

n_epochs = 50
n_policies = 200
traj_per_policy = 30
traj_len = 200
q_param = 0.75

history = train(env, agent, n_epochs, n_policies, traj_per_policy, traj_len, q_param)

history.show()
