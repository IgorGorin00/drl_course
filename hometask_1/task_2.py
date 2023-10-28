import time
import warnings
import numpy as np
import gym
from typing import List
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    total_reward: int

    def print(self):
        print("States", self.states)
        print("Actions", self.actions)
        print("Total reward", self.total_reward)


class crossEntropyAgent():
    def __init__(self, state_n: int, action_n: int) -> None:
        self.state_n = state_n
        self.action_n = action_n
        self.policy = np.ones((state_n, action_n)) / action_n

    def get_action(self, state: int) -> int:
        action_probs = self.policy[state]
        return np.random.choice(np.arange(self.action_n), p=action_probs)

    def update_policy(self, elite_trajectories: List[Trajectory],
                      method: str = "default",
                      laplace_smoothing: int = 0,
                      policy_lambda: float = 0.0) -> None:

        new_policy = np.zeros((self.state_n, self.action_n))

        for t in elite_trajectories:
            for state, action in zip(t.states, t.actions):
                new_policy[state][action] += 1
        if method == "default":
            for state in range(self.state_n):
                s = np.sum(new_policy[state])
                if s:
                    new_policy[state] /= s
                else:
                    new_policy[state] = self.policy[state]

        elif method == "laplace_smoothing":
            if not laplace_smoothing:
                raise RuntimeError("Smoothing factor is 0!")

            for state in range(self.state_n):
                s = np.sum(new_policy[state])
                if s:
                    new_policy[state] += laplace_smoothing
                    new_policy[state] /= (s + laplace_smoothing * self.action_n)

                else:
                    new_policy[state] = self.policy[state]

        elif method == "policy_smoothing":
            if not policy_lambda:
                raise RuntimeError("Policy lambda is 0.0!")
            new_policy *= policy_lambda
            self.policy *= (1 - policy_lambda)
            new_policy += self.policy
            s = np.sum(new_policy, axis=1).reshape(-1, 1)
            new_policy /= s

        self.policy = new_policy


def get_trajectory(
        env: gym.wrappers.time_limit.TimeLimit,
        agent: crossEntropyAgent,
        trajectory_len: int,
        viz: bool = False, delay: int = 0
        ) -> Trajectory:

    states, actions, total_reward = [], [], 0
    state = env.reset()
    for i in range(trajectory_len):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        if viz:
            env.render()
            time.sleep(delay)
        total_reward += reward
        if done:
            break
        state = next_state
    trajectory = Trajectory(np.array(states), np.array(actions), total_reward)
    return trajectory


def get_elite_trajectories(
        trajectories: List[Trajectory],
        epoch_rewards: np.ndarray,
        q_param: float
        ) -> List[Trajectory]:
    q_value = np.quantile(epoch_rewards, q_param)
    return [t for t in trajectories if t.total_reward > q_value]


@dataclass
class History:
    mean_rewards: List[int] = field(default_factory=list)
    elite_trajectories_ns: List[int] = field(default_factory=list)
    rewards: List[np.ndarray] = field(default_factory=list)

    def print(self):
        print("Rewards:\n", self.mean_rewards)
        print()
        print("Elite trajectories n:\n", self.elite_trajectories_ns)
        print()

    def show(self, title: str, save: bool = False) -> None:
        with plt.style.context("dark_background"):
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                                           figsize=(16, 9), sharex=True)
            x = np.arange(len(self.mean_rewards))
            ax1.scatter(x=x, y=self.mean_rewards, color="red", zorder=3)
            ax1.plot(self.mean_rewards, color="red", zorder=2)
            sns.boxplot(data=self.rewards, color="cyan",
                        fill=False, ax=ax1, zorder=1)
            ax1.set_title("Rewards per epoch")
            ax1.set_ylabel("Reward")

            ax2.plot(self.elite_trajectories_ns, color="red")
            ax2.scatter(x=x, y=self.elite_trajectories_ns, color="red")
            ax2.set_title("N of elite trajectories per epoch")
            ax2.set_ylabel("N")

            if save:
                plt.savefig(f"{title}.png")
            plt.show()


def train(env: gym.wrappers.time_limit.TimeLimit,
          agent: crossEntropyAgent,
          n_epochs: int,
          trajectory_len: int,
          traj_per_epoch: int,
          q_param: float,
          method: str = "default",
          laplace_smoothing: int = 0,
          policy_lambda: float = 0.5
          ) -> History:
    mean_rewards = []
    elite_trajectories_ns = []
    rewards = []
    for epoch in range(n_epochs):
        trajectories = [get_trajectory(env, agent, trajectory_len)
                        for _ in range(traj_per_epoch)]

        if epoch > n_epochs * 0.3:
            laplace_smoothing -= 0.05
            laplace_smoothing = max(laplace_smoothing, 0.1)


        epoch_rewards = np.array([t.total_reward for t in trajectories])
        elite_trajectories = get_elite_trajectories(trajectories,
                                                    epoch_rewards, q_param)
        agent.update_policy(elite_trajectories, method=method,
                            laplace_smoothing=laplace_smoothing,
                            policy_lambda=policy_lambda)

        epoch_mean_reward = np.round(np.mean(epoch_rewards), 2)
        elite_trajectories_n = len(elite_trajectories)
        mean_rewards.append(epoch_mean_reward)
        elite_trajectories_ns.append(elite_trajectories_n)
        rewards.append(epoch_rewards)
        print(f"[{epoch = }]\t[{epoch_mean_reward = }]\
                \t[{elite_trajectories_n = }]")
    history = History(mean_rewards=mean_rewards,
                      elite_trajectories_ns=elite_trajectories_ns,
                      rewards=rewards)
    return history


def main():
    STATE_N = 500
    ACTION_N = 6
    env = gym.make("Taxi-v3")
    agent = crossEntropyAgent(STATE_N, ACTION_N)
    trajecotry_len = 500
    q_param = 0.8
    traj_per_epoch = 500
    n_epochs = 30

#    method = "laplace_smoothing"
    laplace_smoothing = 0

    method = "policy_smoothing"
    policy_lambda = 0.85

    history = train(env, agent, n_epochs,
                    trajecotry_len, traj_per_epoch, q_param,
                    method=method,
                    laplace_smoothing=laplace_smoothing,
                    policy_lambda=policy_lambda)
    last_trajecotry = get_trajectory(env, agent,
                                     trajecotry_len, viz=True, delay=0.1)
    print(last_trajecotry.total_reward)
    history.show("task_2", save=True)
    history.print()


if __name__ == "__main__":
    main()
