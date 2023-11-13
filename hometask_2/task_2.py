from dataclasses import dataclass
import gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    total_reward: int

    def print(self) -> None:
        print("\nStates:\n", self.states)
        print("\nActions:\n", self.actions)
        print("\nTotal reward:\n", self.total_reward)


class crossEntropyAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(crossEntropyAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
                nn.Linear(self.state_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, self.action_dim),
                )
        self.loss_f = nn.MSELoss()
        self.exploration_rate = 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state)
        action = self.forward(state).detach().numpy()
        mu, sigma = 0, 2
        s = np.random.normal(mu, sigma)
        # s = np.random.uniform(-1., 1.) * self.exploration_rate
        action += s * self.exploration_rate
        return action

    def training_step(self, elite_trajectories: list[Trajectory]) -> torch.Tensor:
        elite_states = torch.FloatTensor(
                np.concatenate([t.states for t in elite_trajectories]))
        elite_actions = torch.FloatTensor(
                np.concatenate([t.actions for t in elite_trajectories]))
        preds = self.forward(elite_states)
        return self.loss_f(preds, elite_actions)


def get_trajectory(env: gym.wrappers.time_limit.TimeLimit,
                   agent: crossEntropyAgent,
                   trajectory_len: int,
                   viz: bool = False) -> Trajectory:
    states, actions, total_reward = [], [], 0
    state = env.reset()
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        states.append(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if viz:
            env.render()
        if done:
            break
#    total_reward -= len(states)
    return Trajectory(
            states=np.array(states),
            actions=np.array(actions),
            total_reward=total_reward)


def get_elite_trajectories(trajectories: list[Trajectory],
                           epoch_rewards: np.ndarray,
                           q_param: float) -> list[Trajectory]:
    q_value = np.quantile(epoch_rewards, q_param)
    print(f"{q_value = }")
    return [t for t in trajectories if t.total_reward > q_value]


@dataclass
class History:
    epoch_mean_rewards: np.ndarray
    epoch_rewards: list[np.ndarray]
    epoch_losses: np.ndarray
    elite_trajectory_ns: np.ndarray
    val_rewards: np.ndarray

    def show(self) -> None:
        with plt.style.context("dark_background"):
            x = np.arange(len(self.epoch_mean_rewards))
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16, 9))
            ax1.plot(self.epoch_mean_rewards, color="#FF0000", zorder=2)
            ax1.scatter(x=x, y=self.epoch_mean_rewards, zorder=3)
            sns.boxplot(data=self.epoch_rewards, ax=ax1, zorder=1)
            ax1.set_title("epoch rewards")

            ax2.plot(self.epoch_losses)
            ax2.set_title("epoch losses")

            ax3.plot(self.val_rewards)
            ax3.set_title("Validation trajectory rewards")

            plt.show()


def train(env: gym.wrappers.time_limit.TimeLimit,
          agent: crossEntropyAgent,
          n_epochs: int,
          traj_per_epoch: int,
          trajectory_len: int,
          q_param: float,
          lr: float,
          decrease_after: int) -> History:
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    epoch_mean_rewards = []
    all_epoch_rewards = []
    epoch_losses = []
    epoch_trajectory_ns = []
    val_rewards = []
    decrease_rate = agent.exploration_rate / (n_epochs - decrease_after)
    for epoch in range(n_epochs):
        if epoch > decrease_after:
            agent.exploration_rate -= decrease_rate
        trajectories = [get_trajectory(env, agent, trajectory_len) for _ in tqdm(range(traj_per_epoch), colour="blue", leave=False)]
        epoch_rewards = np.array([t.total_reward for t in trajectories])
        elite_trajectories = get_elite_trajectories(trajectories, epoch_rewards, q_param)

        if len(elite_trajectories) > 0:
            loss = agent.training_step(elite_trajectories)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss = loss.detach().item()
        else:
            loss = -1

        epoch_mean_reward = np.mean(epoch_rewards)
        elite_trajectory_n = len(elite_trajectories)

        epoch_mean_rewards.append(epoch_mean_reward)
        all_epoch_rewards.append(epoch_rewards)
        epoch_losses.append(loss)
        epoch_trajectory_ns.append(elite_trajectory_n)

        viz = False
        if epoch % 10 == 0:
            viz = True
        val_trajectory = get_trajectory(env, agent, trajectory_len, viz=viz)

        epoch_mean_reward = round(epoch_mean_reward, 2)
        loss = round(loss, 2)
        val_reward = val_trajectory.total_reward  # + len(val_trajectory.states)
        val_reward = round(val_reward, 2)
        val_rewards.append(val_reward)

        print(f"{epoch = }, {epoch_mean_reward = }, {elite_trajectory_n = }, {loss = }, {val_reward = }, {agent.exploration_rate = }")

    return History(
            epoch_mean_rewards=np.array(epoch_mean_rewards),
            epoch_rewards=all_epoch_rewards,
            epoch_losses=np.array(epoch_losses),
            elite_trajectory_ns=np.array(epoch_trajectory_ns),
            val_rewards=np.array(val_rewards))


def main():
    env = gym.make("MountainCarContinuous-v0")
    STATE_DIM = 2
    ACTION_DIM = 1
    agent = crossEntropyAgent(STATE_DIM, ACTION_DIM)
    n_epochs = 100
    traj_per_epoch = 100
    trajectory_len = 1000
    q_param = 0.95
    lr = 0.05
    mu, sigma = 0, 2
    decrease_after = 10
    history = train(env, agent, n_epochs, traj_per_epoch, trajectory_len, q_param, lr, decrease_after)
    last_trajectory = get_trajectory(env, agent, trajectory_len=999, viz=True)
    env.close()
    history.show()
    last_trajectory.print()
    print("\nLast trajectory reward:\n", last_trajectory.total_reward)


if __name__ == "__main__":
    main()
