from dataclasses import dataclass
import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


@dataclass
class Trajectory:
    states: np.ndarray
    actions: np.ndarray
    total_reward: int

    def print(self):
        print("States:\n", self.states)
        print("\nActions:\n", self.actions)
        print("\nTotal reward:\n", self.total_reward)


class crossEntropyAgent(nn.Module):
    def __init__(self, state_dim: int, action_n: int):
        super(crossEntropyAgent, self).__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.net = nn.Sequential(
                nn.Linear(self.state_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, self.action_n))
        self.softmax = nn.Softmax()
        self.possible_acitons = np.arange(self.action_n)
        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, state: np.ndarray) -> int:
        logits = self.forward(torch.FloatTensor(state))
        action_probs = self.softmax(logits).detach().numpy()
        return np.random.choice(self.possible_acitons, p=action_probs)

    def training_step(self,
                      elite_trajectories: list[Trajectory]) -> torch.Tensor:
        elite_states = torch.FloatTensor(
                np.concatenate([t.states for t in elite_trajectories]))
        elite_actions = torch.LongTensor(
                np.concatenate([t.actions for t in elite_trajectories]))
        preds = self.forward(elite_states)
        return self.loss_f(preds, elite_actions)


def get_trajectory(env: gym.wrappers.time_limit.TimeLimit,
                   agent: crossEntropyAgent,
                   trajectory_len: int,
                   viz: bool = False
                   ) -> Trajectory:
    states = []
    actions = []
    total_reward = 0
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
    return Trajectory(
            states=np.array(states),
            actions=np.array(actions),
            total_reward=total_reward)


def get_elite_trajectories(trajectories: list[Trajectory],
                           epoch_rewards: np.ndarray,
                           q_param: float) -> list[Trajectory]:
    q_value = np.quantile(epoch_rewards, q_param)
    print(f"{q_param = }, {q_value = }")
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
            ax3.set_title("Val trajectory rewards")

            plt.show()


def train(env: gym.wrappers.time_limit.TimeLimit,
          agent: crossEntropyAgent,
          n_epochs: int,
          traj_per_epoch: int,
          trajectory_len: int,
          q_param: float,
          lr: float) -> History:
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    epoch_mean_rewards = []
    all_epoch_rewards = []
    epoch_losses = []
    epoch_trajectory_ns = []
    val_rewards = []
    for epoch in range(n_epochs):
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
        val_trajectory = get_trajectory(env, agent, trajectory_len)
        val_rewards.append(val_trajectory.total_reward)
        epoch_mean_reward = round(epoch_mean_reward, 2)
        loss = round(loss, 2)
        val_reward = round(val_trajectory.total_reward, 2)

        print(f"{epoch = }, {epoch_mean_reward = }, {elite_trajectory_n = }, {loss = }, {val_reward = }")

    return History(
            epoch_mean_rewards=np.array(epoch_mean_rewards),
            epoch_rewards=all_epoch_rewards,
            epoch_losses=np.array(epoch_losses),
            elite_trajectory_ns=np.array(epoch_trajectory_ns),
            val_rewards=np.array(val_rewards))


def main():
    env = gym.make("LunarLander-v2")
    STATE_DIM = 8
    ACTION_N = 4
    agent = crossEntropyAgent(STATE_DIM, ACTION_N)
    n_epochs = 50
    traj_per_epoch = 250
    trajectory_len = 300
    q_param = 0.7
    lr = 0.05
    history = train(env, agent, n_epochs, traj_per_epoch, trajectory_len, q_param, lr)
    last_trajectory = get_trajectory(env, agent, trajectory_len, viz=True)
    env.close()
    print("Last trajectory reward", last_trajectory.total_reward)
    history.show()


if __name__ == "__main__":
    main()


