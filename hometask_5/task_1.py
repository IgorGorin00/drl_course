# Compare performance of deep cross-entropy and classic DQN on 
# Acrobot-v1 (continuous observation space, discrete action space)
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import gym
from random import sample
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


@dataclass
class Trajectory:
    states: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]
    total_reward: int


class crossEntropyMethod(nn.Module):
    def __init__(self, env, q_param: float, lr: float):
        super(crossEntropyMethod, self).__init__()
        
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_n = self.env.action_space.n
        self.t_max = 500
        
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, self.action_n)
        )
        self.softmax = nn.Softmax(dim=0)
        self.loss_f = nn.CrossEntropyLoss()
        self.q_param = q_param
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def get_action(self, state: npt.NDArray[np.float32]) -> int:
        logits = self.forward(torch.FloatTensor(state))
        probs = self.softmax(logits).detach().numpy()
        return np.random.choice(self.action_n, p=probs)
    
    def update_policy(self,
                      elite_states: npt.NDArray[np.float32],
                      elite_actions: list[int]):
        states = torch.FloatTensor(elite_states)
        
        actions = torch.LongTensor(elite_actions)
        
        preds = self.forward(states)
        loss = self.loss_f(preds, actions)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
    
    def get_trajectory(self) -> Trajectory:
        states, actions, total_reward = [], [], 0        
        state = self.env.reset()
        for _ in range(self.t_max):
            action = self.get_action(state)
            states.append(state)
            actions.append(float(action))
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return Trajectory(np.array(states), np.array(actions), total_reward)
    
    def get_elite_states_and_actions(self, states, actions, rewards, q_param
                                     )-> tuple[
                                             list[npt.NDArray[np.float32]], 
                                             list[int]
                                             ]:
        q_value = np.quantile(rewards, q_param)
        elite_states, elite_actions = [], []
        for s, a, r, in zip(states, actions, rewards):
            if r > q_value:
                elite_states.extend(s)
                elite_actions.extend(a)
        return elite_states, elite_actions
    
    def train(self, epoch_n: int, traj_per_epoch: int) -> npt.NDArray[np.float64]:
        all_rewards = np.zeros(epoch_n * traj_per_epoch)
        for epoch in tqdm(range(epoch_n), colour="blue"):
            epoch_states, epoch_actions, epoch_rewards =\
                    [], [], np.zeros(traj_per_epoch)
            for t_idx in range(traj_per_epoch):
                t = self.get_trajectory()
                states, actions, total_reward = t.states, t.actions, t.total_reward
                epoch_states.append(states)
                epoch_actions.append(actions)
                epoch_rewards[t_idx] = total_reward
            elite_states, elite_actions = self.get_elite_states_and_actions(
                    epoch_states, epoch_actions, epoch_rewards, self.q_param)
            if elite_states:
                elite_states = np.array(elite_states)
                self.update_policy(elite_states, elite_actions)
            
            all_rewards[epoch * traj_per_epoch : (epoch + 1) * traj_per_epoch]=\
                    epoch_rewards
        return all_rewards


class qFunction(nn.Module):
    def __init__(self, state_dim, action_dim):
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
            done: bool, next_state: npt.NDArray[np.float64]
            ) -> torch.Tensor | None:
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


def train_dqn(env, agent, episode_n: int, lr: float,
              epsilons: npt.NDArray[np.float64], t_max: int = 500):
    opt = torch.optim.Adam(agent.q_function.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=1, steps_per_epoch=episode_n)

    episode_rewards = np.zeros(episode_n)
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
            if done:
                break
            state = next_state
        episode_rewards[episode] = episode_reward
        lr_sched.step()
        if (episode + 1) / 70 == 0:
            print(f"{episode = } \t {episode_reward = }")
    return episode_rewards


def median_of_segments(array, segment_size=70):
    reshaped_array = array[:len(array) - len(array) % segment_size].reshape(-1, segment_size)
    medians = np.median(reshaped_array, axis=1)
    return medians


def main():
    env = gym.make("Acrobot-v1")
    ce_method = crossEntropyMethod(env, q_param=0.85, lr=1e-3)
    print("training cross-entropy")
    ce_rewards = ce_method.train(epoch_n=100, traj_per_epoch=70)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    episode_n = 7_000
    lr = 1e-3
    epsilons = np.exp(np.linspace(0, -5, episode_n))
    agent = deepQNetwork(state_dim, action_dim)
    print("training dqn")
    dqn_history = train_dqn(env, agent, episode_n, lr, epsilons)

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
        ax1.plot(ce_rewards, label="Depp cross-entropy")
        ax1.plot(dqn_history, label="Deep Q Network")
        ax1.axhline(-100, color="red", label="-100 reward treshold")
        ax1.set_xlabel("Trajectory N")
        ax1.set_ylabel("Reward")
        ax1.set_title("Rewards comparison for acrobot")
        ax1.legend()

        ax2.plot(median_of_segments(ce_rewards), label="Depp cross-entropy")
        ax2.plot(median_of_segments(dqn_history), label="Deep Q Network")
        ax2.axhline(-100, color="red", label="-100 reward treshold")
        ax2.set_xlabel("Trajectory N")
        ax2.set_ylabel("Reward")
        ax2.set_title("Smoothed rewards comparison for acrobot (meadian of 70 elements)")
        ax2.legend()
        plt.show()    


if __name__ == "__main__":
    main()

