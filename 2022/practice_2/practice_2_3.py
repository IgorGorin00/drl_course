import warnings
import gym
import numpy as np
import time
import torch
import torch.nn as nn
warnings.filterwarnings(
    action='ignore',
    module='gym')


class CEM(nn.Module):
    def __init__(self, state_dim, action_n):
        super(CEM, self).__init__()
        self.state_dim = state_dim
        self.action_n = action_n

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, self.action_n),
        )

        self.softmax = nn.Softmax(dim=0)
        self.loss_f = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.net(x)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self(state)
        action_probs = self.softmax(logits)
        action = np.random.choice(
            self.action_n, p=action_probs.detach().numpy())
        return action

    def update_policy(self, elite_trajectories):
        elite_states, elite_actions = [], []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        loss = self.loss_f(self.forward(elite_states), elite_actions)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss.item()


def get_trajectory(env, agent, trajectory_len=500, viz=False):
    trajectory = {'states': [], 'actions': [], 'total_reward': 0}
    state = env.reset()
    trajectory['states'].append(state)
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _ = env.step(action)
        trajectory['total_reward'] += reward
        if done:
            break
        trajectory['states'].append(state)
    if viz:
        state = env.reset()
        env.render()
        time.sleep(2)
        for _ in range(trajectory_len):
            action = agent.get_action(state)

            state, reward, done, _ = env.step(action)
            env.render()
            time.sleep(0.15)
            if done:
                break
        for key, val in trajectory.items():
            if key != 'states':
                print(key, val)
                print()
    return trajectory


def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    q_value = np.quantile(total_rewards, q_param)
    mean_reward = np.mean(total_rewards)
    return mean_reward, [trajectory for trajectory in trajectories if trajectory['total_reward'] > q_value]


def train(epochs, traj_per_epoch, trajectory_len, env, agent, q_param):
    loss = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        trajectories = [get_trajectory(env, agent, trajectory_len)
                        for _ in range(traj_per_epoch)]
        mean_reward, elite_trajectories = get_elite_trajectories(
            trajectories, q_param)
        if len(elite_trajectories) > 0:
            loss = agent.update_policy(elite_trajectories)
        if epoch % 20 == 0:
            print(f'{epoch=} \t\t {loss=} \t\t {mean_reward=}')
    end = time.perf_counter()
    print(f'Training took {round(end-start, 4)} secs')


def main():
    env = gym.make('CartPole-v1')
    state_dim = 4
    action_n = 2
    agent = CEM(state_dim, action_n)

    epochs = 201
    traj_per_epoch = 20
    trajectory_len = 500
    q_param = 0.85

    train(epochs, traj_per_epoch, trajectory_len, env, agent, q_param)
    get_trajectory(env, agent, viz=True)


if __name__ == '__main__':
    main()
