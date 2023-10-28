import gym
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


class CEM():
    def __init__(self, state_n, action_n, possible_actions):
        self.state_n = state_n                  # number of possble states
        self.action_n = action_n                # number of possible actions
        # matrix that presents probs of choosing an action for particular state
        self.policy = possible_actions

    def get_action(self, state):
        '''Choosing an action based on its probability in self.policy'''
        return int(np.random.choice(self.action_n, p=self.policy[state]))

    def update_policy(self, elite_trajectories, laplace_smoothing=False, policy_lambda=False):
        '''Updates current policy based on elite trajectories by modifying action probs'''

        pre_policy = np.zeros((self.state_n, self.action_n))
        # extract actions from elite trajectories
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                pre_policy[state][action] += 1
        # update policy
        for state in range(self.state_n):

            # change probs if action in elite trajectories
            if sum(pre_policy[state]) != 0:
                if laplace_smoothing:
                    self.policy[state] = (
                        pre_policy[state] + self.policy[state]) / (sum(pre_policy[state]) + 1)
                elif policy_lambda:
                    if policy_lambda > 1 or policy_lambda <= 0:
                        raise ValueError(
                            "Policy lambda must be in range (0, 1]")
                    pre_policy[state] /= sum(pre_policy[state])
                    self.policy[state] = (
                        pre_policy[state] * policy_lambda) + (self.policy[state] * (1 - policy_lambda))
                else:
                    self.policy[state] = pre_policy[state] / \
                        sum(pre_policy[state])


def get_possible_actions(env, state_n, action_n):
    possible_actions = np.zeros((state_n, action_n))
    for state in range(state_n):
        possible_actions[state] = env.action_mask(
            state) / sum(env.action_mask(state))
    return possible_actions


def get_trajectory(env, agent):
    trajectory = {'states': [], 'actions': [], 'max_reward': 0}
    max_trajectory_len = 200
    state = env.reset()
    trajectory['states'] += [state]

    for i in range(max_trajectory_len):
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)

        trajectory['actions'] += [action]
        trajectory['states'] += [state]
        trajectory['max_reward'] += reward

        if done:
            break
    return trajectory


def get_elite_trajectories(trajectories, q):

    rewards = [trajectory['max_reward'] for trajectory in trajectories]
    q_value = np.quantile(a=rewards, q=q)

    return np.mean(rewards), [trajectory for trajectory in trajectories if trajectory['max_reward'] > q_value]


def train(n_epochs, n_traj_per_epoch, env, agent, q=0.9):

    start_training = time.perf_counter()
    history = []
    mean_reward = float('-inf')
    for epoch in range(n_epochs):

        # generate trajectories
        trajectories = [get_trajectory(env, agent)
                        for _ in range(n_traj_per_epoch)]

        mean_reward, elite_trajectories = get_elite_trajectories(
            trajectories, q)

        if len(elite_trajectories) > 0:
            agent.update_policy(elite_trajectories, policy_lambda=0.6)

        history += [mean_reward]
        if epoch % 10 == 0:
            print(f'{epoch=} \t\t train {mean_reward=}')

    end_training = time.perf_counter()
    print(f'Training took {end_training - start_training} secs')
    return history


def main():

    env = gym.make('Taxi-v3')
    print('env created')
    action_n = 6
    state_n = 500
    possible_actions = get_possible_actions(env, state_n, action_n)
    agent = CEM(state_n, action_n, possible_actions)

    n_epochs = 150
    n_traj_per_epoch = 1000

    history = train(n_epochs, n_traj_per_epoch, env, agent)

    state = env.reset()
    env.render()
    time.sleep(2)
    done = False
    while not done:
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.5)


if __name__ == '__main__':
    main()
