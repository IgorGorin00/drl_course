from typing import List
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataclasses import dataclass
import time


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
                      laplace_smoothing: float = 0.,
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
        ) -> Trajectory:

    states, actions, total_reward = [], [], 0
    state = env.reset()
    for _ in range(trajectory_len):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        state = next_state
    return Trajectory(np.array(states), np.array(actions), total_reward)


def get_elite_trajectories(
        trajectories: List[Trajectory],
        epoch_rewards: np.ndarray,
        q_param: float
        ) -> List[Trajectory]:
    q_value = np.quantile(epoch_rewards, q_param)
    return [t for t in trajectories if t.total_reward > q_value]



def train_ce(env: gym.wrappers.time_limit.TimeLimit,
          agent: crossEntropyAgent,
          n_epochs: int,
          trajectory_len: int,
          traj_per_epoch: int,
          q_param: float,
          method: str = "default",
          laplace_smoothing: float = 0.,
          policy_lambda: float = 0.5
          ) -> List[int]:
    rewards = []
    for epoch in tqdm(range(n_epochs), leave=False, colour="blue"):
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
        rewards.extend(epoch_rewards)
    return rewards 


def get_epsilon_greedy_action(q_values: np.ndarray,
                              epsilon: float, action_n: int) -> int:
    argmax_action = np.argmax(q_values)
    probs = epsilon * np.ones(action_n) / action_n
    probs[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=probs)
    return action


def MonteCarlo(env: gym.wrappers.time_limit.TimeLimit,
               episode_n: int, t_max=500, gamma=0.99) -> List[int]:
    state_n = env.observation_space.n
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))
    n_values = np.zeros((state_n, action_n))
    epsilon = 1

    total_rewards = []
    for _ in tqdm(range(episode_n), leave=False, colour="blue"):
        states, actions, rewards = [], [], []

        state = env.reset()
        for _ in range(t_max):
            states.append(state)
            action = get_epsilon_greedy_action(q_values[state], epsilon,
                                               action_n)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break


        total_rewards.append(sum(rewards))
        g_values = np.zeros(len(rewards))
        g_values[-1] = rewards[-1]
        for t in range(len(rewards) - 2, -1, -1):
            g_values[t] = rewards[t] + gamma * g_values[t + 1]

        for t in range(len(rewards)):
            q_values[states[t]][actions[t]] +=\
                    (g_values[t] - q_values[states[t]][actions[t]])\
                    / (n_values[states[t]][actions[t]] + 1)
            n_values[states[t]][actions[t]] += 1

        epsilon -= 1 / episode_n
    
    return total_rewards




#   SARSA Algorithm
#   Let Q(s, a) = 0 and epsilon = 1
#   For each episode k do:
#       while episode is not over:
#       1.  being in the state S_t making action A_t ~ policy(S_t) where policy is epsilon-greedy (Q)
#           getting reward R_t moving to the state S_t+1, making action A_t+1 ~ policy(S_t+1)
#       2.  According to (S_t, A_t, R_t, S_t+1, A_t+1) updating Q:
#               Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (R_t + gamma * Q(S_t+1, A_t+a) - Q(S_t, A_t))
#       decrease epsilon



def SARSA(env: gym.wrappers.time_limit.TimeLimit,
          episode_n: int, t_max: int, 
          gamma: float, alpha: float) -> List[int]:

    state_n = env.observation_space.n
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))
    epsilon = 1

    total_rewards = []
    for _ in tqdm(range(episode_n), leave=False, colour="blue"):
        
        trajectory_reward = 0

        state = env.reset()
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        
        for _ in range(t_max):
            

            next_state, reward, done, _ = env.step(action)
            trajectory_reward += reward

            next_action = get_epsilon_greedy_action(q_values[next_state],\
                                                    epsilon, action_n)

            q_values[state][action] += alpha *\
                    (reward + gamma * q_values[next_state][next_action]\
                    - q_values[state][action])

            state = next_state
            action = next_action
            if done:
                break
        total_rewards.append(trajectory_reward)
        epsilon -= 1 / episode_n
    return total_rewards



#   3. Q-Learning Algorithm
#
#   Let Q(s, a) = 0 and epsilon = 1
#   For each episode k do:
#       wilhe episode is not over:
#        1. Being in the state S_t making action A_t ~ policy(s_t),
#           where policy = epsilon-greedy (Q), getting reward R_t,
#           going to the state S_t+1.
#        2. According to the (S_t, A_t, R_t, S_t+1) update Q:
#               Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (R_t + gamma * max_a' (Q(S_t+1, a') - Q(S_t, A_t)))
#       decrease epsilon


def QLearning(env, episode_n, t_max, gamma, alpha):
    state_n = env.observation_space.n
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))
    epsilon = 1

    total_rewards = []

    for _ in tqdm(range(episode_n), leave=False, colour="blue"):
        

        state = env.reset()
        trajectory_reward = 0
        for _ in range(t_max):

            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, done, _ = env.step(action)
            trajectory_reward += reward

            q_values[state][action] += alpha *\
                    (reward + gamma * np.max(q_values[next_state])\
                    - q_values[state][action])
            
            state = next_state
            
            if done:
                break
        epsilon -= 1 / episode_n
        total_rewards.append(trajectory_reward)
    return total_rewards


def smooth(rewards: List[int]):
    # Set the window size
    window_size = 500

    # Calculate the number of windows
    num_windows = len(rewards) // window_size

    # Create an rewards to store the median values
    median_values = np.zeros(num_windows)

    # Iterate through the rewards and calculate the median for each window
    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size
        window = rewards[start_index:end_index]
        median_values[i] = np.median(window)
    return median_values


def main():
    env = gym.make("Taxi-v3")
    STATE_N = 500
    ACTION_N = 6

    mc_start = time.perf_counter()
    mc_rewards = MonteCarlo(env, episode_n=20000)
    mc_end = time.perf_counter()

    sarsa_start = time.perf_counter()
    sarsa_rewards = SARSA(env, episode_n=20000, t_max=500,
                          gamma=0.99, alpha=0.5)
    sarsa_end = time.perf_counter()

    qlearning_start = time.perf_counter()
    qlearning_rewards = QLearning(env, episode_n=20000, t_max=500,
                                  gamma=0.99, alpha=0.5)
    qlearning_end = time.perf_counter()

    agent = crossEntropyAgent(STATE_N, ACTION_N)
    trajecotry_len = 500
    q_param = 0.85
    traj_per_epoch = 500
    n_epochs = 40
    method = "policy_smoothing"
    policy_lambda = 0.85

    ce_start = time.perf_counter()
    ce_rewards = train_ce(env, agent, n_epochs,
                    trajecotry_len, traj_per_epoch, q_param,
                    method=method,
                    laplace_smoothing=0.0,
                    policy_lambda=policy_lambda)

    ce_end = time.perf_counter()
    
    smoothed_mc_rewards = smooth(mc_rewards)
    smoothed_sarsa_rewards = smooth(sarsa_rewards)
    smoothed_qlearning_rewards = smooth(qlearning_rewards)
    smoothed_ce_rewards = smooth(ce_rewards)

    print(f"Cross-entropy time taken: [{ce_end - ce_start}]")
    print(f"Monte-Carlo time taken: [{mc_end - mc_start}]")
    print(f"SARSA time taken: [{sarsa_end - sarsa_start}]")
    print(f"Q-Learning time taken: [{qlearning_end - qlearning_start}]")
    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
        ax1.plot(ce_rewards, label="Cross-Entropy")
        ax1.plot(mc_rewards, label="Monte-Carlo")
        ax1.plot(sarsa_rewards, label="SARSA")
        ax1.plot(qlearning_rewards, label="Q-Learning")
        ax1.legend()
        ax1.set_xlabel("trajectory n")
        ax1.set_ylabel("reward")
        ax1.set_title("Taxi-v3 rewards comparison")


        ax2.plot(smoothed_ce_rewards, label="Cross-Entropy")
        ax2.plot(smoothed_mc_rewards, label="Monte-Carlo")
        ax2.plot(smoothed_sarsa_rewards, label="SARSA")
        ax2.plot(smoothed_qlearning_rewards, label="Q-Learning")
        ax2.legend()
        ax2.set_xlabel("trajectory n")
        ax2.set_ylabel("reward")
        ax2.set_title("Taxi-v3 smoothed rewards comparison (meadian of 500)")
        

        plt.show()


if __name__ == "__main__":
    main()
