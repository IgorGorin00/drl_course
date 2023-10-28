import numpy as np
import gym
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def QLearning(env, episode_n, noisy_episode_n,
              gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)

    state_n = env.observation_space.n
    action_n = env.action_space.n

    q = np.zeros((state_n, action_n))

    for episode in tqdm(range(episode_n), colour='cyan'):
        epsilon = 1 / (episode + 1)

        state = env.reset()

        for _ in range(trajectory_len):

            action = get_epsilon_greedy_action(q[state], epsilon, action_n)
            next_state, reward, done, _ = env.step(action)

            max_next_action = np.argmax(q[next_state])
            q[state][action] = q[state][action] + alpha * \
                (reward + gamma * q[next_state]
                 [max_next_action] - q[state][action])

            state = next_state

            total_rewards[episode] += reward

            if done:
                break

    return total_rewards, q


def test(env, q_values, viz=False):
    done = False
    total_reward = 0
    state = env.reset()
    while not done:

        action = get_epsilon_greedy_action(
            q_values[state], epsilon=0,
            action_n=env.action_space.n
        )
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if viz:
            env.render()
            time.sleep(0.3)
    return total_reward


def main():
    env = gym.make("Taxi-v3")

    total_rewards, q_values = QLearning(
        env, episode_n=500, noisy_episode_n=400,
        trajectory_len=1000, gamma=0.999, alpha=0.5
    )

    last_reward = test(env, q_values, viz=True)
    print(f'{last_reward = }')

    plt.plot(total_rewards)
    plt.show()


if __name__ == '__main__':
    main()
