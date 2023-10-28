from Frozen_Lake import FrozenLakeEnv
import numpy as np
from tqdm import tqdm


def init_policy(env):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy


def get_q_values(env, values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                reward = env.get_reward(state, action, next_state)
                transition_prob = env.get_transition_prob(
                    state, action, next_state)
                next_value = values[next_state]
                q_values[state][action] += reward + \
                    gamma * transition_prob * next_value

    return q_values


def init_values(env):
    return {state: 0 for state in env.get_all_states()}


def policy_evaluation_step(env, policy, values, gamma):
    q_values = get_q_values(env, values, gamma)
    new_values = {}
    for state in env.get_all_states():
        new_values[state] = 0
        for action in env.get_possible_actions(state):
            new_values[state] += policy[state][action] * \
                q_values[state][action]
    return new_values


def policy_evaluation(env, policy, gamma, evaluation_step_n):
    values = init_values(env)
    for _ in range(evaluation_step_n):
        values = policy_evaluation_step(env, policy, values, gamma)
    q_values = get_q_values(env, values, gamma)
    return q_values


def policy_improvement(env, q_values):
    new_policy = {}
    for state in env.get_all_states():
        new_policy[state] = {}
        max_action = None
        max_q_value = float('-inf')
        for action in env.get_possible_actions(state):
            if q_values[state][action] > max_q_value:
                max_q_value = q_values[state][action]
                max_action = action
        for action in env.get_possible_actions(state):
            new_policy[state][action] = 1 if action == max_action else 0
    return new_policy


def create_policies(env, epochs=20, evaluation_step_n=60):

    gammas = np.linspace(0, 1, 20)
    policies = []

    policy = init_policy(env)
    for gamma in gammas:
        for epoch in range(epochs):
            q_values = policy_evaluation(env, policy, gamma, evaluation_step_n)
            policy = policy_improvement(env, q_values)
        policies += [policy]

    return policies


def test(env, policies):
    mean_rewards = []
    for policy in policies:
        mean_reward = 0
        for i in tqdm(range(10000), colour='blue', leave=True):
            total_reward = 0
            state = env.reset()
            for _ in range(100):
                action = np.random.choice(env.get_possible_actions(
                    state), p=list(policy[state].values()))
                state, reward, done, _ = env.step(action)
                # env.render()
                # time.sleep(0.5)
                total_reward += reward

                if done:
                    break

            mean_reward += total_reward
        mean_reward /= 10000
        mean_rewards += [mean_reward]
    return mean_rewards


def compare(mean_rewards):
    import matplotlib.pyplot as plt

    gammas = np.linspace(0, 1, 20)

    with plt.style.context('ggplot'):
        plt.figure(figsize=(15, 8))
        plt.plot(gammas, mean_rewards)
        plt.scatter(gammas, mean_rewards)
        plt.xlabel('gamma')
        plt.ylabel('mean_reward')
        plt.title('Gammas X Mean rewards')
        plt.show()


def main():
    env = FrozenLakeEnv()
    policies = create_policies(env)
    mean_rewards = test(env, policies)
    compare(mean_rewards)


if __name__ == '__main__':
    main()
