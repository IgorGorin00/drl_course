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


def policy_evaluation_step(policy, values, gamma, env):
    q_values = get_q_values(values, gamma, env)
    new_values = {}
    for state in env.get_all_states():
        new_values[state] = 0
        for action in env.get_possible_actions(state):
            new_values[state] += policy[state][action] * \
                q_values[state][action]
    return new_values


def init_values(env):
    return {state: 0 for state in env.get_all_states()}


def get_q_values(values, gamma, env):
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


def policy_evaluation(policy, values, gamma, evaluation_step_n, env):

    for _ in range(evaluation_step_n):
        values = policy_evaluation_step(policy, values, gamma, env)
    q_values = get_q_values(values, gamma, env)
    return q_values


def policy_evaluation_with_resetting(policy, gamma, evaluation_step_n, env):
    values = init_values(env)
    for _ in range(evaluation_step_n):
        values = policy_evaluation_step(policy, values, gamma, env)
    q_values = get_q_values(values, gamma, env)
    return q_values


def policy_improvement(q_values, env):
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


def train(epochs, evaluation_step_n, gamma, env, reset_values=False):
    policy = init_policy(env)
    values = init_values(env)
    for epoch in range(epochs):
        if reset_values:
            q_values = policy_evaluation_with_resetting(
                policy, gamma, evaluation_step_n, env)
        else:
            q_values = policy_evaluation(
                policy, values, gamma, evaluation_step_n, env)
        policy = policy_improvement(q_values, env)

    return policy


def test(policy, env, vizualize=False):
    total_reward = 0
    state = env.reset()
    for _ in range(100):
        action = np.random.choice(env.get_possible_actions(
            state), p=list(policy[state].values()))
        state, reward, done, _ = env.step(action)
        if vizualize:
            env.render()
        total_reward += reward

        if done:
            break

    return total_reward


def main():

    env = FrozenLakeEnv()

    epochs = 20
    evaluation_step_n = 100
    gamma = 0.9

    policy_with_resetting = train(
        epochs, evaluation_step_n, gamma, env, reset_values=False)
    policy_without_resetting = train(
        epochs, evaluation_step_n, gamma, env, reset_values=False)

    rewards_with_resetting = [test(policy_with_resetting, env)
                              for _ in tqdm(range(10000))]
    rewards_without_resetting = [
        test(policy_without_resetting, env) for _ in tqdm(range(10000))]

    print(
        f'Mean reward for 10_000 tries with resetting [{np.mean(rewards_with_resetting)}]')
    print(
        f'Mean reward for 10_000 tries with resetting [{np.mean(rewards_without_resetting)}]')


if __name__ == '__main__':
    main()
