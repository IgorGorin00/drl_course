from Frozen_Lake import FrozenLakeEnv
import numpy as np
from tqdm import tqdm


def init_values(env):
    return {state: 0 for state in env.get_all_states()}


# iterate through actions per state:
#     find
#     reward + gamma * sum(transition_prob * values_next_state)
#     choose max
#


def value_function(env, values, gamma):
    state_action_dict = {}
    for state in env.get_all_states():
        state_action_dict[state] = {}
        for action in env.get_possible_actions(state):
            state_action_dict[state][action] = 0
            for next_state in env.get_next_states(state, action):
                reward = env.get_reward(state, action, next_state)
                transition_prob = env.get_transition_prob(
                    state, action, next_state)
                next_value = values[next_state]
                state_action_dict[state][action] += transition_prob * next_value
                state_action_dict[state][action] += reward
            state_action_dict[state][action] *= gamma

        if state_action_dict[state]:
            max_action_value = max(
                state_action_dict[state], key=state_action_dict[state].get)
            values[state] = state_action_dict[state][max_action_value]

    return state_action_dict


def value_iteration(n_iterations, env, gamma):
    values = init_values(env)

    for iteration in tqdm(range(n_iterations)):
        state_action_dict = value_function(env, values, gamma)
        print(f'{iteration = } \t {env.counter = }')
    policy = {}
    for state in env.get_all_states():
        if state_action_dict[state]:
            max_action_value = max(
                state_action_dict[state], key=state_action_dict[state].get)
            policy[state] = max_action_value

    return policy


def evaluate(env, policy, vizualize=False):
    state = env.reset()
    for _ in range(100):
        action = policy[state]
        state, reward, done, _ = env.step(action)
        if vizualize:
            env.render()
        if done:
            break
    return reward


def main():

    env = FrozenLakeEnv()

    states_n = len(env.get_all_states())
    total_actions_n = 4

    values = init_values(env)

    # n_iterations = states_n ** 2 * total_actions_n
    n_iterations = 4
    gamma = 0.9

    policy = value_iteration(n_iterations, env, gamma)

    rewards = [evaluate(env, policy) for _ in range(1000)]
    mean_reward = sum(rewards) / len(rewards)
    print(f'{mean_reward = } for 1000 iterations')


if __name__ == '__main__':
    main()
