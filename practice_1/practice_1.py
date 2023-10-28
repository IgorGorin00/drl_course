import gym
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

env = gym.make('maze-sample-5x5-v0')

obs = env.reset()
print(obs)
action = 2
for _ in range(10):
    next_obs, reward, _, _ = env.step(np.random.randint(0, 3))

    env.render()
    time.sleep(0.2)
    print(next_obs)
