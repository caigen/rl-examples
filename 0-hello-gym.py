# ipython for display
import numpy as np
import gym

# ================================
name = "FrozenLake-v0"
env = gym.make(name)

action_size = env.action_space.n
state_size = env.observation_space.n

print(name)
print("action_size:", action_size)
print("state_size:", state_size)

env.reset()
env.render()
env.close()

# ================================
name = "CartPole-v0"
env = gym.make(name)
env.reset()
print(name)
print("action space:", env.action_space)
print("action_size:", env.action_space.n)

print("observation space:", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)

for i_episode in range(10):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()

# ================================
from gym import envs
print(envs.registry.all())
