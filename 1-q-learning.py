import numpy as np
import gym
import random

env = gym.make("FrozenLake-v0", is_slippery=False)

action_size = env.action_space.n
state_size = env.observation_space.n

print("action_size:", action_size)
print("state_size:", state_size)

qtable = np.zeros((state_size, action_size))
print("qtable:\n", qtable)

# Specify the hyperparameters
total_episodes = 1000
learning_rate = 0.8
max_steps = 99
gamma = 0.95        # Discounting rate

# Exploration parameters
epsilon = 1.0       # Exploration rate
max_epsilon = 1.0   # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

# List of rewards
rewards = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # Exploitation or exploration
        exp_exp_tradeoff = random.uniform(0, 1)

        # Exploitation
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

            # =!!!!!!=
            # Avoid invalid exploitation
            if action == np.argmin(qtable[state, :]):
                action = env.action_space.sample()
            # =!!!!!!=
        else:
            action = env.action_space.sample()

        # Take the action, get the new state and reward
        new_state, reward, done, info = env.step(action)

        qtable[state, action] = qtable[state, action] \
            + learning_rate * (reward + gamma *
                               np.max(qtable[new_state, :])
                               - qtable[state, action])

        total_rewards += reward

        state = new_state

        # If done (if we're dead): finish episode
        if done:
            break

    # Reduce the epsilon, less exploration
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
        np.exp(-decay_rate * episode)

    rewards.append(total_rewards)

print("Score over time:" + str(sum(rewards) / total_episodes))
print(qtable)

# Play
env.reset()

for episode in range(1):
    state = env.reset()
    step = 0
    done = False
    print("********************************************")
    print("EPISODE", episode)
    env.render()

    for step in range(max_steps):
        # Take the action that have maximum expected future reward
        action = np.argmax(qtable[state, :])

        print("state & action:", state, action)

        new_state, reward, done, info = env.step(action)
        env.render()

        if done:
            # env.render()
            print("Number of steps", step)
            break
        state = new_state

env.close()
