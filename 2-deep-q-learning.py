import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Setup env
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("state_size and action_size:", state_size, action_size)

# Hyperparameters
gamma = 0.95        # Discounting rate
epsilon = 1.0       # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001

# DQN model
model = Sequential()
model.add(Dense(24, input_dim=state_size, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(action_size, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=learning_rate))

# Run
done = False
batch_size = 32
memory = deque(maxlen=2000)

episodes = 50
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for step in range(500):
        env.render()

        # Exploitation or exploration
        action = 0
        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)
        else:
            action_values = model.predict(state)
            action = np.argmax(action_values[0])

        # Step
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        if done:
            reward = -10

        # Remember for sample training
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, episodes, step, epsilon))
            break

        # Learn from the memory, avoid oscillations
        if len(memory) > batch_size:
            mini_batch = random.sample(memory, batch_size)
            for m_state, m_action, m_reward, m_next_state, m_done \
                    in mini_batch:
                target_reward = m_reward
                if not m_done:
                    target_reward = (m_reward + gamma *
                                     np.amax(model.predict(m_next_state)[0]))

                predict_reward = model.predict(m_state)
                predict_reward[0][m_action] = target_reward
                model.fit(m_state, predict_reward, epochs=1, verbose=0)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
