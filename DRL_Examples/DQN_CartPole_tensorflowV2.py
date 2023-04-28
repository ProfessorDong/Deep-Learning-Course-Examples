# -*- coding: utf-8 -*-

"""Created on Thu Apr 27 22:49:20 2023

@author: Liang_Dong

"""

# pip install -e C:\Users\Liang_Dong\2023DeepLearning\gym-master


import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from collections import deque
import random

ENV_NAME = "CartPole-v1"
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = tf.keras.Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, targets_f = [], []
        for state, action, reward, state_next, terminal in batch:
            target = reward
            if not terminal:
                target = reward + GAMMA * np.amax(self.model.predict(state_next)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets_f.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def cartpole():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state, info = env.reset()
        # print(state)
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal, _ = env.step(action)[:4]
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print(f"Run: {run}, exploration: {dqn_solver.exploration_rate}, score: {step}")
                break
            dqn_solver.experience_replay()

if __name__ == "__main__":
    cartpole()
