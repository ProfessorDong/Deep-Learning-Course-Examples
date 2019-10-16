# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#ENV_NAME = 'CartPole-v1'
ENV_NAME = 'MountainCar-v0'
#ENV_NAME = 'Alien-v0'
#ENV_NAME = 'Pendulum-v0'

import gym
env = gym.make(ENV_NAME)
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()