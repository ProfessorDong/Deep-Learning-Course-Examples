# Import the gym module
import gym

#ENV_NAME = 'CartPole-v1'
#ENV_NAME = 'MountainCar-v0'
#ENV_NAME = 'Alien-v0'
#ENV_NAME = 'Pendulum-v0'
ENV_NAME = 'Breakout-v0'
#ENV_NAME = 'BreakoutDeterministic-v4'

env = gym.make(ENV_NAME)
frame = env.reset()  # Reset it, returns the starting frame

is_done = False
while not is_done:
    env.render()  # Render
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())

'''
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
'''

env.close()