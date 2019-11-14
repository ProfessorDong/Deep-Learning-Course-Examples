import os
import gym
from dqn_keras import Agent
import numpy as np
#from gym import wrappers
#import matplotlib.pyplot as plt

def preprocess(observation):
    observation = observation / 255
    return np.mean(observation[30:,:], axis=2).reshape(180,160,1)

def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx,:] = frame
    else:
        stacked_frames[0:buffer_size-1,:] = stacked_frames[1:,:]
        stacked_frames[buffer_size-1, :] = frame

    stacked_frames = stacked_frames.reshape(1, *frame.shape[0:2], buffer_size)

    return stacked_frames


if __name__ == '__main__':

    env = gym.make('BreakoutDeterministic-v4')
    load_checkpoint = False
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.000025, input_dims=(180,160,4), 
                  n_actions=3, mem_size=25000, batch_size=64)
    if load_checkpoint:
        agent.load_models()
    scores = []
    eps_history = []
    numGames = 48 # Change the number of games to a large number in real training, e.g., 50000.
    stack_size = 4
    score = 0

    n_steps = 0
    for i in range(numGames):
        done = False
        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)
        score = 0
        while not done:
            action = agent.choose_action(observation)
            action += 1
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            observation_ = stack_frames(stacked_frames, preprocess(observation_), stack_size)
            score += reward
            action -= 1
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_
            if n_steps % 4 == 0:
                agent.learn()
        if i % 12 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-12):(i+1)])
            print('episode: ', i,'score: ', score,
                 ' average score %.3f' % avg_score,
                'epsilon %.3f' % agent.epsilon)
            agent.save_models()
        else:
            print('episode: ', i,'score: ', score)
        eps_history.append(agent.epsilon)
        scores.append(score)
    
    # Test the trained model
    done = False
    observation = env.reset()
    observation = preprocess(observation)
    stacked_frames = None
    observation = stack_frames(stacked_frames, observation, stack_size)
    score = 0
    while not done:
        env.render()  # Render
        state = np.array(observation, copy=False, dtype=np.float32)
        actions = agent.q_eval.predict(state)
        action = np.argmax(actions)
        action += 1
        observation_, reward, done, info = env.step(action)
        observation_ = stack_frames(stacked_frames, preprocess(observation_), stack_size)
        score += reward
        action -= 1
        observation = observation_
    
    env.close()
    print('Model is trained with %i games' % numGames)
    print('Test score: ', score)