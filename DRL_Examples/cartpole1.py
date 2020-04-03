import gym
#import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from statistics import mean, median
from collections import Counter

env = gym.make('CartPole-v1')
#env.reset()
goal_steps = 200
score_requirement = 50
initial_games = 10000

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
            
# some_random_games_first()

def generate_training_data():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        env.reset()
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            # action = random.randrange(0,2)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
                
            prev_observation = observation
            score += reward
            if done:
                break
        
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                #output = data[1]
                training_data.append([data[0], data[1]])
                
        # env.reset()
        scores.append(score)
        
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

training_data = generate_training_data()


#input_dim = len(training_data[0][0])

# Creating a Sequential Model and adding the layers
model = Sequential()
#model.add(Dense(64, input_dim=input_dim, activation='relu'))
model.add(Dense(64, input_shape=env.observation_space.shape, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(env.action_space.n, activation='softmax'))

print(model.summary())


# Training
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

X_train = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
X_train = X_train.astype('float32')
y_train = np.array([i[1] for i in training_data])
y_train = y_train.astype('uint8')

model.fit(x=X_train,y=y_train, epochs=5)

# Evaluation
#[Loss_value, Metrics_value] = model.evaluate(x_test, y_test)
#print("Loss Value:", Loss_value)
#print("Metrics Value:", Metrics_value)

# Inference
scores = []
choices = []

for EachGame in range(10):
    score = 0
    prev_observation = []
    env.reset()
    
    for _ in range(goal_steps):
        env.render()
        #action = random.randrange(0,2)
        #action = env.action_space.sample()
        
        if len(prev_observation) == 0:
            #action = random.randrange(0,2)
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation)).astype('float32')))

        choices.append(action)

        observation, reward, done, info = env.step(action)
        prev_observation = observation
        score += reward
        if done:
            break
        
    scores.append(score)

env.close()
    
print('Average Score', sum(scores)/len(scores))
print('Choice 0: {}, Choice 1: {}'.format(choices.count(0)/len(choices),choices.count(1)/len(choices)))

