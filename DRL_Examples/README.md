# Deep Learning Course Examples
Baylor University, Department of Electrical and Computer Engineering

## Deep Reinforcement Learning (DRL)

Course Examples in TensorFlow and Keras:

1. Use gym environments for developing and comparing reinforcement learning algorithms. 
   http://gym.openai.com/
   
   gym_display.py  (Gym environment display - random run of a chosen gym environment)

2. Q-learning for an agent to play the cartpole game.
   
   cartpole.py  (Seperated training-data collection and network training)
   
3. Deep Q Learning to play the cartpole game.
   
   DQN_run_cartpole.py  (Main program with (1) training while playing the game (2) experience reply (3) fixed Q-targets.)
   RL_DeepQNetwork.py  (Deep Q Network Class)

4. Policy Gradient Method to play the cartpole game.

   PolicyGradient_run_cartpole.py  (Main program with (1) update over episodes)
   RL_PolicyGradient.py  (Policy Gradient Class)
   
5. Actor Critic Method to play the cartpole game.

6. Deep Deterministic Policy Gradient to play the pendulum game (continuous action values)
