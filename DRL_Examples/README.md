# Deep Learning Course Examples
Baylor University, Department of Electrical and Computer Engineering

## Deep Reinforcement Learning (DRL)

Course Examples in TensorFlow and Keras:

1. Use gym environments for developing and comparing reinforcement learning algorithms. 
   http://gym.openai.com/
   
   **gym_display.py**  
   
   Note: Gym environment display - Random operation of the selected gym environment.
   

2. Q-learning for an agent to play the cartpole game.
   
   **cartpole.py**
   
   Note: The training-data collection phase and the network training phase are separate.
   
   
3. Deep Q Learning to play the cartpole game.
   
   **DQN_run_cartpole.py**  (Main program)
   
   **RL_DeepQNetwork.py**  (Deep Q Network Class)
   
   Note: (1) Train DNN and increase experience while playing games; (2) Single-step update of DNN parameters; (3) Experience replay with low correlation; (4) Fixed Q-targets.
   

4. Policy Gradient Method to play the cartpole game.

   **PolicyGradient_run_cartpole.py**  (Main program with (1) update over episodes)
   
   **RL_PolicyGradient.py**  (Policy Gradient Class)
   
   Note: (1) Full-episode update of DNN parameters.
   
   
5. Actor-Critic Method to play the cartpole game.


6. Deep Deterministic Policy Gradient to play the pendulum game (continuous-valued actions)
