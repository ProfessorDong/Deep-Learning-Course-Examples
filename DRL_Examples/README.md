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
   
   **DQN_run_cartpole.py**  (Main program - Tensorflow 1.0)
   
   **RL_DeepQNetwork.py**  (Deep Q Network Class)
   
   Note: (1) Train DNN while playing games and increasing experience; (2) Single-step update of DNN parameters; (3) Experience replay with low correlation; (4) Fixed Q-targets.
   
   Algorithm for Deep Q Learning with Experience Replay:
   
   ![Image of DNN with Experience Replay](https://github.com/ProfessorDong/Deep-Learning-Course-Examples/blob/master/figures/DQN_experiencereplay.png) 
   

4. Policy Gradient Method to play the cartpole game.

   **PolicyGradient_run_cartpole.py**  (Main program - Tensorflow 1.0)
   
   **RL_PolicyGradient.py**  (Policy Gradient Class)
   
   Note: (1) Full-episode update of DNN parameters.
   
   Algorithm for REINFORCE Policy Gradients:
   
   ![Image of REINFORCE](https://github.com/ProfessorDong/Deep-Learning-Course-Examples/blob/master/figures/REINFORCE.jpeg) 
   
5. Actor-Critic Method to play the cartpole game.


6. Deep Deterministic Policy Gradient to play the pendulum game (continuous-valued actions)
   
   **DDPG_pendulum.py**  (Tensorflow 1.0)
   
   DDPG Algorithm:
   
   ![Image of REINFORCE](https://github.com/ProfessorDong/Deep-Learning-Course-Examples/blob/master/figures/DDPG_algorithm.png) 
