# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:52:39 2023

@author: Liang_Dong

https://www.youtube.com/watch?v=kopoLzvh5jY

https://gymnasium.farama.org/
https://github.com/Farama-Foundation/Gymnasium

pip install gymnasium
pip install gymnasium[classic-control]

"""


import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Two CartPole-v1 envs:
#   - `env`        runs silently for fast training
#   - `render_env` opens a pygame window for periodic class demos
# Rendering every training step would slow learning ~30x, so we only
# render selected demo episodes and keep training itself headless.
env        = gym.make("CartPole-v1", render_mode=None)
render_env = gym.make("CartPole-v1", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# How often to pause training and run a rendered greedy demo.
RENDER_EVERY     = 50    # episodes between checkpoint demos
N_FINAL_DEMOS    = 3     # rendered showcase episodes after training finishes

# Experience Replay Buffer
Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self,*args):
        self.memory.append(Transition(*args))
        # Save a transition

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    # Multilayer perceptron with three layers
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations,128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 128 # number of transitions samples from the experience replay buffer
GAMMA = 0.99  # discount factor
EPS_START = 0.9 # Starting value of epsilon
EPS_END = 0.05
EPS_DECAY = 1000 # rate of exponential decay of epsilon, higher means slower decay
TAU = 0.005 # update rate of the target network
LR =  1e-4 # learning rate of AdamW optimizer

n_actions = env.action_space.n  # Get the number of actions from gym action space

state, info = env.reset()
n_observations = len(state) # Get the number of features in the state observations

# Target network is initialized with the same weights as the policy network
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(10000)

# Keep track of the number of steps taken by the agent
steps_done = 0 


# Input current state and return an action
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row
            # second column on max result is index of where max element was found, so we pick action with the larger expected reward
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# It is used to keep track of the duration of each episode
episode_durations = []


def render_demo(label):
    """
    Run ONE greedy episode in the rendered env and pop up the live
    CartPole window.  Uses argmax(Q) -- no exploration noise -- so the
    students see exactly what the current policy looks like.
    """
    state, _ = render_env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total = 0
    while True:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
        obs, r, terminated, truncated, _ = render_env.step(action.item())
        total += r
        if terminated or truncated:
            break
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"  [demo {label}] duration = {int(total)}")


def optimize_model():
    # Check if we have enough samples for a mini bacth
    if len(memory) < BATCH_SIZE:
        return

    # Extract a mini-batch of transition (state,action,reward,next_state) from the experience reply memory
    transition = memory.sample(BATCH_SIZE)

    # Convert batch-array of Transitions to Transition of batch-arrays
    batch = Transition(*zip(*transition))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Expect Q-value for each transition using the target network
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # smooth approximation of the mean square error loss, less sensitive
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    # maximum value = 100 to prevent exploding gradient problem
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


  
if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 500

# ---- baseline demo: untrained policy (essentially random) ----
print("Demo before training (untrained policy)...")
render_demo(label="ep   0")

# Iterate for episodes
for i_episode in range(num_episodes):
    # Intialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # print("State:", state)
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # frame = env.render()
        # plt.imshow(frame)
        # plt.show()

        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Optimize
        optimize_model()

        # Soft (Polyak) update of the target network: θ⁻ ← τ·θ + (1-τ)·θ⁻
        # NOTE: previous draft had a bug here -- it read policy_net_state_dict
        # from target_net.state_dict() (typo), so the target net never updated.
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()  # <-- correct source

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break

    if (i_episode + 1) % 20 == 0:
        recent = episode_durations[-20:]
        print(f"Episode {i_episode+1:4d}/{num_episodes}  "
              f"duration(last 20) avg = {sum(recent)/len(recent):6.1f}")

    # ---- periodic class demo with the current greedy policy ----
    if (i_episode + 1) % RENDER_EVERY == 0:
        render_demo(label=f"ep{i_episode+1:4d}")

print('Complete')

# ---- final showcase: a few rendered episodes with the trained policy ----
print(f"Final showcase ({N_FINAL_DEMOS} rendered episodes)...")
for k in range(N_FINAL_DEMOS):
    render_demo(label=f"final {k+1}")

# ---------------------------- plot ------------------------------------------
plt.figure(figsize=(8, 4))
durations_t = torch.tensor(episode_durations, dtype=torch.float)
plt.plot(durations_t.numpy(), alpha=0.4, label="episode duration")
if len(durations_t) >= 100:
    means = durations_t.unfold(0, 100, 1).mean(1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy(), label="100-ep moving average")
plt.xlabel("Episode")
plt.ylabel("Duration")
plt.title("DQN on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.savefig("DQN_CartPole_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()