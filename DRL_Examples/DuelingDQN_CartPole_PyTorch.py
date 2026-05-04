# -*- coding: utf-8 -*-
"""
=============================================================================
ELC 5365: Deep Learning, Spring 2026
Dr. Liang Dong, Baylor University

Dueling DQN on CartPole-v1
--------------------------
Network decomposes Q into a state-value V(s) and an action-advantage
A(s, a):

        Q(s, a) = V(s) + ( A(s, a) - mean_{a'} A(s, a') )

The mean-subtraction enforces identifiability (V can absorb a constant
into A and back, otherwise).  Empirically, sharing V across actions
speeds learning when many actions have similar values.

We pair the dueling architecture with the Double-DQN target for clarity.

Reference:
  Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement
  Learning," ICML 2016.

Install:
  pip install gymnasium gymnasium[classic-control] torch matplotlib
=============================================================================
"""

import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import matplotlib.pyplot as plt


# ------------------------------ environment ---------------------------------
env        = gym.make("CartPole-v1", render_mode=None)
render_env = gym.make("CartPole-v1", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

RENDER_EVERY  = 50
N_FINAL_DEMOS = 3


# ---------------------------- replay buffer ---------------------------------
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ---------------------------- Dueling Q-network -----------------------------
class DuelingQNet(nn.Module):
    """
    Shared trunk -> two heads:
        V(s)        : scalar
        A(s, a)     : vector of size n_actions
    Combined as:
        Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))
    """

    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value_head    = nn.Linear(128, 1)            # V(s)
        self.adv_head      = nn.Linear(128, n_actions)    # A(s, .)

    def forward(self, x):
        h = self.trunk(x)
        v = self.value_head(h)                              # [B, 1]
        a = self.adv_head(h)                                # [B, |A|]
        # mean-subtraction trick to make (V, A) identifiable
        return v + (a - a.mean(dim=1, keepdim=True))


# ---------------------------- hyperparameters -------------------------------
BATCH_SIZE = 128
GAMMA      = 0.99
EPS_START  = 0.9
EPS_END    = 0.05
EPS_DECAY  = 1000
TAU        = 0.005
LR         = 1e-4
NUM_EPISODES = 500 if device.type == "cpu" else 800

n_actions = env.action_space.n
state, _  = env.reset()
n_obs     = len(state)

policy_net = DuelingQNet(n_obs, n_actions).to(device)
target_net = DuelingQNet(n_obs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory    = ReplayMemory(10_000)
steps_done = 0


def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device, dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Double-DQN target combined with the dueling Q-net
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states.size(0) > 0:
        with torch.no_grad():
            next_actions = policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
            next_q = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
        next_state_values[non_final_mask] = next_q
    expected = (next_state_values * GAMMA) + reward_batch

    loss = nn.SmoothL1Loss()(state_action_values, expected.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def render_demo(label):
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


# ---------------------------- training loop ---------------------------------
episode_durations = []

print("Demo before training (untrained policy)...")
render_demo(label="ep   0")

for i_episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        next_state = (
            None if terminated
            else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        )
        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()

        tgt = target_net.state_dict()
        pol = policy_net.state_dict()
        for key in pol:
            tgt[key] = pol[key] * TAU + tgt[key] * (1 - TAU)
        target_net.load_state_dict(tgt)

        if done:
            episode_durations.append(t + 1)
            break

    if (i_episode + 1) % 20 == 0:
        recent = episode_durations[-20:]
        print(f"Episode {i_episode+1:4d}/{NUM_EPISODES}  "
              f"len(last 20) avg = {sum(recent)/len(recent):6.1f}")

    if (i_episode + 1) % RENDER_EVERY == 0:
        render_demo(label=f"ep{i_episode+1:4d}")

print(f"Final showcase ({N_FINAL_DEMOS} rendered episodes)...")
for k in range(N_FINAL_DEMOS):
    render_demo(label=f"final {k+1}")


# ---------------------------- plot ------------------------------------------
plt.figure(figsize=(8, 4))
durs = torch.tensor(episode_durations, dtype=torch.float)
plt.plot(durs.numpy(), alpha=0.4, label="episode length")
if len(durs) >= 100:
    means = durs.unfold(0, 100, 1).mean(1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy(), label="100-ep moving average")
plt.xlabel("Episode")
plt.ylabel("Duration (steps)")
plt.title("Dueling DQN on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.savefig("DuelingDQN_CartPole_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()
