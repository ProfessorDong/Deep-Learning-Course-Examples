# -*- coding: utf-8 -*-
"""
=============================================================================
ELC 5365: Deep Learning, Spring 2026
Dr. Liang Dong, Baylor University

Prioritized Experience Replay (PER) DQN on CartPole-v1
------------------------------------------------------
Vanilla replay samples uniformly from the buffer, wasting compute on
transitions the network has already mastered.  PER samples transition i
with probability proportional to its TD error:

        p_i  proportional to  ( |delta_i| + epsilon )^alpha

Because this distorts the gradient, every sampled term is reweighted
by an importance-sampling correction:

        w_i = ( N * p_i )^(-beta)         beta annealed from 0.4 to 1

For pedagogy we use a simple proportional-priority array (no sum-tree).
A real implementation would replace the O(N) sampling with O(log N)
sum-tree access, but the algorithm is identical.

Reference:
  T. Schaul, J. Quan, I. Antonoglou, D. Silver, "Prioritized Experience
  Replay," ICLR 2016.

Install:
  pip install gymnasium gymnasium[classic-control] torch numpy matplotlib
=============================================================================
"""

import math
import random
from itertools import count

import numpy as np
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


# ---------------------------- hyperparameters -------------------------------
BATCH_SIZE   = 128
GAMMA        = 0.99
EPS_START    = 0.9
EPS_END      = 0.05
EPS_DECAY    = 1000
TAU          = 0.005
LR           = 1e-4
NUM_EPISODES = 500 if device.type == "cpu" else 800

# PER-specific hyperparameters
ALPHA        = 0.6           # priority exponent (0 = uniform; 1 = full priority)
BETA_START   = 0.4           # initial IS-correction strength
BETA_END     = 1.0           # final IS-correction strength
PER_EPS      = 1e-6          # small constant so zero-error transitions can still be sampled
BUFFER_CAP   = 10_000


# ---------------------------- prioritized replay buffer ---------------------
class PrioritizedReplay:
    """
    Numpy-backed proportional-priority buffer.
    Stores (state, action, reward, next_state, done) tuples and a per-slot
    priority.  Sampling is O(N); good enough for the CartPole-scale demo.
    """

    def __init__(self, capacity, n_obs, alpha=ALPHA):
        self.capacity = capacity
        self.alpha    = alpha
        self.pos      = 0
        self.size     = 0
        # storage
        self.states      = np.zeros((capacity, n_obs), dtype=np.float32)
        self.actions     = np.zeros((capacity,),         dtype=np.int64)
        self.rewards     = np.zeros((capacity,),         dtype=np.float32)
        self.next_states = np.zeros((capacity, n_obs), dtype=np.float32)
        self.dones       = np.zeros((capacity,),         dtype=np.float32)
        # priorities (raw |delta| values, NOT raised to alpha)
        self.priorities  = np.zeros((capacity,),         dtype=np.float64)

    def push(self, s, a, r, s_next, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.states     [self.pos] = s
        self.actions    [self.pos] = a
        self.rewards    [self.pos] = r
        self.next_states[self.pos] = s_next
        self.dones      [self.pos] = float(done)
        self.priorities [self.pos] = max_prio    # new transitions get max priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        prios = self.priorities[:self.size] ** self.alpha
        probs = prios / prios.sum()
        idx   = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()                # normalize for stable updates
        return (
            self.states     [idx],
            self.actions    [idx],
            self.rewards    [idx],
            self.next_states[idx],
            self.dones      [idx],
            weights.astype(np.float32),
            idx,
        )

    def update_priorities(self, idx, new_prios):
        self.priorities[idx] = np.abs(new_prios) + PER_EPS

    def __len__(self):
        return self.size


# ---------------------------- Q-network -------------------------------------
class QNet(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------- setup -----------------------------------------
n_actions = env.action_space.n
state, _  = env.reset()
n_obs     = len(state)

policy_net = QNet(n_obs, n_actions).to(device)
target_net = QNet(n_obs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory    = PrioritizedReplay(BUFFER_CAP, n_obs)
steps_done = 0


def beta_by_step(step, total_steps):
    """Linear anneal of IS-correction beta from BETA_START -> BETA_END."""
    frac = min(1.0, step / total_steps)
    return BETA_START + frac * (BETA_END - BETA_START)


def select_action(state_t):
    global steps_done
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps:
        with torch.no_grad():
            return int(policy_net(state_t).argmax(dim=1).item())
    return env.action_space.sample()


def optimize_model(beta):
    if len(memory) < BATCH_SIZE:
        return
    s, a, r, s_next, d, w, idx = memory.sample(BATCH_SIZE, beta)
    s        = torch.tensor(s,        device=device)
    a        = torch.tensor(a,        device=device).unsqueeze(1)
    r        = torch.tensor(r,        device=device)
    s_next   = torch.tensor(s_next,   device=device)
    d        = torch.tensor(d,        device=device)
    w        = torch.tensor(w,        device=device)

    q_sa = policy_net(s).gather(1, a).squeeze(1)
    with torch.no_grad():
        # Double-DQN target (online selects, target evaluates)
        next_a = policy_net(s_next).argmax(dim=1, keepdim=True)
        next_q = target_net(s_next).gather(1, next_a).squeeze(1)
        target = r + GAMMA * (1.0 - d) * next_q

    td_err = q_sa - target
    # IS-weighted Huber loss
    loss   = (w * nn.functional.smooth_l1_loss(q_sa, target, reduction="none")).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    # update priorities with the freshly-computed |TD error|
    memory.update_priorities(idx, td_err.detach().cpu().numpy())


def render_demo(label):
    state, _ = render_env.reset()
    state_t  = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total = 0
    while True:
        with torch.no_grad():
            action = int(policy_net(state_t).argmax(dim=1).item())
        obs, r, terminated, truncated, _ = render_env.step(action)
        total += r
        if terminated or truncated:
            break
        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"  [demo {label}] duration = {int(total)}")


# ---------------------------- training loop ---------------------------------
episode_durations = []
total_steps_target = NUM_EPISODES * 200  # rough; used only for beta anneal

print("Demo before training (untrained policy)...")
render_demo(label="ep   0")

for i_episode in range(NUM_EPISODES):
    s, _ = env.reset()
    s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        a = select_action(s_t)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        memory.push(s, a, r, s_next, terminated)

        s = s_next
        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

        beta = beta_by_step(steps_done, total_steps_target)
        optimize_model(beta)

        # Polyak target update
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
              f"len(last 20) avg = {sum(recent)/len(recent):6.1f}  "
              f"beta = {beta_by_step(steps_done, total_steps_target):.2f}")

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
plt.title("PER + Double-DQN on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.savefig("PER_DQN_CartPole_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()
