# -*- coding: utf-8 -*-
"""
=============================================================================
ELC 5365: Deep Learning, Spring 2026
Dr. Liang Dong, Baylor University

Deep Deterministic Policy Gradient (DDPG) on Pendulum-v1
--------------------------------------------------------
DDPG is the off-policy actor-critic for *continuous* action spaces
where the DQN max_a is intractable.  It uses:

  - A deterministic actor mu_theta : S -> A  (output squashed by tanh
    and rescaled to the action range).
  - A Q-critic Q_phi(s, a)  trained by TD on a replay buffer.
  - Two target networks  mu_{theta^-}  and  Q_{phi^-}  updated by
    Polyak averaging:    theta^- <- tau * theta + (1-tau) * theta^-
  - Exploration via Gaussian noise on the actor output.

The Deterministic Policy Gradient theorem (Silver et al., 2014) gives
the actor update:

      grad_theta J = E_s[ ( grad_theta mu_theta(s) )^T  grad_a Q(s, a) | a = mu(s) ]

In code: actor_loss = - E[ Q(s, mu_theta(s)) ]  (autograd does the chain).

References:
  D. Silver et al., "Deterministic Policy Gradient Algorithms," ICML 2014.
  T. Lillicrap et al., "Continuous control with deep reinforcement
  learning," ICLR 2016.

Install:
  pip install gymnasium gymnasium[classic-control] torch numpy matplotlib
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

import gymnasium as gym
import matplotlib.pyplot as plt


# ------------------------------ environment ---------------------------------
env        = gym.make("Pendulum-v1", render_mode=None)
render_env = gym.make("Pendulum-v1", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

RENDER_EVERY  = 30
N_FINAL_DEMOS = 3


# ---------------------------- hyperparameters -------------------------------
GAMMA        = 0.99
TAU          = 0.005
LR_A         = 1e-4
LR_C         = 1e-3
BATCH_SIZE   = 128
BUFFER_CAP   = 100_000
EXPLORE_NOISE = 0.1
MAX_EPISODES  = 150
WARMUP_STEPS  = 1000   # random actions until buffer has data


# ---------------------------- networks --------------------------------------
class Actor(nn.Module):
    def __init__(self, n_obs, n_act, act_high):
        super().__init__()
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=device)
        self.net = nn.Sequential(
            nn.Linear(n_obs, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_act), nn.Tanh(),
        )

    def forward(self, s):
        # tanh -> [-1, 1], then rescale to action range
        return self.net(s) * self.act_high


class Critic(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


# ---------------------------- replay buffer ---------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        return (
            torch.tensor(s,      dtype=torch.float32, device=device),
            torch.tensor(a,      dtype=torch.float32, device=device),
            torch.tensor(r,      dtype=torch.float32, device=device),
            torch.tensor(s_next, dtype=torch.float32, device=device),
            torch.tensor(done,   dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------- setup -----------------------------------------
n_obs    = env.observation_space.shape[0]
n_act    = env.action_space.shape[0]
act_high = env.action_space.high                  # numpy [n_act]
act_low  = env.action_space.low

actor        = Actor(n_obs, n_act, act_high).to(device)
actor_target = Actor(n_obs, n_act, act_high).to(device)
critic        = Critic(n_obs, n_act).to(device)
critic_target = Critic(n_obs, n_act).to(device)
actor_target .load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

opt_a = optim.Adam(actor.parameters(),  lr=LR_A)
opt_c = optim.Adam(critic.parameters(), lr=LR_C)

buffer = ReplayBuffer(BUFFER_CAP)


def polyak(net, target, tau):
    for p, p_t in zip(net.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def select_action(state, noise_scale):
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        a = actor(s).cpu().numpy()[0]
    a = a + np.random.normal(0.0, noise_scale * act_high, size=n_act)
    return np.clip(a, act_low, act_high).astype(np.float32)


def update():
    if len(buffer) < BATCH_SIZE:
        return
    s, a, r, s_next, done = buffer.sample(BATCH_SIZE)

    # ----- critic update -----
    with torch.no_grad():
        a_next = actor_target(s_next)
        q_next = critic_target(s_next, a_next)
        target = r + GAMMA * (1.0 - done) * q_next

    q = critic(s, a)
    critic_loss = nn.functional.mse_loss(q, target)
    opt_c.zero_grad()
    critic_loss.backward()
    opt_c.step()

    # ----- actor update (DPG) -----
    actor_loss = -critic(s, actor(s)).mean()
    opt_a.zero_grad()
    actor_loss.backward()
    opt_a.step()

    # ----- Polyak target update -----
    polyak(actor,  actor_target,  TAU)
    polyak(critic, critic_target, TAU)


def render_demo(label):
    """One deterministic-actor episode in the rendered env (no exploration noise)."""
    s, _   = render_env.reset()
    total  = 0.0
    while True:
        s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a = actor(s_t).cpu().numpy()[0]
        a = np.clip(a, act_low, act_high).astype(np.float32)
        s, r, terminated, truncated, _ = render_env.step(a)
        total += r
        if terminated or truncated:
            break
    print(f"  [demo {label}] return = {total:7.1f}")


# ---------------------------- training loop ---------------------------------
episode_returns = []
total_steps     = 0

print("Demo before training (untrained actor)...")
render_demo(label="ep   0")

for ep in range(MAX_EPISODES):
    s, _    = env.reset()
    ep_ret  = 0.0
    while True:
        if total_steps < WARMUP_STEPS:
            a = env.action_space.sample().astype(np.float32)
        else:
            a = select_action(s, EXPLORE_NOISE)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        # Pendulum returns no truly-terminated states; keep done=truncated for buffer mask
        buffer.push(s, a, r, s_next, float(terminated))

        s = s_next
        ep_ret += r
        total_steps += 1
        update()

        if done:
            break

    episode_returns.append(ep_ret)
    if (ep + 1) % 10 == 0:
        recent = episode_returns[-10:]
        print(f"Episode {ep+1:3d}/{MAX_EPISODES}  "
              f"return(last 10) avg = {sum(recent)/len(recent):8.1f}  "
              f"buffer = {len(buffer)}")

    if (ep + 1) % RENDER_EVERY == 0:
        render_demo(label=f"ep{ep+1:3d}")

print(f"Final showcase ({N_FINAL_DEMOS} rendered episodes)...")
for k in range(N_FINAL_DEMOS):
    render_demo(label=f"final {k+1}")


# ---------------------------- plot ------------------------------------------
plt.figure(figsize=(8, 4))
ret = torch.tensor(episode_returns, dtype=torch.float)
plt.plot(ret.numpy(), alpha=0.5, label="episode return")
if len(ret) >= 10:
    means = ret.unfold(0, 10, 1).mean(1)
    means = torch.cat((torch.zeros(9), means))
    plt.plot(means.numpy(), label="10-ep moving average")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DDPG on Pendulum-v1")
plt.legend()
plt.tight_layout()
plt.savefig("DDPG_Pendulum_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()
