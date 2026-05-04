# -*- coding: utf-8 -*-
"""
=============================================================================
ELC 5365: Deep Learning, Spring 2026
Dr. Liang Dong, Baylor University

Twin Delayed DDPG (TD3) on Pendulum-v1
--------------------------------------
Three improvements over DDPG (Fujimoto, van Hoof, Meger, ICML 2018):

  (1) Clipped double Q-learning.
      Maintain TWO critics Q1, Q2.  Use the smaller of the two in the
      target to mitigate maximization bias.

  (2) Delayed policy updates.
      Update the actor and target nets only once every POLICY_DELAY
      critic updates.  Letting the critic catch up before improving
      the policy reduces error propagation.

  (3) Target policy smoothing.
      Add clipped Gaussian noise to the target action when forming the
      target value.  Acts as a regularizer on the critic in a small
      neighborhood of mu_target(s').

References:
  S. Fujimoto, H. van Hoof, D. Meger, "Addressing Function Approximation
  Error in Actor-Critic Methods," ICML 2018.

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
GAMMA          = 0.99
TAU            = 0.005
LR_A           = 1e-4
LR_C           = 1e-3
BATCH_SIZE     = 128
BUFFER_CAP     = 100_000
EXPLORE_NOISE  = 0.1
TARGET_NOISE   = 0.2     # clipped Gaussian on target action
TARGET_CLIP    = 0.5     # bound on the target-action noise
POLICY_DELAY   = 2       # update actor once per POLICY_DELAY critic updates
MAX_EPISODES   = 150
WARMUP_STEPS   = 1000


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
        return self.net(s) * self.act_high


class TwinCritic(nn.Module):
    """Two independent Q-heads sharing input but with separate parameters."""

    def __init__(self, n_obs, n_act):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(n_obs + n_act, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(n_obs + n_act, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1)).squeeze(-1)


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
act_high = env.action_space.high
act_low  = env.action_space.low
act_high_t = torch.tensor(act_high, dtype=torch.float32, device=device)

actor        = Actor(n_obs, n_act, act_high).to(device)
actor_target = Actor(n_obs, n_act, act_high).to(device)
critic        = TwinCritic(n_obs, n_act).to(device)
critic_target = TwinCritic(n_obs, n_act).to(device)
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


update_count = 0


def update():
    global update_count
    if len(buffer) < BATCH_SIZE:
        return
    s, a, r, s_next, done = buffer.sample(BATCH_SIZE)

    # ----- target policy smoothing: noisy clipped target action -----
    with torch.no_grad():
        noise = (torch.randn_like(a) * TARGET_NOISE).clamp(-TARGET_CLIP, TARGET_CLIP)
        a_next = (actor_target(s_next) + noise * act_high_t).clamp(
            torch.tensor(act_low,  dtype=torch.float32, device=device),
            torch.tensor(act_high, dtype=torch.float32, device=device),
        )
        # ----- clipped double Q -----
        q1_t, q2_t = critic_target(s_next, a_next)
        q_next     = torch.min(q1_t, q2_t)
        target     = r + GAMMA * (1.0 - done) * q_next

    # ----- critic update -----
    q1, q2 = critic(s, a)
    critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
    opt_c.zero_grad()
    critic_loss.backward()
    opt_c.step()

    update_count += 1

    # ----- delayed actor & target updates -----
    if update_count % POLICY_DELAY == 0:
        actor_loss = -critic.q1_only(s, actor(s)).mean()
        opt_a.zero_grad()
        actor_loss.backward()
        opt_a.step()

        polyak(actor,  actor_target,  TAU)
        polyak(critic, critic_target, TAU)


def render_demo(label):
    """Deterministic-actor episode in the rendered env."""
    s, _  = render_env.reset()
    total = 0.0
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
    s, _   = env.reset()
    ep_ret = 0.0
    while True:
        if total_steps < WARMUP_STEPS:
            a = env.action_space.sample().astype(np.float32)
        else:
            a = select_action(s, EXPLORE_NOISE)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

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
              f"return(last 10) avg = {sum(recent)/len(recent):8.1f}")

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
plt.title("TD3 on Pendulum-v1")
plt.legend()
plt.tight_layout()
plt.savefig("TD3_Pendulum_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()
