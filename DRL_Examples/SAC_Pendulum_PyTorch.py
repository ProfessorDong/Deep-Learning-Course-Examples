# -*- coding: utf-8 -*-
"""
=============================================================================
ELC 5365: Deep Learning, Spring 2026
Dr. Liang Dong, Baylor University

Soft Actor-Critic (SAC) on Pendulum-v1
--------------------------------------
SAC is the state-of-the-art off-policy actor-critic for continuous
control.  It maximizes a *maximum-entropy* objective

      J(pi) = E_tau [ sum_t  r_t + alpha * H( pi(.|s_t) ) ]

so the policy is rewarded for being both rewarding *and* stochastic.
Three pieces make it work in practice:

  (1) Squashed-Gaussian policy.
      The actor outputs (mu, log_sigma).  We sample u ~ N(mu, sigma)
      and squash a = tanh(u) * act_high using the reparameterization
      trick.  The Jacobian of tanh contributes a correction term to
      the log-density:

           log pi(a|s) = log N(u|mu,sigma) - sum_i log(1 - tanh(u_i)^2)

      (the second term is computed in a numerically stable form).

  (2) Twin critics with clipped double Q-learning (as in TD3).

  (3) Automatic entropy tuning.
      Treat alpha as a learnable parameter; minimize

           J(alpha) = E[ -alpha * (log pi(a|s) + H_target) ]

      with H_target = -dim(A).  This adapts the exploration pressure
      to the actor's current entropy.

References:
  T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, "Soft Actor-Critic:
  Off-Policy Maximum Entropy Deep RL with a Stochastic Actor,"
  ICML 2018.
  T. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications,"
  arXiv:1812.05905, 2018.

Install:
  pip install gymnasium gymnasium[classic-control] torch numpy matplotlib
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
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
LR             = 3e-4
BATCH_SIZE     = 256
BUFFER_CAP     = 100_000
MAX_EPISODES   = 150
WARMUP_STEPS   = 1000

LOG_SIGMA_MIN  = -20.0
LOG_SIGMA_MAX  = 2.0


# ---------------------------- networks --------------------------------------
class SquashedGaussianActor(nn.Module):
    """
    Outputs mean and log_sigma.  Sampling uses the reparameterization
    trick;  the squashed action is tanh(u) * act_high, with the
    standard tanh-Jacobian correction applied to the log density.
    """

    def __init__(self, n_obs, n_act, act_high):
        super().__init__()
        self.act_high = torch.tensor(act_high, dtype=torch.float32, device=device)
        self.trunk = nn.Sequential(
            nn.Linear(n_obs, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu_head        = nn.Linear(256, n_act)
        self.log_sigma_head = nn.Linear(256, n_act)

    def forward(self, s):
        h = self.trunk(s)
        mu        = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mu, log_sigma

    def sample(self, s):
        mu, log_sigma = self.forward(s)
        sigma = log_sigma.exp()
        dist = Normal(mu, sigma)
        u    = dist.rsample()                                # reparameterized
        a_t  = torch.tanh(u)                                 # in [-1, 1]
        a    = a_t * self.act_high

        # log pi = log N(u) - sum log(1 - tanh(u)^2)
        # Numerically stable form: log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u))
        log_prob = dist.log_prob(u)
        log_prob -= 2.0 * (np.log(2.0) - u - nn.functional.softplus(-2.0 * u))
        log_prob = log_prob.sum(dim=-1)
        return a, log_prob


class TwinCritic(nn.Module):
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

actor          = SquashedGaussianActor(n_obs, n_act, act_high).to(device)
critic         = TwinCritic(n_obs, n_act).to(device)
critic_target  = TwinCritic(n_obs, n_act).to(device)
critic_target.load_state_dict(critic.state_dict())

opt_a = optim.Adam(actor.parameters(),  lr=LR)
opt_c = optim.Adam(critic.parameters(), lr=LR)

# ----- automatic entropy tuning -----
target_entropy = -float(n_act)                       # heuristic from the SAC paper
log_alpha = torch.zeros(1, requires_grad=True, device=device)
opt_alpha = optim.Adam([log_alpha], lr=LR)

buffer = ReplayBuffer(BUFFER_CAP)


def polyak(net, target, tau):
    for p, p_t in zip(net.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def select_action(state, deterministic=False):
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            mu, _ = actor(s)
            a = torch.tanh(mu) * actor.act_high
        else:
            a, _ = actor.sample(s)
    return a.cpu().numpy()[0].astype(np.float32)


def update():
    if len(buffer) < BATCH_SIZE:
        return
    s, a, r, s_next, done = buffer.sample(BATCH_SIZE)
    alpha = log_alpha.exp().detach()

    # ----- target value: r + gamma * (min Q' - alpha * log pi) -----
    with torch.no_grad():
        a_next, logp_next = actor.sample(s_next)
        q1_t, q2_t = critic_target(s_next, a_next)
        q_next     = torch.min(q1_t, q2_t) - alpha * logp_next
        target     = r + GAMMA * (1.0 - done) * q_next

    # ----- critic update -----
    q1, q2 = critic(s, a)
    critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
    opt_c.zero_grad()
    critic_loss.backward()
    opt_c.step()

    # ----- actor update (reparameterized) -----
    a_pi, logp_pi = actor.sample(s)
    q1_pi, q2_pi  = critic(s, a_pi)
    q_pi          = torch.min(q1_pi, q2_pi)
    actor_loss    = (alpha * logp_pi - q_pi).mean()
    opt_a.zero_grad()
    actor_loss.backward()
    opt_a.step()

    # ----- temperature update -----
    alpha_loss = -(log_alpha * (logp_pi.detach() + target_entropy)).mean()
    opt_alpha.zero_grad()
    alpha_loss.backward()
    opt_alpha.step()

    # ----- target critic Polyak update -----
    polyak(critic, critic_target, TAU)


def render_demo(label):
    """Deterministic (mean) policy episode in the rendered env."""
    s, _  = render_env.reset()
    total = 0.0
    while True:
        a = select_action(s, deterministic=True)
        s, r, terminated, truncated, _ = render_env.step(a)
        total += r
        if terminated or truncated:
            break
    print(f"  [demo {label}] return = {total:7.1f}")


# ---------------------------- training loop ---------------------------------
episode_returns = []
total_steps     = 0

print("Demo before training (untrained policy)...")
render_demo(label="ep   0")

for ep in range(MAX_EPISODES):
    s, _   = env.reset()
    ep_ret = 0.0
    while True:
        if total_steps < WARMUP_STEPS:
            a = env.action_space.sample().astype(np.float32)
        else:
            a = select_action(s)
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
              f"return(last 10) avg = {sum(recent)/len(recent):8.1f}  "
              f"alpha = {log_alpha.exp().item():.3f}")

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
plt.title("SAC on Pendulum-v1")
plt.legend()
plt.tight_layout()
plt.savefig("SAC_Pendulum_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()
