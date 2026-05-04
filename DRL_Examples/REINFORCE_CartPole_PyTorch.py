# -*- coding: utf-8 -*-
"""
=============================================================================
ELC 5365: Deep Learning, Spring 2026
Dr. Liang Dong, Baylor University

REINFORCE (vanilla policy gradient) on CartPole-v1
--------------------------------------------------
We learn a stochastic policy pi_theta(a|s) directly by ascending the
policy-gradient estimator (Williams 1992):

      grad J(theta) = E_tau [ sum_t  gamma^t  G_t  grad log pi_theta(a_t|s_t) ]

where G_t = sum_{k>=0} gamma^k r_{t+k+1} is the rewards-to-go.

To reduce variance we subtract a state-independent baseline b(s):
the running mean of the returns within the batch.  Subtracting any
function of s leaves the gradient unbiased because
E_a [ grad log pi(a|s) ] = 0.

This is the simplest deep policy-gradient demo --- one forward pass
through the network, one Monte-Carlo rollout, one gradient update.

Reference:
  R. J. Williams, "Simple statistical gradient-following algorithms for
  connectionist reinforcement learning," Machine Learning 8 (1992).
  R. Sutton, D. McAllester, S. Singh, Y. Mansour, "Policy Gradient
  Methods for RL with Function Approximation," NIPS 1999.

Install:
  pip install gymnasium gymnasium[classic-control] torch matplotlib
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
import matplotlib.pyplot as plt


# ------------------------------ environment ---------------------------------
env        = gym.make("CartPole-v1", render_mode=None)
render_env = gym.make("CartPole-v1", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

RENDER_EVERY  = 100
N_FINAL_DEMOS = 3


# ---------------------------- hyperparameters -------------------------------
GAMMA        = 0.99
LR           = 1e-3
NUM_EPISODES = 600 if device.type == "cpu" else 1000
USE_BASELINE = True       # subtract running mean of returns


# ---------------------------- policy network --------------------------------
class PolicyNet(nn.Module):
    """Outputs logits over discrete actions."""

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

policy = PolicyNet(n_obs, n_actions).to(device)
optimizer = optim.Adam(policy.parameters(), lr=LR)


def select_action(state):
    """Sample an action from pi_theta(.|s); return action and log-prob."""
    logits = policy(torch.tensor(state, dtype=torch.float32, device=device))
    dist   = Categorical(logits=logits)
    a      = dist.sample()
    return int(a.item()), dist.log_prob(a)


def discounted_returns(rewards, gamma):
    """Compute G_t = sum_{k>=0} gamma^k r_{t+k+1} (rewards-to-go)."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def render_demo(label):
    """Greedy (argmax of logits) episode in the rendered env."""
    state, _ = render_env.reset()
    total = 0
    while True:
        with torch.no_grad():
            logits = policy(torch.tensor(state, dtype=torch.float32, device=device))
            action = int(logits.argmax().item())
        state, r, terminated, truncated, _ = render_env.step(action)
        total += r
        if terminated or truncated:
            break
    print(f"  [demo {label}] return = {total:.0f}")


# ---------------------------- training loop ---------------------------------
episode_returns = []
running_mean = 0.0           # for baseline subtraction

print("Demo before training (untrained policy)...")
render_demo(label="ep   0")

for i_episode in range(NUM_EPISODES):
    state, _ = env.reset()
    log_probs, rewards = [], []

    while True:
        action, lp = select_action(state)
        state, r, terminated, truncated, _ = env.step(action)
        log_probs.append(lp)
        rewards.append(r)
        if terminated or truncated:
            break

    # ------- compute discounted rewards-to-go ----------
    G = discounted_returns(rewards, GAMMA)

    # ------- baseline subtraction (variance reduction) ----------
    if USE_BASELINE:
        # exponential running mean acts as a state-independent baseline
        running_mean = 0.95 * running_mean + 0.05 * G.mean().item()
        adv = G - running_mean
    else:
        adv = G

    # standardize for numerical stability (does not change the gradient direction)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ------- policy-gradient loss = -E[ gamma^t * G_t * log pi(a_t|s_t) ] ----
    T   = len(rewards)
    gam = torch.tensor([GAMMA ** t for t in range(T)],
                       dtype=torch.float32, device=device)
    log_probs = torch.stack(log_probs)
    loss = -(gam * adv * log_probs).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_returns.append(sum(rewards))
    if (i_episode + 1) % 20 == 0:
        recent = episode_returns[-20:]
        print(f"Episode {i_episode+1:4d}/{NUM_EPISODES}  "
              f"return(last 20) avg = {sum(recent)/len(recent):6.1f}")

    if (i_episode + 1) % RENDER_EVERY == 0:
        render_demo(label=f"ep{i_episode+1:4d}")

print(f"Final showcase ({N_FINAL_DEMOS} rendered episodes)...")
for k in range(N_FINAL_DEMOS):
    render_demo(label=f"final {k+1}")


# ---------------------------- plot ------------------------------------------
plt.figure(figsize=(8, 4))
ret = torch.tensor(episode_returns, dtype=torch.float)
plt.plot(ret.numpy(), alpha=0.4, label="episode return")
if len(ret) >= 100:
    means = ret.unfold(0, 100, 1).mean(1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy(), label="100-ep moving average")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("REINFORCE on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.savefig("REINFORCE_CartPole_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()
