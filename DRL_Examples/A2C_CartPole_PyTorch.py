# -*- coding: utf-8 -*-
"""
=============================================================================
ELC 5365: Deep Learning, Spring 2026
Dr. Liang Dong, Baylor University

Advantage Actor-Critic (A2C) on CartPole-v1
-------------------------------------------
Two networks share the same trunk:
    Actor  pi_theta(a|s)        : Categorical over actions
    Critic V_phi(s)             : scalar baseline (state value)

Per transition we compute the one-step TD residual

        delta_t = r_{t+1} + gamma * V_phi(s_{t+1})  -  V_phi(s_t)

which serves as an unbiased low-variance estimate of the advantage
A^pi(s_t, a_t).

  Actor  loss : - log pi_theta(a_t|s_t) * delta_t.detach()
  Critic loss : 0.5 * delta_t^2

In contrast to REINFORCE, A2C bootstraps from V_phi instead of
waiting until the episode ends -- much lower variance, faster learning,
at the cost of some bias from V_phi's approximation error.

References:
  V. Mnih et al., "Asynchronous Methods for Deep RL" (A3C), ICML 2016.

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
LR           = 3e-4
ENTROPY_COEF = 0.01           # encourages exploration
NUM_EPISODES = 800 if device.type == "cpu" else 1500


# ---------------------------- shared-trunk model ----------------------------
class ActorCritic(nn.Module):
    """Shared MLP trunk -> two heads (logits, scalar value)."""

    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.actor_head  = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, s):
        h = self.trunk(s)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


# ---------------------------- setup -----------------------------------------
n_actions = env.action_space.n
state, _  = env.reset()
n_obs     = len(state)

net       = ActorCritic(n_obs, n_actions).to(device)
optimizer = optim.Adam(net.parameters(), lr=LR)


def select_action(state):
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logits, value = net(s)
    dist = Categorical(logits=logits)
    a    = dist.sample()
    return int(a.item()), dist.log_prob(a), dist.entropy(), value.squeeze(0)


def render_demo(label):
    """Greedy (argmax of actor logits) episode in the rendered env."""
    state, _ = render_env.reset()
    total = 0
    while True:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(s)
            action = int(logits.argmax(dim=1).item())
        state, r, terminated, truncated, _ = render_env.step(action)
        total += r
        if terminated or truncated:
            break
    print(f"  [demo {label}] return = {total:.0f}")


# ---------------------------- training loop ---------------------------------
episode_returns = []

print("Demo before training (untrained policy)...")
render_demo(label="ep   0")

for i_episode in range(NUM_EPISODES):
    state, _ = env.reset()
    log_probs, values, rewards, entropies, dones = [], [], [], [], []

    while True:
        action, lp, ent, v = select_action(state)
        next_state, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log_probs.append(lp)
        values   .append(v)
        rewards  .append(r)
        entropies.append(ent)
        dones    .append(float(terminated))      # use terminated (not truncated) for bootstrap masking

        state = next_state
        if done:
            break

    # ------- one-step TD targets and advantages over the trajectory ---------
    T = len(rewards)
    with torch.no_grad():
        s_T = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        _, v_last = net(s_T)
        v_last = v_last.squeeze(0)
        # if the episode terminated, bootstrap target at the final step is 0
        if dones[-1] == 1.0:
            v_last = torch.zeros_like(v_last)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_t  = torch.stack(values)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    # bootstrap value V(s_{t+1}); for t = T-1 use v_last
    next_values = torch.cat([values_t[1:], v_last.unsqueeze(0)])
    # zero out the bootstrap on terminated transitions
    not_done    = 1.0 - torch.tensor(dones, dtype=torch.float32, device=device)
    td_target   = rewards_t + GAMMA * next_values * not_done
    advantages  = (td_target - values_t).detach()

    # ------- losses ---------------------------------------------------------
    actor_loss  = -(log_probs * advantages).mean()
    critic_loss = 0.5 * (td_target.detach() - values_t).pow(2).mean()
    entropy_bonus = entropies.mean()

    loss = actor_loss + critic_loss - ENTROPY_COEF * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
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
plt.title("A2C on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.savefig("A2C_CartPole_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()
