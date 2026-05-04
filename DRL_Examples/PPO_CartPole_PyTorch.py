# -*- coding: utf-8 -*-
"""
=============================================================================
ELC 5365: Deep Learning, Spring 2026
Dr. Liang Dong, Baylor University

Proximal Policy Optimization (PPO, clipped surrogate) on CartPole-v1
--------------------------------------------------------------------
PPO is the workhorse on-policy policy-gradient algorithm.  Per iteration
it does:

  1) Roll out the current policy pi_{theta_old} for T steps (here, until
     a fixed roll-out length is reached, possibly across episodes).
  2) Compute Generalized Advantage Estimation (GAE) advantages and
     value targets.
  3) Run K epochs of mini-batch SGD on the clipped surrogate

        L_CLIP(theta) = E_t[ min( r_t * A_hat ,
                                  clip(r_t, 1-eps, 1+eps) * A_hat ) ]

     where r_t = pi_theta(a_t|s_t) / pi_{theta_old}(a_t|s_t).

  4) Combined loss:

        L = -L_CLIP + c1 * L_VF - c2 * H[pi]

     ( -L_CLIP makes it a minimization; L_VF = (V_phi(s_t) - V_target)^2;
       H[pi] is the policy entropy; c2 * H is the entropy bonus that
       discourages premature policy collapse. )

Reference:
  J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal
  Policy Optimization Algorithms," arXiv:1707.06347, 2017.

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

RENDER_EVERY_ITERS = 5     # PPO iterations (rollouts) between rendered demos
N_FINAL_DEMOS      = 3


# ---------------------------- hyperparameters -------------------------------
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
LR             = 3e-4
EPOCHS         = 4              # SGD epochs per rollout
MINIBATCH_SIZE = 64
ROLLOUT_STEPS  = 1024
VF_COEF        = 0.5
ENT_COEF       = 0.01
TOTAL_STEPS    = 100_000        # total environment steps; ~100 PPO iterations


# ---------------------------- shared-trunk actor-critic ---------------------
class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_obs, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head  = nn.Linear(64, 1)

    def forward(self, s):
        h = self.trunk(s)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ---------------------------- setup -----------------------------------------
n_actions = env.action_space.n
s, _      = env.reset()
n_obs     = len(s)

net       = ActorCritic(n_obs, n_actions).to(device)
optimizer = optim.Adam(net.parameters(), lr=LR)


@torch.no_grad()
def policy_step(state):
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logits, value = net(s)
    dist = Categorical(logits=logits)
    a    = dist.sample()
    return int(a.item()), dist.log_prob(a).squeeze(0), value.squeeze(0)


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    GAE(gamma, lambda):
        delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t) - V(s_t)
        A_t = sum_{l>=0} (gamma * lam)^l * delta_{t+l}
    Computed by backward recursion:  A_t = delta_t + gamma*lam*(1-done_t)*A_{t+1}.
    """
    T   = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32, device=device)
    last_gae = 0.0
    next_val = last_value
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta    = rewards[t] + gamma * next_val * not_done - values[t]
        adv[t]   = last_gae = delta + gamma * lam * not_done * last_gae
        next_val = values[t]
    returns = adv + values
    return adv, returns


def render_demo(label):
    """Greedy (argmax of policy logits) episode in the rendered env."""
    s, _ = render_env.reset()
    total = 0
    while True:
        with torch.no_grad():
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(s_t)
            a = int(logits.argmax(dim=1).item())
        s, r, terminated, truncated, _ = render_env.step(a)
        total += r
        if terminated or truncated:
            break
    print(f"  [demo {label}] return = {total:.0f}")


# ---------------------------- training loop ---------------------------------
episode_returns = []
ep_return = 0.0
state, _ = env.reset()
total_steps_done = 0
iters_done = 0

print("Demo before training (untrained policy)...")
render_demo(label="iter  0")

while total_steps_done < TOTAL_STEPS:
    # storage for one rollout
    s_buf, a_buf, lp_buf, v_buf, r_buf, d_buf = [], [], [], [], [], []

    for _ in range(ROLLOUT_STEPS):
        a, lp, v = policy_step(state)
        next_state, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        s_buf.append(state)
        a_buf.append(a)
        lp_buf.append(lp)
        v_buf.append(v)
        r_buf.append(r)
        d_buf.append(float(terminated))   # only terminated breaks bootstrap

        ep_return += r
        state = next_state
        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            state, _ = env.reset()

    total_steps_done += ROLLOUT_STEPS

    # bootstrap value at final state
    with torch.no_grad():
        s_T = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        _, last_value = net(s_T)
        last_value = last_value.squeeze(0)

    rewards = torch.tensor(r_buf, dtype=torch.float32, device=device)
    dones   = torch.tensor(d_buf, dtype=torch.float32, device=device)
    values  = torch.stack(v_buf)
    adv, returns = compute_gae(rewards, values, dones, last_value)

    # ------ flatten to tensors for SGD ------
    states_t   = torch.tensor(s_buf, dtype=torch.float32, device=device)
    actions_t  = torch.tensor(a_buf, dtype=torch.long,    device=device)
    old_logp   = torch.stack(lp_buf).detach()
    returns    = returns.detach()
    advantages = adv.detach()

    # ------ K epochs of minibatch updates ------
    N = ROLLOUT_STEPS
    idx = torch.arange(N, device=device)
    for epoch in range(EPOCHS):
        idx = idx[torch.randperm(N)]
        for start in range(0, N, MINIBATCH_SIZE):
            mb = idx[start:start + MINIBATCH_SIZE]

            logits, value = net(states_t[mb])
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(actions_t[mb])
            entropy  = dist.entropy().mean()

            # ----- normalized advantage (per minibatch) -----
            adv_mb = advantages[mb]
            adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

            # ----- clipped surrogate -----
            ratio = torch.exp(new_logp - old_logp[mb])
            unclipped = ratio * adv_mb
            clipped   = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_mb
            policy_loss = -torch.min(unclipped, clipped).mean()

            # ----- value loss -----
            value_loss = (returns[mb] - value).pow(2).mean()

            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

    iters_done += 1
    if episode_returns:
        recent = episode_returns[-20:]
        print(f"Steps {total_steps_done:6d}/{TOTAL_STEPS}  "
              f"episodes done = {len(episode_returns):4d}  "
              f"return(last 20) avg = {sum(recent)/len(recent):6.1f}")

    if iters_done % RENDER_EVERY_ITERS == 0:
        render_demo(label=f"iter{iters_done:3d}")

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
plt.title("PPO on CartPole-v1")
plt.legend()
plt.tight_layout()
plt.savefig("PPO_CartPole_curve.png", dpi=120)
plt.show()

env.close()
render_env.close()
