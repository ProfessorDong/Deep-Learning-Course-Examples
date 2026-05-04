# Deep Reinforcement Learning — PyTorch Demos

**ELC 5365: Deep Learning, Spring 2026**
**Dr. Liang Dong, Baylor University**

Self-contained, single-file PyTorch demos for the Deep RL lecture. Each
script trains from scratch, **pops up a live pygame window for periodic
class demos**, prints progress to the console, and saves a learning-curve
PNG on completion. They share a consistent template so two scripts can
be diffed side-by-side to highlight exactly what changes between
algorithms.

## Install

```
pip install gymnasium "gymnasium[classic-control]" torch numpy matplotlib
```

`pygame` is pulled in by `gymnasium[classic-control]` and is what
renders the live CartPole / Pendulum window.

## The 10 demos

### Value-based (discrete action) — CartPole-v1

| File | Algorithm | Key idea |
|------|-----------|----------|
| `DQN_CartPole_PyTorch.py`        | DQN          | Q-learning with replay buffer + target network |
| `DoubleDQN_CartPole_PyTorch.py`  | Double DQN   | Decouple action selection (online) from evaluation (target) to fix maximization bias |
| `DuelingDQN_CartPole_PyTorch.py` | Dueling DQN  | Q(s,a) = V(s) + (A(s,a) − mean A) — better state-value learning |
| `PER_DQN_CartPole_PyTorch.py`    | Prioritized DQN | Sample transitions ∝ |TD error|^α with importance-sampling correction |

### Policy-gradient (discrete action) — CartPole-v1

| File | Algorithm | Key idea |
|------|-----------|----------|
| `REINFORCE_CartPole_PyTorch.py` | REINFORCE | Monte-Carlo policy gradient with running-mean baseline |
| `A2C_CartPole_PyTorch.py`       | A2C       | One-step TD advantage with shared-trunk actor-critic |
| `PPO_CartPole_PyTorch.py`       | PPO       | Clipped surrogate, GAE, K-epoch minibatch SGD over rollouts |

### Continuous control — Pendulum-v1

| File | Algorithm | Key idea |
|------|-----------|----------|
| `DDPG_Pendulum_PyTorch.py` | DDPG | Deterministic actor-critic, replay + Polyak target nets |
| `TD3_Pendulum_PyTorch.py`  | TD3  | DDPG + twin critics (clipped double Q) + delayed policy + target smoothing |
| `SAC_Pendulum_PyTorch.py`  | SAC  | Maximum-entropy RL with squashed-Gaussian policy and auto-tuned α |

## Running

Every script is a plain `python3` invocation:

```
python3 DQN_CartPole_PyTorch.py
```

Each script auto-detects CUDA. On a modern GPU training finishes in
1–5 minutes; on CPU expect 5–20 minutes. **Add ~1–3 minutes for the
on-screen demo episodes** (numbers below).

## Live class-demo pattern

Every script trains a *silent* `env` for speed and uses a separate
**`render_env`** with `render_mode="human"` for visualization.
Rendering every training step would slow learning ~30×, so we render
only at three points:

1. **Before training** — one episode with the *untrained* policy.
   Students see the cart fall in 0.2 s (or the pendulum flail
   randomly): the baseline.
2. **Every `RENDER_EVERY` training episodes** — one *greedy*
   (deterministic) demo with the current policy. Over the course of
   training the cart lasts longer and longer; the pendulum slowly
   learns to swing up and balance.
3. **After training** — `N_FINAL_DEMOS` showcase episodes (default 3)
   with the trained greedy policy.

Two knobs at the top of every script control this:

```python
RENDER_EVERY  = 50    # (or 30, or RENDER_EVERY_ITERS=5 in PPO)
N_FINAL_DEMOS = 3
```

Increase `RENDER_EVERY` to spend less wall-clock on demos; set it
larger than the training length to get only the before / after demos.

| Script | `RENDER_EVERY` default | Total demo episodes |
|--------|------------------------|---------------------|
| DQN, DoubleDQN, DuelingDQN, PER-DQN | 50 (episodes) | ~10–20 |
| REINFORCE, A2C | 100 (episodes) | ~6–15 |
| PPO | 5 (PPO iterations) | ~20 |
| DDPG, TD3, SAC | 30 (episodes) | 5 |

### Headless / SSH use

If you are running over SSH or want **no** pygame window, replace
`render_env = gym.make(..., render_mode="human")` with
`render_mode="rgb_array"` and either delete the `render_demo` body
or save frames to disk. The training env (`env`) is already silent
either way.

## Suggested classroom order

The demos are intentionally arranged to match the lecture progression:

1. **DQN → Double DQN → Dueling DQN → PER-DQN** — incremental
   improvements over the same value-based template.
2. **REINFORCE → A2C → PPO** — vanilla policy gradient → bootstrapped
   actor-critic → trust-region-style clipping.
3. **DDPG → TD3 → SAC** — extending actor-critic to continuous actions
   and stochastic max-entropy policies.

For a single-lecture demo, run **DQN** (fast, dramatic
"falls → balances" arc), **PPO** (state-of-the-art on-policy), and
**SAC** (state-of-the-art off-policy + continuous).

## Notes

- All scripts use the modern `gymnasium` API: `obs, info = env.reset()`
  and the 5-tuple `obs, r, terminated, truncated, info = env.step(a)`.
- `terminated` is the only flag that should zero out bootstrap;
  episodes ending only by `truncated` (time-limit) still bootstrap
  from V or Q.
- Hyperparameters were chosen to learn within ~150–800 episodes (or
  100k steps for PPO) on the listed environments — they are tuned for
  *clarity*, not benchmark scores.
- Demo episodes use the *deterministic / greedy* action of each
  algorithm (argmax Q, argmax logits, deterministic actor, or the
  squashed-Gaussian *mean* for SAC) so what the class sees is "what
  the policy has learned," not exploration noise.
