"""PPO (Proximal Policy Optimization) poker agent.

Uses reinforcement learning via self-play to learn poker strategy.
Policy and value networks are pure-Python SimpleNN (no PyTorch dependency).

Key features:
- PPO-clip objective with epsilon=0.2
- GAE (Generalized Advantage Estimation) for variance reduction
- Experience buffer for batch updates
- Compatible with BaselineAgent decide() interface
"""

from packages.ppo_agent.agent import PPOAgent

__all__ = ["PPOAgent"]
