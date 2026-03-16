"""Deep Monte-Carlo (DMC) Agent — DouZero-style deep RL for poker.

Pure deep RL via self-play: no CFR, no tree search. Learns Q(s,a) directly
from experience using a replay buffer and target network with soft updates.

Reference: Zha et al. (2021) "DouZero: Mastering DouDiZhu with Self-Play Deep RL"
"""

from packages.dmc_agent.agent import DMCAgent, ExperienceReplay

__all__ = ["DMCAgent", "ExperienceReplay"]
