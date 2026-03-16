"""Information Set Monte Carlo Tree Search (IS-MCTS) Agent.

Combines MCTS with opponent modeling for poker decision-making.
Uses determinization to handle hidden information and UCB1 for
tree exploration.

Reference: Cowling et al. (2012) "Information Set Monte Carlo Tree Search"
"""

from packages.mcts_agent.agent import ISMCTSAgent, MCTSNode

__all__ = ["ISMCTSAgent", "MCTSNode"]
