from __future__ import annotations

from enum import Enum


class Street(str, Enum):
    PRE_FLOP = "pre_flop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    TERMINAL = "terminal"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    POST_SMALL_BLIND = "post_small_blind"
    POST_BIG_BLIND = "post_big_blind"
