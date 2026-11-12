"""
Argentine Truco Game Engine & System 1 Small Language Model (SLM) package.
"""

from truco.engine import (
    Action,
    ACTION_NAMES,
    Card,
    compute_envido,
    DECK,
    NUM_ACTIONS,
    PendingState,
    Suit,
    TrucoHandState,
)
from truco.model import System1TrucoSLM
from truco.tokenizer import TrucoTokenizer, infoset_key
from truco.trainer import (
    evaluate_agent,
    HeuristicAnchorAgent,
    MCCFRTrainer,
    train_system1_slm,
)

__all__ = [
    "Action",
    "ACTION_NAMES",
    "Card",
    "compute_envido",
    "DECK",
    "NUM_ACTIONS",
    "PendingState",
    "Suit",
    "TrucoHandState",
    "System1TrucoSLM",
    "TrucoTokenizer",
    "infoset_key",
    "evaluate_agent",
    "HeuristicAnchorAgent",
    "MCCFRTrainer",
    "train_system1_slm",
]
