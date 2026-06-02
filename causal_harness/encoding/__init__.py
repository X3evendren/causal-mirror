"""Behavior encoding — two-layer representation for CSSR + risk models."""

from causal_harness.encoding.symbols import (
    ActionSymbol,
    BOUNDARY,
    TOOL_TO_SYMBOL,
    SYMBOL_COUNT,
)
from causal_harness.encoding.encoder import BehaviorEncoderV2

__all__ = [
    "ActionSymbol",
    "BOUNDARY",
    "TOOL_TO_SYMBOL",
    "SYMBOL_COUNT",
    "BehaviorEncoderV2",
]
