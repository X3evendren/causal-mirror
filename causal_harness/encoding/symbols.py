"""Layer A symbol alphabet for CSSR.

11 canonical action symbols replace the old 16-symbol action×quality alphabet.
Each symbol captures the behavioral intent, not quality (quality goes to Layer B).
"""

from enum import Enum


class ActionSymbol(str, Enum):
    """Canonical action symbols for CSSR causal state discovery.

    Intentionally narrow — 11 symbols to keep the alphabet tractable.
    Quality (correct/partial/incorrect) belongs in the side-channel (Layer B).
    """
    OBSERVE = "OBSERVE"          # read, search, list, browse, inspect
    PLAN = "PLAN"                # propose, analyze, reason aloud
    READ_FILE = "READ_FILE"      # file content read
    EDIT_FILE = "EDIT_FILE"      # file write or edit
    RUN_TEST = "RUN_TEST"        # execute tests
    VERIFY = "VERIFY"            # check result correctness
    ERROR = "ERROR"              # error detected (any tool)
    REPAIR = "REPAIR"            # fix attempt after error
    ASK_USER = "ASK_USER"        # request human input
    ESCALATE = "ESCALATE"        # request higher authority / approval
    COMPLETE = "COMPLETE"        # task or episode finished


BOUNDARY: str = "BOUNDARY"

# 11 canonical action symbols + 1 BOUNDARY marker
SYMBOL_COUNT: int = len(ActionSymbol)
EFFECTIVE_ALPHABET_SIZE: int = len(ActionSymbol) + 1

# --- Mapping tables ---

TOOL_TO_SYMBOL: dict[str, ActionSymbol] = {
    "read_file": ActionSymbol.READ_FILE,
    "search": ActionSymbol.OBSERVE,
    "list_files": ActionSymbol.OBSERVE,
    "edit_file": ActionSymbol.EDIT_FILE,
    "write_file": ActionSymbol.EDIT_FILE,
    "exec": ActionSymbol.RUN_TEST,
    "run_test": ActionSymbol.RUN_TEST,
    "shell": ActionSymbol.RUN_TEST,
    "ask": ActionSymbol.ASK_USER,
    "message": ActionSymbol.ASK_USER,
    "web_search": ActionSymbol.OBSERVE,
    "web_fetch": ActionSymbol.OBSERVE,
    "spawn": ActionSymbol.PLAN,
    "cron": ActionSymbol.PLAN,
    "mcp": ActionSymbol.OBSERVE,
    "sandbox": ActionSymbol.RUN_TEST,
    "self": ActionSymbol.PLAN,
    "image_generation": ActionSymbol.PLAN,
    "notebook": ActionSymbol.EDIT_FILE,
    "file_state": ActionSymbol.OBSERVE,
}

EVENT_TO_SYMBOL: dict[str, ActionSymbol] = {
    "tool_called": None,          # resolved via TOOL_TO_SYMBOL above
    "tool_result": None,           # paired with tool_called
    "error_detected": ActionSymbol.ERROR,
    "repair_attempted": ActionSymbol.REPAIR,
    "verification_run": ActionSymbol.VERIFY,
    "user_approval_requested": ActionSymbol.ESCALATE,
    "episode_completed": ActionSymbol.COMPLETE,
    "action_proposed": ActionSymbol.PLAN,
}
