"""Map nanobot tool calls and results into canonical ledger events.

These adapters are the translation layer between nanobot's internal
data structures and the causal harness event schema.
"""

from causal_harness.events.schema import Event, EventType, EventOutcome, new_event
from causal_harness.events.adapters import classify_error
from causal_harness.encoding.symbols import TOOL_TO_SYMBOL, ActionSymbol


def tool_name_to_symbol(tool_name: str) -> str:
    """Map a nanobot tool name to a Layer A behavior symbol."""
    sym = TOOL_TO_SYMBOL.get(tool_name)
    return sym.value if sym else ActionSymbol.PLAN.value


def map_tool_call_to_event(
    tool_name: str,
    tool_args: dict,
    episode_id: str,
    turn_id: int,
) -> Event:
    """Create a TOOL_CALLED event from a nanobot tool invocation."""
    category = tool_name_to_symbol(tool_name)
    return new_event(
        EventType.TOOL_CALLED,
        episode_id=episode_id,
        turn_id=turn_id,
        actor="executor",
        payload={
            "tool": tool_name,
            "category": category,
            "target": tool_args.get("file_path", tool_args.get("path", "")),
            "args_summary": _summarize(tool_args),
        },
        outcome=EventOutcome.PENDING,
    )


def map_tool_result_to_event(
    call_event: Event,
    result: str = "",
    success: bool = True,
    error_type: str = "",
    error_message: str = "",
) -> Event:
    """Create a TOOL_RESULT event paired to a previous TOOL_CALLED."""
    outcome = EventOutcome.SUCCESS if success else EventOutcome.FAILURE
    payload: dict = {
        "tool": call_event.payload.get("tool", ""),
        "category": call_event.payload.get("category", ""),
        "success": success,
        "summary": result[:300] if result else "",
    }
    if not success:
        payload["error_class"] = classify_error(error_type, error_message)
        payload["error_message"] = error_message[:300]

    return new_event(
        EventType.TOOL_RESULT,
        episode_id=call_event.episode_id,
        turn_id=call_event.turn_id,
        actor="executor",
        payload=payload,
        outcome=outcome,
    )


def map_error_to_event(
    episode_id: str,
    turn_id: int,
    error_type: str,
    error_message: str,
) -> Event:
    """Create an ERROR_DETECTED event from a nanobot error."""
    return new_event(
        EventType.ERROR_DETECTED,
        episode_id=episode_id,
        turn_id=turn_id,
        actor="system",
        payload={
            "error_class": classify_error(error_type, error_message),
            "error_type": error_type,
            "message": error_message[:500],
        },
        outcome=EventOutcome.FAILURE,
    )


def _summarize(args: dict, max_len: int = 150) -> dict:
    """Truncate large argument values."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > max_len:
            out[k] = v[:max_len] + "..."
        else:
            out[k] = v
    return out
