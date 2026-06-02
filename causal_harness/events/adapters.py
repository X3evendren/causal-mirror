"""Adapters — map nanobot tool calls and errors into ledger events.

These adapters translate harness-specific data into the canonical event
schema so the ledger remains independent of any single agent framework.
"""

from causal_harness.events.schema import (
    Event, EventType, EventOutcome, new_event,
)

# --- Error classification ---

ERROR_CLASS_MAP: dict[str, str] = {
    "FileNotFoundError": "missing_file",
    "IsADirectoryError": "wrong_type",
    "PermissionError": "permission_denied",
    "TimeoutError": "timeout",
    "ConnectionError": "network_error",
    "AssertionError": "assertion_failure",
    "ValueError": "invalid_value",
    "TypeError": "invalid_value",
    "KeyError": "missing_key",
    "IndexError": "out_of_range",
    "ImportError": "missing_dependency",
    "ModuleNotFoundError": "missing_dependency",
    "SyntaxError": "syntax_error",
    "RuntimeError": "runtime_error",
}


def classify_error(exc_type: str, exc_message: str = "") -> str:
    """Map a Python exception type to a stable error class."""
    for py_exc, error_class in ERROR_CLASS_MAP.items():
        if py_exc in exc_type:
            return error_class
    return "unknown_error"


# --- Tool call adapters ---

# Tool category → action symbol (used by behavior encoder later)
TOOL_CATEGORY_MAP: dict[str, str] = {
    "read_file": "READ",
    "search": "READ",
    "list_files": "READ",
    "edit_file": "EDIT",
    "write_file": "EDIT",
    "exec": "EXEC",
    "run_test": "TEST",
    "shell": "EXEC",
    "ask": "ASK",
    "message": "SEND",
    "web_search": "READ",
    "web_fetch": "READ",
    "spawn": "DELEGATE",
}


def tool_event(
    tool_name: str,
    tool_args: dict,
    episode_id: str,
    turn_id: int,
) -> Event:
    """Create a TOOL_CALLED event from a nanobot tool invocation."""
    return new_event(
        EventType.TOOL_CALLED,
        episode_id=episode_id,
        turn_id=turn_id,
        actor="executor",
        payload={
            "tool": tool_name,
            "category": TOOL_CATEGORY_MAP.get(tool_name, "OTHER"),
            "target": tool_args.get("file_path", tool_args.get("path", "")),
            "args": _summarize_args(tool_args),
        },
        outcome=EventOutcome.PENDING,
    )


def tool_result_event(
    call_event: Event,
    success: bool,
    result_summary: str = "",
    error_info: dict | None = None,
) -> Event:
    """Create a TOOL_RESULT event paired to a TOOL_CALLED event."""
    outcome = EventOutcome.SUCCESS if success else EventOutcome.FAILURE
    payload: dict = {
        "tool": call_event.payload.get("tool", ""),
        "category": call_event.payload.get("category", ""),
        "success": success,
        "summary": result_summary[:500],
    }
    if error_info:
        payload["error_class"] = classify_error(
            error_info.get("type", ""), error_info.get("message", "")
        )
        payload["error_message"] = error_info.get("message", "")[:500]

    return new_event(
        EventType.TOOL_RESULT,
        episode_id=call_event.episode_id,
        turn_id=call_event.turn_id,
        actor="executor",
        payload=payload,
        outcome=outcome,
    )


def error_event(
    episode_id: str,
    turn_id: int,
    error_class: str,
    message: str,
    recoverable: bool = True,
) -> Event:
    """Create an ERROR_DETECTED event."""
    return new_event(
        EventType.ERROR_DETECTED,
        episode_id=episode_id,
        turn_id=turn_id,
        actor="system",
        payload={
            "error_class": error_class,
            "message": message[:1000],
            "recoverable": recoverable,
        },
        outcome=EventOutcome.FAILURE,
    )


def verification_event(
    episode_id: str,
    turn_id: int,
    verification_type: str,
    passed: bool,
    details: str = "",
) -> Event:
    """Create a VERIFICATION_RUN event."""
    return new_event(
        EventType.VERIFICATION_RUN,
        episode_id=episode_id,
        turn_id=turn_id,
        actor="verifier",
        payload={
            "type": verification_type,
            "passed": passed,
            "details": details[:500],
        },
        outcome=EventOutcome.SUCCESS if passed else EventOutcome.FAILURE,
    )


def approval_event(
    episode_id: str,
    turn_id: int,
    action: str,
    risk_tier: str,
    granted: bool = False,
) -> Event:
    """Create a USER_APPROVAL_REQUESTED event."""
    return new_event(
        EventType.USER_APPROVAL_REQUESTED,
        episode_id=episode_id,
        turn_id=turn_id,
        actor="governor",
        payload={
            "action": action,
            "risk_tier": risk_tier,
            "granted": granted,
        },
        outcome=EventOutcome.PENDING if not granted else EventOutcome.SUCCESS,
    )


# -- helpers --

def _summarize_args(args: dict, max_len: int = 200) -> dict:
    """Truncate large argument values for compact event storage."""
    result = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > max_len:
            result[k] = v[:max_len] + "..."
        else:
            result[k] = v
    return result
