#!/usr/bin/env python3
"""Causal Mirror V2 — End-to-End Pipeline Demo.

Simulates a coding agent session across multiple episodes, showing the
complete observe → encode → analyze → predict → intervene pipeline.

No external dependencies (no nanobot needed). Uses simulated events to
demonstrate every component working together.
"""

import sys
import os
import tempfile

# Add project root to path for direct invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cssr import CSSRConfig
from causal_harness.events.schema import (
    Event, EventType, EventOutcome, new_event,
)
from causal_harness.encoding.symbols import ActionSymbol, BOUNDARY
from causal_harness.controller.policy import PolicyResult
from causal_harness.pipeline import CausalPipeline


# ── helpers ────────────────────────────────────────────────────

def make_tool_call(name: str, ep_id: str = "ep1", turn: int = 1) -> Event:
    return new_event(
        EventType.TOOL_CALLED, ep_id, turn,
        payload={"tool": name},
    )


def make_tool_result(
    name: str, success: bool, ep_id: str = "ep1", turn: int = 1,
) -> Event:
    return new_event(
        EventType.TOOL_RESULT, ep_id, turn,
        payload={"tool": name, "success": success},
        outcome=EventOutcome.SUCCESS if success else EventOutcome.FAILURE,
    )


def make_error(ep_id: str = "ep1", turn: int = 1) -> Event:
    return new_event(
        EventType.ERROR_DETECTED, ep_id, turn,
        payload={"error_class": "assertion_failure", "message": "test failed"},
        outcome=EventOutcome.FAILURE,
    )


def simulate_episode(
    pipeline: CausalPipeline,
    goal: str,
    tool_sequence: list[tuple[str, bool]],
    verbose: bool = True,
) -> str:
    """Simulate one episode with a sequence of (tool_name, success) actions."""
    ep_id = pipeline.start_episode(goal)

    if verbose:
        print(f"\n{'='*56}")
        print(f"  Episode: {goal}")
        print(f"{'='*56}")

    for i, (tool_name, success) in enumerate(tool_sequence):
        turn = i + 1
        # Tool call
        tc = make_tool_call(tool_name, ep_id, turn)
        sym = pipeline.record_event(tc)

        # Evaluate intervention
        decision = pipeline.evaluate_action(sym or "PLAN")

        if verbose:
            status = "OK" if success else "ERR"
            risk_str = f"P(fail)={decision.risk.p_failure:.2f}"
            interv_str = decision.intervention.result.value
            print(f"  [{status}] {tool_name:12s}  {risk_str:14s}  -> {interv_str}")

        # Tool result
        tr = make_tool_result(tool_name, success, ep_id, turn)
        pipeline.record_event(tr)

        if not success:
            err = make_error(ep_id, turn)
            pipeline.record_event(err)

    # Determine episode outcome
    any_failure = any(not s for _, s in tool_sequence)
    outcome = EventOutcome.FAILURE if any_failure else EventOutcome.SUCCESS
    pipeline.end_episode(outcome)

    if verbose:
        print(f"  Outcome: {outcome.value}")

    return ep_id


# ── main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Causal Mirror V2 — 端到端流水线演示")
    print("=" * 60)
    print()

    # Create pipeline
    pipeline = CausalPipeline(
        cssr_config=CSSRConfig(L_max=None, alpha=0.05, min_count=2),
        markov_order=3,
        risk_threshold=0.50,
        loop_window=5,
        loop_threshold=3,
    )

    # ── Episode 1: initial attempt with errors ──
    simulate_episode(pipeline, "Debug auth module", [
        ("read_file", True),
        ("edit_file", False),   # ← mistake
        ("edit_file", True),    # ← fix
        ("shell", True),        # run tests
    ])

    # Analyze after first episode
    metrics = pipeline.analyze()
    if metrics:
        print(f"\n  ── CSSR Analysis (after Ep 1) ──")
        print(f"  n_states = {metrics['n_states']}")
        print(f"  C_mu     = {metrics['statistical_complexity']:.4f} bits")
        print(f"  h_mu     = {metrics['entropy_rate']:.4f} bits")
        print(f"  E        = {metrics['excess_entropy']:.4f} bits")
        print(f"  χ        = {metrics['predictive_information']:.4f} bits")

    # ── Episode 2: another work session ──
    simulate_episode(pipeline, "Add file upload feature", [
        ("read_file", True),
        ("read_file", True),
        ("edit_file", True),
        ("shell", True),
    ])

    # ── Episode 3: task with repeated failures ──
    simulate_episode(pipeline, "Fix flaky test (tricky!)", [
        ("read_file", True),
        ("edit_file", False),   # fix attempt 1 fails
        ("edit_file", False),   # fix attempt 2 fails — pattern detected
        ("edit_file", True),    # finally works
        ("shell", True),
    ])

    # Analyze again after more data
    metrics2 = pipeline.analyze()
    if metrics2:
        print(f"\n  ── CSSR Analysis (after Ep 3) ──")
        print(f"  n_states = {metrics2['n_states']}")
        print(f"  C_mu     = {metrics2['statistical_complexity']:.4f} bits")
        print(f"  h_mu     = {metrics2['entropy_rate']:.4f} bits")
        print(f"  E        = {metrics2['excess_entropy']:.4f} bits")
        print(f"  χ        = {metrics2['predictive_information']:.4f} bits")

        if metrics2['n_states'] > (metrics.get('n_states', 0) if metrics else 0):
            print(f"  -> Causal states increased! Behavior structure emerging.")

    # ── Train risk model ──
    print(f"\n  ── Training Risk Model ──")
    trained = pipeline.train()
    print(f"  Training data: {len(pipeline._training_episodes)} episodes")
    print(f"  Training result: {'success' if trained else 'failed'}")

    # ── Evaluate: what would happen next? ──
    print(f"\n  ── Intervention Test: Proposed Actions ──")
    test_actions = [
        ("OBSERVE", {}),
        ("EDIT_FILE", {"risk_tier": "medium", "reversible": True}),
        ("EDIT_FILE", {"risk_tier": "high", "reversible": False, "ambiguity": "high"}),
        ("EDIT_FILE", {"last_error_class": "assertion_failure"}),
    ]

    for action, extra in test_actions:
        decision = pipeline.evaluate_action(action, extra_context=extra)
        symbol = "WARN" if decision.intervention.result != PolicyResult.ALLOW else "OK"
        print(
            f"  {symbol} {action:12s}  "
            f"risk={decision.risk.p_failure:.2f}  "
            f"conf={decision.risk.confidence:.2f}  "
            f"→ {decision.intervention.result.value}"
        )
        if decision.intervention.reason:
            print(f"     reason: {decision.intervention.reason}")

    # Record error for repair-switching policy test
    pipeline.policy.record_error("assertion_failure")
    pipeline.policy.record_error("assertion_failure")

    print(f"\n  ── After recording 2 assertion_failure errors ──")
    decision = pipeline.evaluate_action(
        "EDIT_FILE",
        extra_context={"last_error_class": "assertion_failure"},
    )
    print(
        f"  {'✓' if decision.intervention.result == PolicyResult.REPLAN else '?'} "
        f"EDIT_FILE  → {decision.intervention.result.value}"
    )
    print(f"     reason: {decision.intervention.reason}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Pipeline Summary")
    print(f"{'='*60}")
    summary = pipeline.summary()
    for k, v in summary.items():
        if v is not None:
            print(f"  {k:24s}: {v}")

    print(f"\n  OK Demo complete. Pipeline data at: {summary['ledger_dir']}")


if __name__ == "__main__":
    # Ensure UTF-8 output on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
