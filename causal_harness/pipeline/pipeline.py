"""CausalPipeline — connect all 7 causal_harness modules into an end-to-end flow.

Data flow:
  Event -> BehaviorEncoderV2 -> symbol accumulation
  Accumulated symbols -> CSSR -> epsilon-machine + metrics
  (context, next_action) -> MarkovRiskModel -> RiskPrediction
  RiskPrediction -> PolicyController -> Intervention
  RiskPrediction + action_symbol -> VerificationGate -> VerificationDecision

The pipeline works independently of nanobot — use it standalone or
attach it to a CausalMirrorHook for live intervention.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cssr import CSSR, CSSRConfig
from causal_harness.events.schema import (
    Event, EventType, EventOutcome, new_event,
)
from causal_harness.events.ledger import EventLedger
from causal_harness.events.episode import EpisodeManager
from causal_harness.encoding.encoder import BehaviorEncoderV2
from causal_harness.encoding.symbols import (
    ActionSymbol, TOOL_TO_SYMBOL, BOUNDARY,
)
from causal_harness.models.base import RiskPrediction, RiskModel
from causal_harness.models.markov_baseline import MarkovRiskModel
from causal_harness.controller.policy import (
    PolicyController, PolicyResult, Intervention,
)
from causal_harness.controller.verification import (
    VerificationGate, VerificationDecision, RiskTier, ACTION_RISK_TIERS,
)
from causal_harness.controller.loop_detection import LoopDetector


@dataclass
class PipelineContext:
    """Runtime context at one decision point.

    Attributes:
        symbol_sequence: Complete accumulated Layer A symbols.
        recent_context: Last N symbols for risk model input.
        current_state_id: CSSR causal state id (-1 if unknown).
        loop_count: From LoopDetector.
        recent_actions: Last W actions from LoopDetector.
        metrics: Latest CSSR metrics (None if not yet analyzed).
        episode_id: Active episode identifier.
        turn_id: Current turn number.
    """
    symbol_sequence: list[str] = field(default_factory=list)
    recent_context: list[str] = field(default_factory=list)
    current_state_id: int = -1
    loop_count: int = 0
    recent_actions: list[str] = field(default_factory=list)
    metrics: dict | None = None
    episode_id: str = ""
    turn_id: int = 0


@dataclass
class PipelineDecision:
    """Combined output of one action evaluation.

    Attributes:
        intervention: Policy controller's recommendation.
        verification: Gate's verification requirement.
        risk: Risk model's prediction.
        context: The pipeline context at evaluation time.
        action_symbol: The action being evaluated.
    """
    intervention: Intervention
    verification: VerificationDecision
    risk: RiskPrediction
    context: PipelineContext
    action_symbol: str = ""


class CausalPipeline:
    """Orchestrator connecting all causal_harness modules.

    Usage (standalone):
        pipeline = CausalPipeline(ledger_dir="pipeline_data")
        pipeline.start_episode("Debug auth module")

        # Record events as they happen
        for event in simulated_events:
            pipeline.record_event(event)

        pipeline.end_episode(EventOutcome.SUCCESS)

        # Analyze behavior structure
        metrics = pipeline.analyze()
        print(f"C_mu = {metrics['statistical_complexity']:.4f}")

        # Train risk model from accumulated episodes
        pipeline.train()

        # Evaluate a proposed action
        decision = pipeline.evaluate_action("EDIT_FILE")
        print(f"Intervention: {decision.intervention.result.value}")

    Usage (with nanobot hook):
        hook = CausalMirrorHook(ledger, pipeline=pipeline)
        # Hook automatically calls pipeline.record_event() and
        # pipeline.evaluate_action() during the agent loop.
    """

    def __init__(
        self,
        ledger_dir: str | Path = "",
        cssr_config: CSSRConfig | None = None,
        markov_order: int = 3,
        risk_threshold: float = 0.60,
        loop_window: int = 10,
        loop_threshold: int = 3,
    ):
        """
        Args:
            ledger_dir: Directory for the event ledger. Uses a temp dir if empty.
            cssr_config: CSSR configuration. Defaults to L_max=None, alpha=0.01.
            markov_order: Order for the Markov risk model.
            risk_threshold: P(failure) above which verification is triggered.
            loop_window: Loop detector sliding window size.
            loop_threshold: Repetitions needed to flag a loop.
        """
        if not ledger_dir:
            ledger_dir = tempfile.mkdtemp(prefix="causal_pipeline_")

        # --- Module 1: Event Ledger ---
        self.ledger = EventLedger(ledger_dir)

        # --- Module 2: Behavior Encoder ---
        self.encoder = BehaviorEncoderV2(insert_boundaries=True)

        # --- Module 3: CSSR (lazily fit) ---
        self._cssr_config = cssr_config or CSSRConfig(
            L_max=None, alpha=0.01, min_count=3,
        )
        self._cssr_model: CSSR | None = None
        self._latest_metrics: dict | None = None

        # --- Module 4: Risk Model ---
        self.risk_model: RiskModel = MarkovRiskModel(
            order=markov_order, min_count=1,
        )

        # --- Module 5: Policy Controller ---
        self.policy = PolicyController()

        # --- Module 6: Verification Gate ---
        self.verification_gate = VerificationGate(
            risk_threshold=risk_threshold,
        )

        # --- Module 7: Loop Detector ---
        self.loop_detector = LoopDetector(
            window=loop_window,
            threshold=loop_threshold,
        )

        # --- Internal state ---
        self._episode_manager = EpisodeManager()
        self._symbol_sequence: list[str] = []
        self._current_turn: int = 0
        self._current_episode_id: str = ""

        # Training buffer
        self._training_episodes: list[list[str]] = []
        self._training_outcomes: list[int] = []
        self._trained: bool = False

        # Episode symbol buffers — track per-episode symbols for
        # incremental training when end_episode is called
        self._episode_symbols: list[str] = []

    # ── Lifecycle ──────────────────────────────────────────────

    def start_episode(
        self, goal: str = "", metadata: dict | None = None,
    ) -> str:
        """Begin a new episode. Auto-completes the previous one.

        Inserts a BOUNDARY marker between episodes.
        """
        ep, start_event, auto_complete = self._episode_manager.begin_episode(
            goal, metadata or {},
        )
        # Record any auto-complete event from prior episode
        if auto_complete is not None:
            self.ledger.append(auto_complete)

        self.ledger.append(start_event)
        self._current_episode_id = ep.episode_id
        self._current_turn = 0
        self._episode_symbols = []

        return ep.episode_id

    def end_episode(self, outcome: EventOutcome = EventOutcome.SUCCESS) -> None:
        """End the current episode. Extracts training data and appends BOUNDARY."""
        end_event = self._episode_manager.end_episode(outcome)
        if end_event is None:
            return
        self.ledger.append(end_event)

        # Encode the completion event
        comp_sym = self.encoder.encode_event(end_event)
        if comp_sym is not None:
            self._episode_symbols.append(comp_sym)
            self._symbol_sequence.append(comp_sym)

        # Accumulate for training (before clearing)
        if self._episode_symbols:
            is_failure = 1 if outcome != EventOutcome.SUCCESS else 0
            self._training_episodes.append(list(self._episode_symbols))
            self._training_outcomes.append(is_failure)

        # Append BOUNDARY marker between episodes
        self._symbol_sequence.append(BOUNDARY)

        self._episode_symbols = []
        self._current_episode_id = ""

    def record_event(self, event: Event) -> str | None:
        """Record one event: append to ledger and encode to symbol.

        Returns the encoded symbol string, or None if the event
        type is not encodable.
        """
        self.ledger.append(event)

        sym = self.encoder.encode_event(event)
        if sym is not None and sym != BOUNDARY:
            self._episode_symbols.append(sym)
            self._symbol_sequence.append(sym)
            self.loop_detector.observe(
                sym, state_id=self.get_current_state_id(),
            )

        return sym

    # ── Analysis ───────────────────────────────────────────────

    def analyze(self) -> dict | None:
        """Run CSSR on accumulated symbols. Returns metrics or None.

        Requires at least 10 symbols. Caches the CSSR model.
        """
        if len(self._symbol_sequence) < 10:
            return None

        try:
            # Adjust L_max to sequence length
            config = CSSRConfig(
                L_max=min(
                    self._cssr_config.L_max or 10,
                    len(self._symbol_sequence) - 1,
                ),
                alpha=self._cssr_config.alpha,
                test=self._cssr_config.test,
                correction=self._cssr_config.correction,
                min_count=self._cssr_config.min_count,
                remove_transient=self._cssr_config.remove_transient,
            )

            cssr = CSSR(config)
            cssr.fit(self._symbol_sequence)
            self._cssr_model = cssr
            self._latest_metrics = dict(cssr.metrics)
            return self._latest_metrics
        except Exception:
            return None

    def get_current_state_id(self) -> int:
        """Resolve the current causal state by tracing transitions.

        Follows the epsilon-machine from start_state along the
        recent symbol context. Returns state_id or -1.
        """
        if self._cssr_model is None or self._cssr_model.machine is None:
            return -1
        machine = self._cssr_model.machine
        if machine.start_state is None:
            return -1

        current = machine.start_state
        # Walk through the last L_max symbols looking for known transitions
        L = min(machine.L_max, len(self._symbol_sequence))
        recent = self._symbol_sequence[-L:] if L > 0 else []

        for sym in recent:
            target_id = current.transitions.get(sym)
            if target_id is None:
                # Transition not defined — stay in current state
                continue
            tid = target_id if isinstance(target_id, int) else target_id.state_id
            next_state = next(
                (s for s in machine.states if s.state_id == tid), None,
            )
            if next_state is None:
                break
            current = next_state

        return current.state_id

    @property
    def metrics(self) -> dict | None:
        """Latest CSSR metrics (None if analyze() hasn't been called)."""
        return self._latest_metrics

    @property
    def n_states(self) -> int:
        """Number of discovered causal states."""
        if self._latest_metrics:
            return self._latest_metrics.get("n_states", 0)
        return 0

    # ── Evaluation ─────────────────────────────────────────────

    def evaluate_action(
        self,
        action_symbol: str,
        extra_context: dict | None = None,
    ) -> PipelineDecision:
        """Evaluate a proposed action through the full pipeline.

        Args:
            action_symbol: Layer A action symbol (e.g. "EDIT_FILE").
            extra_context: Additional context for policy evaluation.

        Returns:
            PipelineDecision with intervention, verification, and risk.
        """
        ctx = self._build_pipeline_context()
        extra = extra_context or {}

        # Risk prediction
        recent = self._symbol_sequence[-self.risk_model.order:]  # type: ignore
        risk = self.risk_model.predict(recent, action_symbol)

        # Policy evaluation
        policy_ctx = self._build_policy_context(extra)
        intervention = self.policy.evaluate(risk, policy_ctx)

        # Verification gate
        verification = self.verification_gate.evaluate(risk, action_symbol)

        return PipelineDecision(
            intervention=intervention,
            verification=verification,
            risk=risk,
            context=ctx,
            action_symbol=action_symbol,
        )

    def _build_pipeline_context(self) -> PipelineContext:
        """Assemble the current PipelineContext from internal state."""
        recent_ct = self._symbol_sequence[-self.risk_model.order:]  # type: ignore
        return PipelineContext(
            symbol_sequence=list(self._symbol_sequence),
            recent_context=list(recent_ct),
            current_state_id=self.get_current_state_id(),
            loop_count=self.loop_detector.loop_count(),
            recent_actions=self.loop_detector.recent_pattern(),
            metrics=self._latest_metrics,
            episode_id=self._current_episode_id,
            turn_id=self._current_turn,
        )

    def _build_policy_context(self, extra: dict) -> dict:
        """Build the context dict for PolicyController.evaluate()."""
        ctx: dict = {
            "loop_count": self.loop_detector.loop_count(),
            "has_read_file": extra.get("has_read_file", True),
            "has_searched": extra.get("has_searched", False),
            "last_error_class": extra.get("last_error_class", ""),
            "risk_tier": extra.get("risk_tier", "medium"),
            "reversible": extra.get("reversible", True),
            "ambiguity": extra.get("ambiguity", "low"),
        }
        ctx.update(extra)
        return ctx

    # ── Training ───────────────────────────────────────────────

    def train(self) -> bool:
        """Extract training data from the ledger and train the risk model.

        Uses TrainingDataExtractor to read all episodes from the ledger
        and build (symbol sequence, outcome) pairs for training.

        Returns True if training succeeded with at least one episode.
        """
        from causal_harness.pipeline.extractor import TrainingDataExtractor

        extractor = TrainingDataExtractor()
        episodes, outcomes = extractor.extract(self.ledger, self.encoder)

        # Also include in-memory episodes that may not be in the ledger yet
        if self._training_episodes:
            episodes = list(episodes) + list(self._training_episodes)
            outcomes = list(outcomes) + list(self._training_outcomes)

        if not episodes:
            return False

        try:
            self.risk_model.train(episodes, outcomes)
            self._trained = True
            return True
        except Exception:
            return False

    def incremental_train(
        self, episode_symbols: list[str], outcome: int,
    ) -> None:
        """Add one episode's data and retrain if enough data exists.

        Args:
            episode_symbols: Symbol sequence for one episode.
            outcome: 0 = success, 1 = failure.
        """
        self._training_episodes.append(list(episode_symbols))
        self._training_outcomes.append(outcome)

        # Retrain when we have enough data (at least 3 episodes)
        if len(self._training_episodes) >= 3:
            all_eps = list(self._training_episodes)
            all_out = list(self._training_outcomes)
            try:
                self.risk_model.train(all_eps, all_out)
                self._trained = True
            except Exception:
                pass

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def symbol_count(self) -> int:
        return len(self._symbol_sequence)

    # ── Summary ────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary dict of the pipeline state."""
        return {
            "total_symbols": len(self._symbol_sequence),
            "n_episodes_trained": len(self._training_episodes),
            "trained": self._trained,
            "metrics": self._latest_metrics,
            "loop_detected": self.loop_detector.is_looping(),
            "cssr_model_fitted": self._cssr_model is not None,
            "ledger_dir": str(self.ledger.ledger_dir),
        }


def _most_restrictive(interventions: list[Intervention]) -> Intervention:
    """Return the most restrictive intervention by priority."""
    priority = {
        PolicyResult.BLOCK: 0,
        PolicyResult.ASK_USER: 1,
        PolicyResult.VERIFY_BEFORE: 2,
        PolicyResult.INSPECT_FIRST: 3,
        PolicyResult.REPLAN: 4,
        PolicyResult.ALLOW: 5,
    }
    if not interventions:
        return Intervention(result=PolicyResult.ALLOW)
    return min(interventions, key=lambda iv: priority.get(iv.result, 99))
