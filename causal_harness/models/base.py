"""Common interfaces for risk prediction models.

All models share the same interface so they can be compared
in A/B experiments without changing the controller.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class RiskPrediction:
    """Output of a risk model for one action decision point.

    Attributes:
        p_failure: Probability of failure given current state and action.
        confidence: Model's confidence in its prediction (0–1).
        recommended_action: Suggested verification or sensing action.
        state_id: Which causal/risk state the model thinks we're in.
        reasoning: Human-readable explanation (for debugging).
    """
    p_failure: float
    confidence: float = 0.5
    recommended_action: str = ""
    state_id: int = 0
    reasoning: str = ""

    def __post_init__(self):
        self.p_failure = max(0.0, min(1.0, self.p_failure))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a risk predictor on held-out data.

    Attributes:
        brier_score: Mean squared error between predicted and actual.
        ece: Expected Calibration Error (binned).
        false_safe_rate: P(predict safe | actual failure).
        false_alarm_rate: P(predict failure | actual safe).
        n_samples: Number of predictions evaluated.
    """
    brier_score: float = 0.0
    ece: float = 0.0
    false_safe_rate: float = 0.0
    false_alarm_rate: float = 0.0
    n_samples: int = 0

    @classmethod
    def compute(
        cls,
        predicted_probs: list[float],
        actual_outcomes: list[int],
        n_bins: int = 10,
    ) -> "CalibrationMetrics":
        """Compute calibration metrics from predictions and binary outcomes.

        Args:
            predicted_probs: P(failure) for each decision point.
            actual_outcomes: 1 if failure occurred, 0 otherwise.
            n_bins: Number of bins for ECE computation.
        """
        preds = np.array(predicted_probs, dtype=np.float64)
        actual = np.array(actual_outcomes, dtype=np.float64)
        n = len(preds)

        if n == 0:
            return cls(n_samples=0)

        # Brier score
        brier = float(np.mean((preds - actual) ** 2))

        # ECE
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece_sum = 0.0
        for i in range(n_bins):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = float(preds[mask].mean())
            bin_acc = float(actual[mask].mean())
            ece_sum += (mask.sum() / n) * abs(bin_acc - bin_conf)
        ece = float(ece_sum)

        # False safe: model predicted P(failure) < 0.5 but actual failure occurred
        threshold = 0.5
        predicted_safe = preds < threshold
        actual_failure = actual == 1
        false_safe = float(np.sum(predicted_safe & actual_failure) / max(1, np.sum(predicted_safe)))
        false_alarm = float(np.sum(~predicted_safe & ~actual_failure) / max(1, np.sum(~predicted_safe)))

        return cls(
            brier_score=brier,
            ece=ece,
            false_safe_rate=false_safe,
            false_alarm_rate=false_alarm,
            n_samples=n,
        )


class RiskModel(ABC):
    """Abstract risk prediction model.

    Each implementation must provide train() and predict().
    The predict() method takes a state (symbol sequence context)
    and an action (next symbol) and returns a RiskPrediction.
    """

    @abstractmethod
    def train(self, episodes: list[list[str]], outcomes: list[int]) -> None:
        """Train the model on labeled episodes.

        Args:
            episodes: List of symbol sequences, one per episode.
            outcomes: 1 if the episode ended in failure, 0 if success.
        """
        ...

    @abstractmethod
    def predict(self, context: list[str], next_action: str) -> RiskPrediction:
        """Predict failure risk for taking `next_action` given `context`.

        Args:
            context: Recent symbol history (oldest first).
            next_action: The proposed next action symbol.
        """
        ...
