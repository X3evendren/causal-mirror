"""Tests for risk prediction models."""
import pytest
from causal_harness.models.base import RiskPrediction, CalibrationMetrics
from causal_harness.models.markov_baseline import MarkovRiskModel


class TestRiskPrediction:
    def test_clamps_probabilities(self):
        rp = RiskPrediction(p_failure=1.5, confidence=2.0)
        assert rp.p_failure == 1.0
        assert rp.confidence == 1.0

        rp = RiskPrediction(p_failure=-0.5, confidence=-0.1)
        assert rp.p_failure == 0.0
        assert rp.confidence == 0.0

    def test_default_recommended_action_empty(self):
        rp = RiskPrediction(p_failure=0.3)
        assert rp.recommended_action == ""


class TestCalibrationMetrics:
    def test_perfect_calibration(self):
        # Perfect predictions
        preds = [0.1, 0.2, 0.8, 0.9]
        actual = [0, 0, 1, 1]
        m = CalibrationMetrics.compute(preds, actual, n_bins=2)
        assert m.brier_score < 0.05
        assert m.ece < 0.2

    def test_totally_wrong(self):
        preds = [0.9, 0.9, 0.1, 0.1]
        actual = [0, 0, 1, 1]
        m = CalibrationMetrics.compute(preds, actual)
        assert m.brier_score > 0.5

    def test_empty_input(self):
        m = CalibrationMetrics.compute([], [])
        assert m.n_samples == 0

    def test_false_safe_rate(self):
        # Model says safe, but it fails
        preds = [0.1, 0.2, 0.3, 0.4]
        actual = [0, 1, 0, 1]
        m = CalibrationMetrics.compute(preds, actual)
        assert m.false_safe_rate > 0  # some "safe" predictions were failures


class TestMarkovRiskModel:
    def test_train_and_predict(self):
        model = MarkovRiskModel(order=1, min_count=1)
        # Episode 1: OBSERVE -> EDIT -> VERIFY success
        ep1 = ["OBSERVE", "EDIT_FILE", "RUN_TEST", "VERIFY"]
        # Episode 2: OBSERVE -> EDIT -> ERROR failure
        ep2 = ["OBSERVE", "EDIT_FILE", "RUN_TEST", "ERROR"]

        model.train([ep1, ep2], [0, 1])

        # (OBSERVE, EDIT_FILE) appears in both, 1 failure / 2 = 0.5
        rp = model.predict(["OBSERVE"], "EDIT_FILE")
        assert rp.p_failure == 0.5

    def test_unseen_context_falls_back(self):
        model = MarkovRiskModel(order=2)
        model.train([["OBSERVE", "EDIT_FILE"]], [0])

        rp = model.predict(["COMPLETELY", "UNSEEN"], "CONTEXT")
        assert rp.confidence == 0.1
        assert rp.p_failure == 0.0  # global rate from training

    def test_context_trimming(self):
        model = MarkovRiskModel(order=3, min_count=1)
        ep1 = ["A", "B", "C", "D"]
        model.train([ep1], [1])

        # Context too long → should trim
        rp = model.predict(["X", "Y", "Z", "A", "B", "C"], "D")
        assert rp.p_failure > 0

    def test_untrained_model(self):
        model = MarkovRiskModel()
        rp = model.predict(["A"], "B")
        assert rp.p_failure == 0.5  # default
        assert rp.confidence == 0.1
