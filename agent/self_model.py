"""自我模型管理器：连接CSSR引擎、行为编码器和自我报告生成器。"""

import time
from cssr import CSSR, CSSRConfig
from agent.behavior_encoder import BehaviorEncoder
from agent.self_report import SelfReportGenerator


class SelfModel:
    """Manages the self-modeling pipeline for a Causal Mirror Agent.

    Ties together:
      1. Behavior encoding (traces -> symbol sequences)
      2. CSSR (symbol sequences -> epsilon-machine + metrics)
      3. Self-report generation (epsilon-machine -> natural language report)
    """

    def __init__(self, encoder: BehaviorEncoder, report_gen: SelfReportGenerator,
                 cssr_config: CSSRConfig | None = None):
        self.encoder = encoder
        self.report_gen = report_gen
        self.cssr_config = cssr_config or CSSRConfig(L_max=None, alpha=0.05, min_count=3)
        self.history: list[dict] = []

    def analyze_round(self, traces: list[str], round_num: int) -> dict:
        """Analyze a round of behavior traces and produce a self-model.

        Args:
            traces: List of reasoning step texts from the agent.
            round_num: Current round number.

        Returns:
            dict with keys: symbols, machine, metrics, report, timestamp.
        """
        # Encode all traces to symbol sequences
        all_symbols = []
        for trace in traces:
            symbols = self.encoder.encode(trace)
            all_symbols.extend(symbols)

        # Run CSSR if enough data
        machine = None
        metrics = None
        report = None

        if len(all_symbols) >= 10:  # Minimum for CSSR
            try:
                cssr = CSSR(self.cssr_config)
                cssr.fit(all_symbols)
                machine = cssr.machine
                metrics = cssr.metrics
            except Exception as e:
                report = f"[CSSR分析失败: {e}]"
        else:
            report = "[数据不足以运行CSSR分析]"

        # Generate report if CSSR succeeded
        if machine is not None and metrics is not None:
            try:
                report = self.report_gen.generate(metrics, machine)
            except Exception as e:
                report = self._basic_report(metrics) if metrics else f"[报告生成失败: {e}]"

        result = {
            "round": round_num,
            "n_symbols": len(all_symbols),
            "symbols": all_symbols,
            "machine": machine,
            "metrics": metrics,
            "report": report,
            "timestamp": time.time(),
        }
        self.history.append(result)
        return result

    def _basic_report(self, metrics: dict) -> str:
        """Minimal report without LLM."""
        return (
            f"# 自我认知 — 度量摘要\n"
            f"C_μ = {metrics['statistical_complexity']:.4f} bits, "
            f"E = {metrics['excess_entropy']:.4f} bits, "
            f"h_μ = {metrics['entropy_rate']:.4f} bits, "
            f"状态数 = {metrics['n_states']}"
        )

    def get_report(self) -> str | None:
        """Get the most recent self-report."""
        if self.history:
            return self.history[-1].get("report")
        return None

    def get_metrics_history(self) -> list[dict]:
        """Get the time series of metrics across rounds."""
        return [
            {
                "round": h["round"],
                **{k: v for k, v in (h.get("metrics") or {}).items()},
            }
            for h in self.history if h.get("metrics")
        ]
