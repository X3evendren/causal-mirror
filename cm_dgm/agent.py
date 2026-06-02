"""CMDGMAgent: 可变策略提示词基因组 + 评估流水线。

每个Agent持有一段策略文本(genome)，注入到TaskSolver的系统提示词中。
评估时：解题 → 编码行为 → CSSR提取ε-machine → 返回准确率+指标。
"""

from cssr import CSSR, CSSRConfig
from agent.task_solver import TaskSolver
from agent.behavior_encoder import BehaviorEncoder
from agent.prompts import SYSTEM_PROMPT_BASE

GENOME_MAX_LEN = 200


class CMDGMAgent:
    """可进化的Agent，基因组为策略提示词文本。"""

    def __init__(self, agent_id: int, genome: str = ""):
        self.agent_id = agent_id
        self.genome = genome[:GENOME_MAX_LEN] if genome else ""
        self.parent_id = None
        self.generation = 0

        # 最近一次评估的结果
        self.accuracy = 0.0
        self.n_correct = 0
        self.n_total = 0
        self.traces: list[str] = []
        self.symbols: list[str] = []
        self.metrics: dict | None = None

    def evaluate(
        self,
        problems: list[dict],
        solver: TaskSolver,
        encoder: BehaviorEncoder,
        cssr_config: CSSRConfig,
    ) -> dict:
        """评估Agent在一组问题上的表现。

        对每个问题：解题 → 记录trace → 判分。
        然后：编码所有trace → CSSR拟合 → 存储指标。
        """
        self.traces = []
        self.n_correct = 0
        self.n_total = len(problems)

        for task in problems:
            result = solver.solve(task["problem"], inject_text=self.genome)
            is_correct = (
                result.get("final_answer") is not None
                and abs(result.get("final_answer", 0) - task["answer"]) < 0.01
            )
            if is_correct:
                self.n_correct += 1
            if result.get("steps_text"):
                self.traces.append(result["steps_text"])

        self.accuracy = self.n_correct / self.n_total if self.n_total > 0 else 0.0

        # 行为编码
        all_symbols = []
        for trace in self.traces:
            symbols = encoder.encode(trace)
            all_symbols.extend(symbols)

        self.symbols = all_symbols

        # CSSR分析
        if len(all_symbols) >= 10:
            try:
                cssr = CSSR(cssr_config)
                cssr.fit(all_symbols)
                self.metrics = dict(cssr.metrics)
            except Exception:
                self.metrics = self._default_metrics()
        else:
            self.metrics = self._default_metrics()

        return {
            "agent_id": self.agent_id,
            "accuracy": self.accuracy,
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "metrics": self.metrics,
            "n_symbols": len(all_symbols),
            "alphabet_used": sorted(set(all_symbols)),
        }

    def _default_metrics(self) -> dict:
        return {
            "statistical_complexity": 0.0,
            "entropy_rate": 0.0,
            "excess_entropy": 0.0,
            "predictive_information": 0.0,
            "n_states": 0,
            "alphabet_size": 0,
        }

    def snapshot(self) -> dict:
        """序列化为可存档的字典。"""
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "genome": self.genome,
            "accuracy": self.accuracy,
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "metrics": self.metrics,
            "n_symbols": len(self.symbols),
            "alphabet_used": sorted(set(self.symbols)),
        }

    def __repr__(self):
        return (
            f"CMDGMAgent(id={self.agent_id}, gen={self.generation}, "
            f"acc={self.accuracy:.2%}, genome='{self.genome[:30]}...')"
        )
