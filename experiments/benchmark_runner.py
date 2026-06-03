"""Graduate Benchmark Runner — validates Causal Mirror on hard problems.

Three-group controlled experiment:
  A (Full):  CSSR analysis + self-report + strategy injection
  B (Observe): CSSR analysis, metrics-only report
  C (Bare):  No enhancement

Supports: offline simulation + live ModelScope/OpenAI API calls.
"""

from __future__ import annotations

import json
import os
import re
import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np

from cssr import CSSR, CSSRConfig
from agent.behavior_encoder import BehaviorEncoder
from agent.prompts import SYSTEM_PROMPT_BASE


# ── Prompt templates ─────────────────────────────────────────

SYSTEM_PROMPT_GROUP_A = SYSTEM_PROMPT_BASE + """

## 自我认知反馈（基于前一轮行为分析）

{self_report}

## 策略建议
{strategy}

请在做题时特别注意上述反馈。"""

SYSTEM_PROMPT_GROUP_B = SYSTEM_PROMPT_BASE + """

## 行为度量报告

{metrics_report}

请继续努力，保持高质量的解题过程。"""


# ── Data classes ─────────────────────────────────────────────

@dataclass
class ProblemResult:
    """Result of solving one problem."""
    problem_idx: int
    domain: str
    difficulty: str
    predicted: float | None
    is_correct: bool
    steps_text: str
    full_response: str
    symbols: list[str] = field(default_factory=list)


@dataclass
class RoundResult:
    """Aggregate result for one round."""
    round_num: int
    accuracy: float
    n_correct: int
    n_total: int
    domain_accuracy: dict[str, float] = field(default_factory=dict)
    cssr_metrics: dict | None = None
    self_report: str = ""
    results: list[ProblemResult] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Complete experiment with three groups."""
    group: str
    config_label: str
    round_history: list[RoundResult] = field(default_factory=list)


# ── LLM Client ───────────────────────────────────────────────

def make_client(config: dict):
    """Create an OpenAI-compatible client from config."""
    provider = config.get("provider", "modelscope").lower()
    api_key_env = config.get("api_key_env", "MODELSCOPE_API_KEY")
    api_key = os.environ.get(api_key_env, "no-key")
    base_url = config.get("base_url", "https://api-inference.modelscope.cn/v1")

    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


# ── Benchmark Runner ─────────────────────────────────────────

class BenchmarkRunner:
    """Load problems, call LLM, parse responses, encode behavior."""

    def __init__(self, client, config: dict):
        self.client = client
        self.model = config.get("model", "deepseek-ai/DeepSeek-V3")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4096)
        self.encoder = BehaviorEncoder()

    def solve(self, problem: dict, system_prompt: str) -> ProblemResult:
        """Solve one problem using the LLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem["problem"]},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            full = response.choices[0].message.content or ""
        except Exception as e:
            return ProblemResult(
                problem_idx=-1, domain=problem.get("domain", ""),
                difficulty=problem.get("difficulty", ""),
                predicted=None, is_correct=False,
                steps_text=f"API ERROR: {e}", full_response=f"ERROR: {e}",
            )

        # Extract steps
        steps = self._extract_steps(full)

        # Extract answer
        answer = self._extract_answer(full)

        # Check correctness
        is_correct = self._check_answer(answer, problem)

        # Encode behavior
        symbols = self.encoder.encode(steps)

        return ProblemResult(
            problem_idx=-1,
            domain=problem.get("domain", ""),
            difficulty=problem.get("difficulty", ""),
            predicted=answer,
            is_correct=is_correct,
            steps_text=steps,
            full_response=full,
            symbols=symbols,
        )

    def solve_batch(
        self, problems: list[dict], system_prompt: str, verbose: bool = False,
    ) -> list[ProblemResult]:
        """Solve a batch of problems."""
        results = []
        for i, prob in enumerate(problems):
            if verbose:
                print(f"    [{i+1}/{len(problems)}] {prob['domain']} ", end="", flush=True)
            result = self.solve(prob, system_prompt)
            result.problem_idx = i
            results.append(result)
            if verbose:
                status = "OK" if result.is_correct else "ERR"
                print(f"{status} (acc={sum(1 for r in results if r.is_correct)}/{len(results)})")
        return results

    # ── Parsing ─────────────────────────────────────────────

    def _extract_steps(self, text: str) -> str:
        steps = re.findall(
            r"\[步骤\s*\d+\].*?(?=\[步骤\s*\d+\]|$)", text, re.DOTALL,
        )
        if steps:
            return "\n".join(s.strip() for s in steps)
        # Fallback: split by lines, filter non-answer lines
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        reasoning = [l for l in lines if not l.startswith("答案") and not l.startswith("最终")]
        return "\n".join(reasoning) if reasoning else text

    def _extract_answer(self, text: str):
        patterns = [
            r"答案[：:]\s*([\d,]+\.?\d*)",
            r"answer\s*(?:is|:)?\s*([\d,]+\.?\d*)",
            r"####\s*([\d,]+\.?\d*)",
        ]
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1].replace(",", ""))
                except ValueError:
                    continue
        # Last number fallback
        nums = re.findall(r"[\d,]+\.?\d*", text)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass
        return None

    def _check_answer(self, predicted, problem: dict) -> bool:
        if predicted is None:
            return False
        ground = problem["answer"]
        check_type = problem.get("check_type", "numeric")

        if check_type == "exact":
            pred_str = str(predicted).strip().lower()
            ground_str = str(ground).strip().lower()
            # Allow partial match for formula answers
            return pred_str == ground_str or ground_str in pred_str

        # Numeric check with tolerance
        tol = problem.get("tolerance", 0.01)
        try:
            return abs(float(predicted) - float(ground)) <= tol + 1e-10
        except (ValueError, TypeError):
            return False


# ── Offline Simulator ────────────────────────────────────────

class OfflineSimulator:
    """Simulate LLM responses for testing the pipeline without API.

    Generates realistic-looking step-by-step reasoning with
    domain-appropriate symbols. Difficulty affects correctness probability.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.encoder = BehaviorEncoder()

    def simulate_batch(
        self, problems: list[dict], baseline_acc: float = 0.60,
    ) -> list[ProblemResult]:
        """Generate simulated results for a batch of problems."""
        results = []
        for i, prob in enumerate(problems):
            is_correct = self.rng.random() < baseline_acc

            # Generate domain-specific reasoning trace
            domain = prob.get("domain", "math")
            if domain == "math":
                trace = self._math_trace(is_correct)
            elif domain == "algo":
                trace = self._algo_trace(is_correct)
            else:
                trace = self._logic_trace(is_correct)

            symbols = self.encoder.encode(trace)

            results.append(ProblemResult(
                problem_idx=i,
                domain=domain,
                difficulty=prob.get("difficulty", "hard"),
                predicted=(
                    prob["answer"] if is_correct
                    else self._wrong_answer(prob["answer"])
                ),
                is_correct=is_correct,
                steps_text=trace,
                full_response=trace,
                symbols=symbols,
            ))
        return results

    def _math_trace(self, correct: bool) -> str:
        if correct:
            return (
                "[步骤 1] [动作: 推理] 仔细分析题目条件，明确已知和所求。\n"
                "[步骤 2] [动作: 计算] 根据公式进行数学推导和计算。\n"
                "[步骤 3] [动作: 验证] 检查计算过程，确认每个步骤的正确性。\n"
                "[步骤 4] [动作: 推理] 得出结论，答案与预期一致。"
            )
        return (
            "[步骤 1] [动作: 推理] 快速浏览题目，开始推导。\n"
            "[步骤 2] [动作: 计算] 进行计算但有一个步骤跳过了验证。\n"
            "[步骤 3] [动作: 推理] 得出初步答案，未充分验证。"
        )

    def _algo_trace(self, correct: bool) -> str:
        if correct:
            return (
                "[步骤 1] [动作: 推理] 分析问题类型，选择合适的数据结构和算法。\n"
                "[步骤 2] [动作: 计算] 逐步执行算法，注意边界条件。\n"
                "[步骤 3] [动作: 验证] 用小规模测试用例验证结果。\n"
                "[步骤 4] [动作: 推理] 确认答案正确。"
            )
        return (
            "[步骤 1] [动作: 推理] 识别问题类型，但可能遗漏了关键约束。\n"
            "[步骤 2] [动作: 计算] 执行计算，有一个步骤的推理不够严谨。\n"
            "[步骤 3] [动作: 推理] 得出答案，未充分检查边界情况。"
        )

    def _wrong_answer(self, answer):
        """Generate a wrong answer that's close-ish to the correct one."""
        if isinstance(answer, (int, float)):
            return float(answer) + self.rng.uniform(-5, 5)
        return str(answer) + " (wrong)"

    def _logic_trace(self, correct: bool) -> str:
        if correct:
            return (
                "[步骤 1] [动作: 推理] 分析逻辑结构，列出所有可能情况。\n"
                "[步骤 2] [动作: 计算] 系统性地计算每种情况。\n"
                "[步骤 3] [动作: 验证] 用容斥原理/对偶验证确认。\n"
                "[步骤 4] [动作: 推理] 结论正确。"
            )
        return (
            "[步骤 1] [动作: 推理] 尝试枚举，但可能遗漏了一些情况。\n"
            "[步骤 2] [动作: 计算] 计算过程中有个分情况讨论不够完整。\n"
            "[步骤 3] [动作: 推理] 答案可能有误。"
        )


# ── Experiment Harness ───────────────────────────────────────

class ExperimentHarness:
    """Run three-group controlled experiments."""

    def __init__(self, config: dict):
        self.config = config
        self.experiment_cfg = config.get("experiment", config)
        self.llm_cfg = config.get("llm", {})
        self.cssr_cfg_dict = config.get("cssr", {})
        self.output_dir = config.get("output", {}).get(
            "results_dir", "experiments/results",
        )

    def run(self, problems: list[dict]) -> dict[str, list[RoundResult]]:
        """Run the full three-group experiment.

        Returns: {"A": [RoundResult, ...], "B": [...], "C": [...]}
        """
        rounds = self.experiment_cfg.get("rounds", 4)
        groups = self.experiment_cfg.get("groups", ["A", "B", "C"])
        offline = self.experiment_cfg.get("offline", False)

        print(f"\n{'='*60}")
        print(f"  研究生难度基准测试实验")
        print(f"  题目数: {len(problems)}, 轮数: {rounds}")
        print(f"  组别: A=Full管线 B=仅观察 C=裸Agent")
        print(f"{'='*60}")

        all_results: dict[str, list[RoundResult]] = defaultdict(list)

        for group in groups:
            print(f"\n─── 组 {group} ───")

            if group == "C":
                # Bare agent — same prompt every round
                results = self._run_bare(problems, rounds, offline)
            elif group == "B":
                results = self._run_observe(problems, rounds, offline)
            else:  # A
                results = self._run_full(problems, rounds, offline)

            all_results[group] = results

        return dict(all_results)

    def _run_bare(
        self, problems: list[dict], rounds: int, offline: bool,
    ) -> list[RoundResult]:
        """Group C: No enhancement."""
        history = []
        if offline:
            sim = OfflineSimulator(seed=0)

        for r in range(rounds):
            print(f"  Round {r+1}/{rounds}: ", end="", flush=True)
            if offline:
                results = sim.simulate_batch(problems)
            else:
                client = make_client(self.llm_cfg)
                runner = BenchmarkRunner(client, self.llm_cfg)
                results = runner.solve_batch(problems, SYSTEM_PROMPT_BASE)

            correct = sum(1 for r in results if r.is_correct)
            acc = correct / len(results) if results else 0
            domain_acc = self._domain_accuracy(results)
            print(f"准确率={acc:.1%} ({correct}/{len(results)})")
            history.append(RoundResult(
                round_num=r, accuracy=acc,
                n_correct=correct, n_total=len(results),
                domain_accuracy=domain_acc, results=results,
            ))
        return history

    def _run_observe(
        self, problems: list[dict], rounds: int, offline: bool,
    ) -> list[RoundResult]:
        """Group B: CSSR analysis, metrics-only report, no intervention."""
        history = []
        inject_text = ""
        cssr_config = CSSRConfig(
            L_max=self.cssr_cfg_dict.get("L_max"),
            alpha=self.cssr_cfg_dict.get("alpha", 0.05),
            test=self.cssr_cfg_dict.get("test", "chi2"),
            correction=self.cssr_cfg_dict.get("correction", "bonferroni"),
            min_count=self.cssr_cfg_dict.get("min_count", 2),
        )
        if offline:
            sim = OfflineSimulator(seed=1)

        for r in range(rounds):
            print(f"  Round {r+1}/{rounds}: ", end="", flush=True)

            # Solve
            sys_prompt = SYSTEM_PROMPT_BASE
            if inject_text:
                sys_prompt = SYSTEM_PROMPT_GROUP_B.format(metrics_report=inject_text)

            if offline:
                results = sim.simulate_batch(problems)
            else:
                client = make_client(self.llm_cfg)
                runner = BenchmarkRunner(client, self.llm_cfg)
                results = runner.solve_batch(problems, sys_prompt)

            correct = sum(1 for r in results if r.is_correct)
            acc = correct / len(results) if results else 0

            # CSSR analysis
            cssr_metrics = None
            if len(results) >= 3:
                all_syms = []
                for res in results:
                    all_syms.extend(res.symbols)
                if len(all_syms) >= 10:
                    try:
                        cssr = CSSR(cssr_config)
                        cssr.fit(all_syms)
                        cssr_metrics = dict(cssr.metrics)
                        inject_text = self._metrics_only_report(cssr_metrics)
                    except Exception:
                        pass

            print(f"准确率={acc:.1%} ({correct}/{len(results)}) ", end="")
            if cssr_metrics:
                print(f"C_mu={cssr_metrics['statistical_complexity']:.3f}")
            else:
                print()

            history.append(RoundResult(
                round_num=r, accuracy=acc,
                n_correct=correct, n_total=len(results),
                cssr_metrics=cssr_metrics,
                results=results,
            ))
        return history

    def _run_full(
        self, problems: list[dict], rounds: int, offline: bool,
    ) -> list[RoundResult]:
        """Group A: Full pipeline — CSSR + risk model + strategy intervention."""
        history = []
        inject_text = ""
        all_traces_so_far: list[str] = []
        cssr_config = CSSRConfig(
            L_max=self.cssr_cfg_dict.get("L_max"),
            alpha=self.cssr_cfg_dict.get("alpha", 0.05),
            test=self.cssr_cfg_dict.get("test", "chi2"),
            correction=self.cssr_cfg_dict.get("correction", "bonferroni"),
            min_count=self.cssr_cfg_dict.get("min_count", 2),
        )
        if offline:
            sim = OfflineSimulator(seed=2)

        for r in range(rounds):
            print(f"  Round {r+1}/{rounds}: ", end="", flush=True)

            # Build strategy from prior analysis
            strategy_text = self._build_strategy(history)
            sys_prompt = SYSTEM_PROMPT_BASE
            if inject_text:
                sys_prompt = SYSTEM_PROMPT_GROUP_A.format(
                    self_report=inject_text,
                    strategy=strategy_text or "运用所学解题策略，注意验证每一步。",
                )

            # Solve
            if offline:
                results = sim.simulate_batch(problems)
            else:
                client = make_client(self.llm_cfg)
                runner = BenchmarkRunner(client, self.llm_cfg)
                results = runner.solve_batch(problems, sys_prompt)

            correct = sum(1 for r in results if r.is_correct)
            acc = correct / len(results) if results else 0

            # Accumulate traces
            for res in results:
                all_traces_so_far.append(res.steps_text)

            # CSSR analysis
            cssr_metrics = None
            if all_traces_so_far:
                encoder = BehaviorEncoder()
                all_syms = []
                for trace in all_traces_so_far[-200:]:  # recent traces
                    syms = encoder.encode(trace)
                    all_syms.extend(syms)
                if len(all_syms) >= 10:
                    try:
                        cssr = CSSR(cssr_config)
                        cssr.fit(all_syms)
                        cssr_metrics = dict(cssr.metrics)
                        inject_text = self._generate_self_report(
                            cssr.metrics, cssr.machine, history, results,
                        )
                    except Exception:
                        pass

            domain_acc = self._domain_accuracy(results)
            print(f"准确率={acc:.1%} ({correct}/{len(results)}) ", end="")
            if cssr_metrics:
                print(f"C_mu={cssr_metrics['statistical_complexity']:.3f} "
                      f"states={cssr_metrics['n_states']}")
            else:
                print()

            history.append(RoundResult(
                round_num=r, accuracy=acc,
                n_correct=correct, n_total=len(results),
                domain_accuracy=domain_acc,
                cssr_metrics=cssr_metrics,
                self_report=inject_text,
                results=results,
            ))
        return history

    # ── Helpers ──────────────────────────────────────────────

    def _build_strategy(self, history: list[RoundResult]) -> str:
        """Build strategic recommendations from CSSR analysis."""
        if not history:
            return ""

        last = history[-1]
        if last.cssr_metrics is None:
            return "仔细推理每一步，不要跳过验证。"

        m = last.cssr_metrics
        strategies = []

        if m["statistical_complexity"] < 0.2:
            strategies.append("策略单一，尝试多样化推理路径")
        elif m["statistical_complexity"] > 1.5:
            strategies.append("行为复杂但可能碎片化，保持策略一致性")

        if m["entropy_rate"] > 1.5:
            strategies.append("每步不确定性高，增加中间验证")
        else:
            strategies.append("推理稳定，继续保持")

        # Analyze error states
        for result in last.results:
            if not result.is_correct:
                strategies.append(
                    f"在 {result.domain} 类问题上需要更仔细的验证步骤"
                )
                break

        return "\n".join(f"- {s}" for s in strategies[:4])

    def _generate_self_report(self, machine, metrics, history, recent):
        """Generate natural-language self-report from CSSR analysis."""
        n_correct = sum(1 for r in recent if r.is_correct)
        n_total = len(recent)

        report = (
            f"## 行为度量 (Round {len(history)+1})\n\n"
            f"- 统计复杂度 C_μ: {metrics.statistical_complexity():.4f} bits\n"
            f"- 熵率 h_μ: {metrics.entropy_rate():.4f} bits\n"
            f"- 超熵 E: {metrics.excess_entropy():.4f} bits\n"
            f"- 因果状态数: {len(machine.states)}\n"
            f"- 预测信息 χ: {metrics.predictive_information():.4f} bits\n\n"
            f"## 本轮表现\n\n"
            f"- 准确率: {n_correct}/{n_total} = {n_correct/n_total:.1%}\n"
        )

        # Error state analysis
        error_states = []
        for s in machine.states:
            error_prob = 0.0
            for sym, prob in s.emission_probs.items():
                if "_INCORRECT" in str(sym) or "_PARTIAL" in str(sym):
                    error_prob += prob
            if error_prob > 0.4:
                error_states.append(f"状态{s.state_id}中错误率高达{error_prob:.1%}")

        if error_states:
            report += "\n⚠️ 高风险状态:\n"
            for es in error_states:
                report += f"- {es}\n"

        # Strategy
        report += "\n## 改进建议\n\n"
        if metrics.statistical_complexity() < 0.3:
            report += "- 行为模式过于简单，尝试更严谨的验证策略\n"
        if n_correct / n_total < 0.6:
            report += "- 准确率偏低，在计算步骤后增加验证环节\n"
        if metrics.entropy_rate() > 1.5:
            report += "- 每步推理的不确定性高，放慢速度仔细检查\n"
        report += "- 在得出结论前，至少用一种替代方法验证\n"

        return report

    def _metrics_only_report(self, metrics: dict) -> str:
        """Generate a metrics-only report (Group B)."""
        return (
            f"C_μ={metrics['statistical_complexity']:.4f}, "
            f"h_μ={metrics['entropy_rate']:.4f}, "
            f"E={metrics['excess_entropy']:.4f}, "
            f"n_states={metrics['n_states']}"
        )

    def _domain_accuracy(self, results: list[ProblemResult]) -> dict[str, float]:
        by_domain: dict[str, list[float]] = defaultdict(list)
        for r in results:
            by_domain[r.domain].append(1.0 if r.is_correct else 0.0)
        return {d: sum(v)/len(v) for d, v in by_domain.items()}


# ── Comparison & Reporting ───────────────────────────────────

def compare_groups(
    group_results: dict[str, list[RoundResult]],
) -> dict:
    """Compare three groups across all rounds."""
    comparison = {}

    for group, history in group_results.items():
        acc_traj = [h.accuracy for h in history if h.n_total > 0]
        if acc_traj:
            comparison[group] = {
                "accuracy_trajectory": acc_traj,
                "initial_accuracy": acc_traj[0],
                "final_accuracy": acc_traj[-1],
                "improvement": acc_traj[-1] - acc_traj[0],
                "max_accuracy": max(acc_traj),
                "cssr_metrics": [
                    h.cssr_metrics for h in history if h.cssr_metrics
                ],
            }

    return comparison


def print_comparison(comparison: dict) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*70}")
    print(f"  实验结果对比")
    print(f"{'='*70}")

    header = f"{'指标':<24}"
    for g in sorted(comparison.keys()):
        header += f"{'组 '+g:>14}"
    print(header)
    print("-" * 70)

    metrics = [
        ("初始准确率", "initial_accuracy"),
        ("最终准确率", "final_accuracy"),
        ("提升幅度", "improvement"),
        ("最高准确率", "max_accuracy"),
    ]
    for label, key in metrics:
        row = f"{label:<24}"
        for g in sorted(comparison.keys()):
            row += f"{comparison[g][key]:>14.1%}"
        print(row)

    # CSSR trajectory
    print(f"\n{'─'*70}")
    print("  C_mu 指标轨迹")
    for g in sorted(comparison.keys()):
        c_mu_traj = [
            m["statistical_complexity"]
            for m in comparison[g]["cssr_metrics"]
            if m
        ]
        if c_mu_traj:
            traj_str = " → ".join(f"{v:.3f}" for v in c_mu_traj)
            print(f"  组 {g}: {traj_str}")


def save_results(
    group_results: dict[str, list[RoundResult]],
    output_dir: str,
) -> str:
    """Save experiment results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    output = {}
    for group, history in group_results.items():
        rounds_out = []
        for hr in history:
            rounds_out.append({
                "round": hr.round_num,
                "accuracy": hr.accuracy,
                "n_correct": hr.n_correct,
                "n_total": hr.n_total,
                "domain_accuracy": hr.domain_accuracy,
                "cssr_metrics": hr.cssr_metrics,
            })
        output[group] = rounds_out

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {path}")
    return path
