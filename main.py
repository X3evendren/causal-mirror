#!/usr/bin/env python3
"""
Causal Mirror — 自指性Agent实验主入口。

三组对照：
  A: 自建模Agent（CSSR自我认知 → 自我报告 → 注入提示词）
  B: 通用反馈Agent（固定改进建议注入）
  C: 裸Agent（无任何增强）

用法:
  python main.py --rounds 3 --tasks-per-round 10
  python main.py --config experiments/config.yaml
  python main.py --offline  # 使用内置模拟数据，无需API
"""

import argparse
import json
import os
import sys
import time
import yaml
from datetime import datetime

from cssr import CSSRConfig
from agent.task_solver import TaskSolver
from agent.behavior_encoder import BehaviorEncoder
from agent.self_model import SelfModel
from agent.self_report import SelfReportGenerator
from agent.prompts import SYSTEM_PROMPT_BASE, GENERIC_FEEDBACK_PROMPT
from experiments.hard_problems import HARD_TASKS


# ─── 内置示例数学题（简单题，用于测试） ───

SAMPLE_TASKS = [
    {
        "problem": "小明有15个苹果，他给了小红3个，又买了8个。他现在有多少个苹果？",
        "answer": 20,
    },
    {
        "problem": "一辆车以每小时60公里的速度行驶了2.5小时，然后以每小时80公里的速度行驶了1.5小时。总行驶距离是多少？",
        "answer": 270,
    },
    {
        "problem": "一个长方形的长是12米，宽是8米。它的面积是多少平方米？",
        "answer": 96,
    },
    {
        "problem": "张三有120元，他花了5分之2买书，又花了剩余钱的3分之1买文具。他还剩多少钱？",
        "answer": 48,
    },
    {
        "problem": "一箱苹果重24公斤，其中8分之3是坏掉的。好的苹果有多少公斤？",
        "answer": 15,
    },
    {
        "problem": "李四每天跑步45分钟，一周跑5天。他一周总共跑步多少小时？",
        "answer": 3.75,
    },
    {
        "problem": "一个三角形底边长10厘米，高6厘米。它的面积是多少平方厘米？",
        "answer": 30,
    },
    {
        "problem": "商店里一件衣服原价200元，打八折后又打九折。最终价格是多少？",
        "answer": 144,
    },
    {
        "problem": "如果3台打印机5分钟打印45页，那么8台打印机10分钟打印多少页？",
        "answer": 240,
    },
    {
        "problem": "王五的年龄是赵六年龄的3倍减去5岁。如果赵六10岁，王五多少岁？",
        "answer": 25,
    },
    {
        "problem": "一个水池有两个进水管。单独开第一个管3小时注满，单独开第二个管6小时注满。两个管同时开，多少小时注满？",
        "answer": 2,
    },
    {
        "problem": "一块地的5分之2种小麦，3分之1种玉米，剩下的种蔬菜。蔬菜占这块地的几分之几？",
        "answer": 0.2667,
    },
    {
        "problem": "小明从家到学校距离3公里。他走路速度是每小时4公里。他需要多少分钟到学校？",
        "answer": 45,
    },
    {
        "problem": "一个数加上它的20%等于60。这个数是多少？",
        "answer": 50,
    },
    {
        "problem": "一班有40名学生，二班有35名。一班平均分85分，二班平均分90分。两班总平均分是多少？",
        "answer": 87.33,
    },
    {
        "problem": "一个圆柱的底面半径是5厘米，高是10厘米。它的体积是多少立方厘米？（π取3.14）",
        "answer": 785,
    },
    {
        "problem": "小红比小明大3岁，5年后两人年龄之和是45岁。小红现在多少岁？",
        "answer": 19,
    },
    {
        "problem": "一本书已读页数与未读页数的比是3:5。如果已读60页，这本书共有多少页？",
        "answer": 160,
    },
    {
        "problem": "浓度为20%的盐水300克，需要加多少克盐才能使浓度变为50%？",
        "answer": 180,
    },
    {
        "problem": "A和B两个工程队合作，6天完成一项工程。如果A单独做需要10天，B单独做需要多少天？",
        "answer": 15,
    },
]


def load_tasks(config: dict, use_hard: bool = False) -> list[dict]:
    """Load task set from config or use built-in samples."""
    n = config.get("tasks_per_round", 10)
    available = HARD_TASKS if use_hard else SAMPLE_TASKS
    # Repeat if we need more tasks than available
    tasks = []
    for i in range(n):
        idx = i % len(available)
        tasks.append(dict(available[idx]))
    return tasks


def setup_llm_client(config: dict):
    """Set up LLM client based on config.

    Supports: openai, modelscope (魔搭), anthropic.
    ModelScope uses the OpenAI-compatible endpoint at api.modelscope.cn/v1.
    """
    provider = config.get("provider", "openai").lower()
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)

    if not api_key:
        print(f"[WARN] No API key found (env: {api_key_env}). Running in offline/mock mode.")
        return None

    if provider in ("openai", "modelscope"):
        from openai import OpenAI
        base_url = config.get("base_url", None)
        return OpenAI(api_key=api_key, base_url=base_url)
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def setup_agent(client, llm_config: dict, inject_text: str = ""):
    """Create a task solver agent."""
    if client is None:
        return None

    sys_prompt = SYSTEM_PROMPT_BASE
    if inject_text:
        sys_prompt = sys_prompt + "\n\n" + inject_text

    return TaskSolver(
        client=client,
        model=llm_config.get("model", "gpt-4o"),
        system_prompt=sys_prompt,
        temperature=llm_config.get("temperature", 0.3),
        max_tokens=llm_config.get("max_tokens", 2048),
    )


def check_answer(predicted, ground_truth, tolerance: float = 0.01) -> bool:
    """Check if predicted answer matches ground truth within tolerance."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(ground_truth)) < tolerance
    except (ValueError, TypeError):
        return False


def run_round(
    tasks: list[dict],
    solver,
    group_name: str,
    round_num: int,
    save_traces: bool = True,
) -> dict:
    """Run one round of problem solving.

    Returns:
        dict with: group, round, accuracy, n_correct, n_total, traces, results.
    """
    results = []
    traces = []
    n_correct = 0

    for i, task in enumerate(tasks):
        result = solver.solve(task["problem"])
        result["ground_truth"] = task["answer"]
        result["is_correct"] = check_answer(result["final_answer"], task["answer"])
        result["task_id"] = i

        if result["is_correct"]:
            n_correct += 1

        results.append(result)
        traces.append(result["steps_text"])

    accuracy = n_correct / len(tasks) if tasks else 0.0

    return {
        "group": group_name,
        "round": round_num,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(tasks),
        "traces": traces,
        "results": results,
        "timestamp": time.time(),
    }


def mock_solve(problem: str, inject_text: str = "") -> dict:
    """Mock solver for offline testing (no API)."""
    import random
    random.seed(hash(problem) % 2**32)
    # Simulate a reasoning trace with steps
    steps = [
        "[步骤 1] [推理] 理解问题...\n",
        "[步骤 2] [计算] 进行计算...\n",
        "[步骤 3] [验证] 检查结果...\n",
    ]
    trace = "".join(steps)
    return {
        "problem": problem,
        "full_response": trace + f"答案: {random.randint(1, 100)}",
        "final_answer": random.randint(1, 100),
        "is_correct": None,
        "steps_text": trace,
    }


def run_offline_experiment(config: dict, use_hard: bool = False) -> dict:
    """Run experiment in offline mode (no API, simulated traces).

    This mode demonstrates the CSSR pipeline with synthetic behavior data,
    useful for testing and debugging without LLM API costs.
    """
    from cssr import CSSR
    from cssr.visualization import complexity_entropy_label

    print("=" * 60)
    print("Causal Mirror — 离线演示模式")
    print("=" * 60)

    cssr_config = CSSRConfig(
        L_max=config.get("L_max"),
        alpha=config.get("alpha", 0.05),
        test=config.get("test", "chi2"),
        correction=config.get("correction", "bonferroni"),
        min_count=config.get("min_count", 3),
    )

    rounds = config.get("rounds", 3)
    n_tasks = config.get("tasks_per_round", 10)

    # Generate synthetic behavior data for demonstration
    # Simulate an agent that improves over rounds
    all_metrics = []

    for r in range(rounds):
        tasks = load_tasks(config, use_hard)
        traces = []
        for t in tasks:
            result = mock_solve(t["problem"])
            # Simulate improving behavior: fewer incorrect steps in later rounds
            if r == 0:
                trace = "INFER_CORRECT CALC_INCORRECT VERIFY_SKIPPED INFER_CORRECT CALC_CORRECT"
            elif r == 1:
                trace = "INFER_CORRECT CALC_PARTIAL VERIFY_CORRECT INFER_CORRECT CALC_CORRECT"
            else:
                trace = "INFER_CORRECT CALC_CORRECT VERIFY_CORRECT INFER_CORRECT CALC_CORRECT VERIFY_CORRECT"
            traces.append(trace)

        # Encode traces to symbol sequences
        all_symbols = []
        for trace in traces:
            symbols = trace.split()
            all_symbols.extend(symbols)

        # Run CSSR
        if len(all_symbols) >= 10:
            cssr = CSSR(cssr_config)
            cssr.fit(all_symbols)
            metrics = cssr.metrics
            label = complexity_entropy_label(cssr.machine)

            print(f"\nRound {r}: {label}")
            print(f"  DOT:\n{cssr.to_dot()[:500]}...")
            all_metrics.append({"round": r, **metrics})
        else:
            print(f"\nRound {r}: 数据不足，跳过CSSR")

    return {
        "mode": "offline",
        "metrics_history": all_metrics,
    }


def run_live_experiment(client, config: dict, use_hard: bool = False) -> dict:
    """Run full three-group live experiment with LLM.

    Returns complete experiment results dict.
    """
    llm_cfg = config["llm"]
    cssr_cfg = CSSRConfig(
        L_max=llm_cfg.get("L_max"),
        alpha=llm_cfg.get("alpha", 0.05),
        test=llm_cfg.get("test", "chi2"),
        correction=llm_cfg.get("correction", "bonferroni"),
        min_count=llm_cfg.get("min_count", 3),
    )

    rounds = config.get("rounds", 5)
    groups = config.get("groups", ["A", "B", "C"])
    save_traces = config.get("save_traces", True)
    save_reports = config.get("save_reports", True)

    # Shared encoder and report generator
    encoder = BehaviorEncoder(
        client=client,
        model=llm_cfg.get("model", "gpt-4o"),
        temperature=0.1,
        max_tokens=256,
    )
    report_gen = SelfReportGenerator(
        client=client,
        model=llm_cfg.get("model", "gpt-4o"),
        temperature=0.3,
        max_tokens=400,
    )

    all_results = {}

    for group in groups:
        print(f"\n{'='*60}")
        print(f"组 {group} 开始...")
        print(f"{'='*60}")

        group_history = []
        inject_text = ""

        if group == "B":
            inject_text = GENERIC_FEEDBACK_PROMPT
        elif group == "A":
            self_model = SelfModel(encoder, report_gen, cssr_cfg)

        for r in range(rounds):
            tasks = load_tasks(config, use_hard)
            solver = setup_agent(client, llm_cfg, inject_text)

            if group == "C":
                inject_text = ""  # Never inject anything

            print(f"  Round {r}: 运行 {len(tasks)} 题...")

            round_result = run_round(tasks, solver, group, r, save_traces)
            group_history.append(round_result)

            acc = round_result["accuracy"]
            print(f"  准确率: {acc:.2%} ({round_result['n_correct']}/{round_result['n_total']})")

            # Group A: CSSR analysis and self-report
            if group == "A":
                traces = round_result["traces"]
                sm_result = self_model.analyze_round(traces, r)
                if sm_result.get("report"):
                    inject_text = sm_result["report"]
                    if save_reports:
                        report_path = f"experiments/results/report_{group}_round{r}.md"
                        os.makedirs("experiments/results", exist_ok=True)
                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(inject_text)
                    print(f"  自我报告已生成 ({len(inject_text)} 字符)")

                # Save metrics history
                if sm_result.get("metrics"):
                    metrics_path = f"experiments/results/metrics_{group}.json"
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(self_model.get_metrics_history(), f, indent=2)

        all_results[group] = group_history

    return all_results


def save_results(results: dict, output_dir: str = "experiments/results"):
    """Save experiment results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save accuracy summary
    summary = {}
    for group, history in results.items():
        summary[group] = [
            {"round": h["round"], "accuracy": h["accuracy"],
             "n_correct": h["n_correct"], "n_total": h["n_total"]}
            for h in history
        ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"experiment_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("实验汇总")
    print("=" * 60)
    header = f"{'Round':<8}"
    for g in sorted(results.keys()):
        header += f"{'组'+g:>12}"
    print(header)
    print("-" * len(header))

    n_rounds = len(next(iter(results.values())))
    for r in range(n_rounds):
        row = f"{r:<8}"
        for g in sorted(results.keys()):
            if r < len(results[g]):
                row += f"{results[g][r]['accuracy']:>12.2%}"
            else:
                row += f"{'N/A':>12}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Causal Mirror 实验系统")
    parser.add_argument("--config", default="experiments/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--rounds", type=int, default=None,
                        help="实验轮数（覆盖配置）")
    parser.add_argument("--tasks-per-round", type=int, default=None,
                        help="每轮任务数（覆盖配置）")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="实验组别，例如: A B C")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式：使用模拟数据，无需API")
    parser.add_argument("--hard", action="store_true",
                        help="使用高难度竞赛题集（预期正确率30-60%）")
    parser.add_argument("--api-key", default=None,
                        help="LLM API密钥（或设置环境变量）")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Read llm/cssr from top level (they sit beside "experiment" in the YAML)
        llm_config = raw.get("llm", {})
        cssr_config = raw.get("cssr", {})
        experiment_config = raw.get("experiment", raw)
    else:
        experiment_config = {
            "rounds": 3,
            "tasks_per_round": 10,
            "groups": ["A", "B", "C"],
        }
        llm_config = {"model": "gpt-4o", "temperature": 0.3}
        cssr_config = {}

    # Override from CLI
    if args.rounds is not None:
        experiment_config["rounds"] = args.rounds
    if args.tasks_per_round is not None:
        experiment_config["tasks_per_round"] = args.tasks_per_round
    if args.groups is not None:
        experiment_config["groups"] = args.groups

    # Merge CSSR and LLM config
    experiment_config["llm"] = {**llm_config, **experiment_config.get("llm", {})}
    experiment_config["cssr"] = {**cssr_config, **experiment_config.get("cssr", {})}

    if args.api_key:
        os.environ[llm_config.get("api_key_env", "OPENAI_API_KEY")] = args.api_key

    print(f"配置: {experiment_config['rounds']}轮 × {experiment_config['tasks_per_round']}题")
    print(f"组别: {experiment_config['groups']}")

    if args.offline:
        results = run_offline_experiment(experiment_config, use_hard=args.hard)
        print("\n离线演示完成。")
        return

    # Setup LLM client
    client = setup_llm_client(experiment_config["llm"])
    if client is None:
        print("\n未检测到API密钥。切换到离线模式。")
        print("设置 OPENAI_API_KEY 或使用 --api-key 参数。")
        results = run_offline_experiment(experiment_config, use_hard=args.hard)
        print("\n离线演示完成。")
        return

    # Run experiment
    results = run_live_experiment(client, experiment_config, use_hard=args.hard)

    # Save
    save_results(
        results,
        output_dir=experiment_config.get("output", {}).get("results_dir", "experiments/results"),
    )


if __name__ == "__main__":
    main()
