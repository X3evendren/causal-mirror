#!/usr/bin/env python3
"""Causal Mirror Benchmark CLI — run graduate-level validation experiments.

Usage:
    # Offline simulation (no API needed)
    python -m experiments.benchmark_cli --offline --rounds 3

    # Real API (ModelScope DeepSeek V4)
    python -m experiments.benchmark_cli --provider modelscope --rounds 4

    # With custom API key
    python -m experiments.benchmark_cli --api-key sk-xxx --rounds 3

    # Only specific groups
    python -m experiments.benchmark_cli --offline --groups A C --rounds 2
"""

import argparse
import os
import sys
import yaml

from experiments.graduate_problems import GRADUATE_PROBLEMS, DOMAIN_MAP
from experiments.benchmark_runner import (
    ExperimentHarness, compare_groups, print_comparison, save_results,
)


def main():
    parser = argparse.ArgumentParser(description="Causal Mirror 基准测试")
    parser.add_argument("--config", default="experiments/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--offline", action="store_true",
                        help="离线模拟模式（无需API）")
    parser.add_argument("--provider", default=None,
                        help="LLM provider: modelscope, openai")
    parser.add_argument("--model", default=None,
                        help="模型名称（覆盖配置文件）")
    parser.add_argument("--rounds", type=int, default=None,
                        help="实验轮数")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="实验组别，如: A B C")
    parser.add_argument("--api-key", default=None,
                        help="API密钥（或设环境变量 MODELScope_API_KEY）")
    parser.add_argument("--output", default="experiments/results",
                        help="结果输出目录")
    parser.add_argument("--problems", type=int, default=None,
                        help="题目数量限制（默认全部30题）")
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        llm_cfg = raw.get("llm", {})
        cssr_cfg = raw.get("cssr", {})
        experiment_cfg = raw.get("experiment", raw)
        config = {
            "llm": dict(llm_cfg),
            "cssr": dict(cssr_cfg),
            "experiment": dict(experiment_cfg),
            "output": raw.get("output", {}),
        }
    else:
        config = {
            "llm": {"provider": "modelscope", "model": "deepseek-ai/DeepSeek-V3"},
            "cssr": {"alpha": 0.05, "test": "chi2", "correction": "bonferroni", "min_count": 2},
            "experiment": {"rounds": 4, "groups": ["A", "B", "C"]},
        }

    # CLI overrides
    if args.provider:
        config["llm"]["provider"] = args.provider
    if args.model:
        config["llm"]["model"] = args.model
    if args.rounds is not None:
        config["experiment"]["rounds"] = args.rounds
    if args.groups is not None:
        config["experiment"]["groups"] = args.groups
    config["experiment"]["offline"] = args.offline

    if args.api_key:
        os.environ[config["llm"].get("api_key_env", "MODELSCOPE_API_KEY")] = args.api_key

    # Select problems
    problems = GRADUATE_PROBLEMS
    if args.problems and args.problems < len(problems):
        import random
        random.seed(0)
        problems = random.sample(problems, args.problems)

    # Print setup
    mode = "offline" if args.offline else "API"
    provider = config.get("llm", {}).get("provider", "?")
    model_name = config.get("llm", {}).get("model", "?")
    print(f"Causal Mirror - graduate benchmark")
    print(f"  problems: {len(problems)}")
    print(f"  rounds: {config['experiment']['rounds']}")
    print(f"  groups: {config['experiment']['groups']}")
    print(f"  mode: {mode} ({provider}/{model_name})")

    # Print domain breakdown
    from collections import Counter
    domains = Counter(p["domain"] for p in problems)
    for d, c in domains.items():
        print(f"    {DOMAIN_MAP.get(d, d)}: {c}")

    # Run experiment
    harness = ExperimentHarness(config)
    group_results = harness.run(problems)

    # Compare
    comparison = compare_groups(group_results)
    print_comparison(comparison)

    # Save
    save_results(group_results, args.output)


if __name__ == "__main__":
    main()
