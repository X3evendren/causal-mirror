"""CM-DGM双分支实验：CM加权 vs 纯准确率进化。

运行两个并行的进化种群（相同初始条件），比较：
- 准确率轨迹
- C_μ轨迹
- 行为多样性

结果保存到 cm_dgm/results/evolution.jsonl
"""

import json
import os
import time
import copy
from datetime import datetime
from openai import OpenAI

from cssr import CSSRConfig
from agent.task_solver import TaskSolver
from agent.behavior_encoder import BehaviorEncoder
from agent.prompts import SYSTEM_PROMPT_BASE
from cm_dgm.population import Population

# 初始策略基因组（6个不同启发式）
INITIAL_GENOMES = [
    "",  # 空策略 (baseline)
    "请仔细检查每一步的计算结果，特别留意进位和单位换算。",
    "解题前先明确已知条件和所求，规划好步骤再开始计算。",
    "得出答案后换一种方法重新计算，确保两次结果一致。",
    "注意审题陷阱：概率题考虑条件概率，计数题注意重复和遗漏。",
    "每一步推理都要写明依据，不跳步，不凭直觉直接写答案。",
]


def run_experiment(
    problems: list[dict],
    llm_base_url: str = "http://localhost:11434/v1",
    llm_model: str = "lfm2.5-thinking:latest",
    llm_max_tokens: int = 4096,
    llm_temperature: float = 0.3,
    n_generations: int = 10,
    n_agents: int = 6,
    elitism: int = 1,
    tournament_k: int = 3,
    cssr_config: CSSRConfig | None = None,
    output_path: str = "cm_dgm/results/evolution.jsonl",
) -> dict:
    """运行CM-DGM双分支实验。

    Returns:
        包含两个分支最终统计的dict
    """
    cssr_config = cssr_config or CSSRConfig(L_max=None, alpha=0.05, min_count=2)

    # 共享LLM客户端
    client = OpenAI(base_url=llm_base_url, api_key="ollama")

    # 共享行为编码器（规则驱动，不需要client）
    encoder = BehaviorEncoder()

    # 克隆问题集确保两个分支使用相同题目
    problems_a = copy.deepcopy(problems)
    problems_b = copy.deepcopy(problems)

    # 创建两个种群
    pop_a = Population(initial_genomes=copy.deepcopy(INITIAL_GENOMES[:n_agents]),
                        branch="cm_dgm")
    pop_b = Population(initial_genomes=copy.deepcopy(INITIAL_GENOMES[:n_agents]),
                        branch="fitness_only")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    archive = open(output_path, "w", encoding="utf-8")

    print("=" * 60)
    print("CM-DGM 双分支进化实验")
    print(f"  代数: {n_generations}, 种群: {n_agents}, 题目: {len(problems)}")
    print(f"  分支A: CM加权适应度 (causal structure bonus)")
    print(f"  分支B: 纯准确率适应度 (baseline)")
    print("=" * 60)

    for gen in range(n_generations):
        t0 = time.time()
        print(f"\n--- 第{gen}代 ---")

        # 分支A: CM-DGM
        solver_factory_a = lambda: TaskSolver(
            client=client, model=llm_model,
            system_prompt=SYSTEM_PROMPT_BASE,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )
        pop_a.evolve(problems_a, solver_factory_a, encoder, cssr_config,
                     mutate_client=client, mutate_model=llm_model,
                     elitism=elitism, tournament_k=tournament_k)
        stats_a = pop_a.get_stats()

        # 分支B: Fitness-Only
        solver_factory_b = lambda: TaskSolver(
            client=client, model=llm_model,
            system_prompt=SYSTEM_PROMPT_BASE,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )
        pop_b.evolve(problems_b, solver_factory_b, encoder, cssr_config,
                     mutate_client=client, mutate_model=llm_model,
                     elitism=elitism, tournament_k=tournament_k)
        stats_b = pop_b.get_stats()

        elapsed = time.time() - t0
        print(f"  [CM-DGM]      平均准确率: {stats_a['mean_accuracy']:.2%}  "
              f"最佳: {stats_a['best_accuracy']:.2%}")
        print(f"  [Fitness-Only] 平均准确率: {stats_b['mean_accuracy']:.2%}  "
              f"最佳: {stats_b['best_accuracy']:.2%}")
        print(f"  耗时: {elapsed:.0f}s")

        # 写入存档
        for snap in pop_a.history[-len(pop_a.agents)-1:]:  # 本代快照 + 摘要
            archive.write(json.dumps(snap, ensure_ascii=False) + "\n")
        for snap in pop_b.history[-len(pop_b.agents)-1:]:
            archive.write(json.dumps(snap, ensure_ascii=False) + "\n")
        archive.flush()

    archive.close()

    # 最终统计
    final_a = pop_a.get_stats()
    final_b = pop_b.get_stats()

    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"  CM-DGM:        avg={final_a['mean_accuracy']:.2%}  "
          f"best={final_a['best_accuracy']:.2%}")
    print(f"  Fitness-Only:  avg={final_b['mean_accuracy']:.2%}  "
          f"best={final_b['best_accuracy']:.2%}")

    # 显示最佳基因组
    print("\n最佳基因组:")
    print(f"  [CM-DGM]       {pop_a.agents[0].genome[:80]}")
    print(f"  [Fitness-Only] {pop_b.agents[0].genome[:80]}")

    print(f"\n完整存档: {output_path}")

    return {
        "cm_dgm": final_a,
        "fitness_only": final_b,
        "archive": output_path,
    }


def main():
    import argparse
    from experiments.hard_problems import HARD_TASKS

    parser = argparse.ArgumentParser(description="CM-DGM 双分支进化实验")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--problems", type=int, default=20,
                        help="每代题目数（从HARD_TASKS取前N道）")
    parser.add_argument("--elitism", type=int, default=1)
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：2代×3Agent×5题")
    args = parser.parse_args()

    if args.quick:
        args.generations = 2
        args.agents = 3
        args.problems = 5

    problems = HARD_TASKS[:args.problems]

    result = run_experiment(
        problems=problems,
        n_generations=args.generations,
        n_agents=args.agents,
        elitism=args.elitism,
    )
    return result


if __name__ == "__main__":
    main()
