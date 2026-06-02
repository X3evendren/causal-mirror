"""种群管理 + 进化循环。

每代：评估所有Agent → 计算适应度 → 锦标赛选择亲本 → 变异 → 替换。
"""

import random
import time
from cm_dgm.agent import CMDGMAgent
from cm_dgm.fitness import cm_weighted_fitness, accuracy_only_fitness
from cm_dgm.mutation import llm_mutate


class Population:
    """管理种群并执行进化循环。"""

    def __init__(self, initial_genomes: list[str], branch: str = "cm_dgm"):
        self.branch = branch
        self.agents = [
            CMDGMAgent(agent_id=i, genome=g)
            for i, g in enumerate(initial_genomes)
        ]
        self.generation = 0
        self.history: list[dict] = []
        self._next_id = len(initial_genomes)

    def evaluate_all(
        self, problems: list[dict], solver_factory, encoder, cssr_config
    ) -> list[dict]:
        """评估种群中所有Agent。"""
        results = []
        for i, agent in enumerate(self.agents):
            solver = solver_factory()
            result = agent.evaluate(problems, solver, encoder, cssr_config)
            agent.generation = self.generation
            results.append(result)
            if len(self.agents) > 1:
                print(f"    [{self.branch}] agent {i+1}/{len(self.agents)} "
                      f"acc={agent.accuracy:.0%} "
                      f"sym={len(agent.symbols)} "
                      f"states={agent.metrics.get('n_states',0) if agent.metrics else 0}")
        return results

    def evolve(
        self,
        problems: list[dict],
        solver_factory,
        encoder,
        cssr_config,
        mutate_client,
        mutate_model: str,
        elitism: int = 1,
        tournament_k: int = 3,
    ):
        """执行一代进化：评估 → 选择 → 变异 → 替换。

        Args:
            problems: 问题列表
            solver_factory: 返回新TaskSolver的可调用对象
            encoder: BehaviorEncoder实例
            cssr_config: CSSRConfig
            mutate_client: LLM客户端（用于变异）
            mutate_model: 变异用模型名
            elitism: 保留的最优个体数
            tournament_k: 锦标赛大小
        """
        t0 = time.time()

        # 1. 评估
        results = self.evaluate_all(problems, solver_factory, encoder, cssr_config)

        # 2. 计算适应度
        fitnesses = []
        for r in results:
            if self.branch == "cm_dgm":
                f = cm_weighted_fitness(r)
            else:
                f = accuracy_only_fitness(r)
            fitnesses.append(f)

        # 3. 存档
        for agent, fit in zip(self.agents, fitnesses):
            snap = agent.snapshot()
            snap["fitness"] = fit
            snap["branch"] = self.branch
            self.history.append(snap)

        # 4. 世代摘要
        accs = [a.accuracy for a in self.agents]
        C_vals = [
            a.metrics.get("statistical_complexity", 0) if a.metrics else 0
            for a in self.agents
        ]
        h_vals = [
            a.metrics.get("entropy_rate", 0) if a.metrics else 0
            for a in self.agents
        ]
        summary = {
            "gen": self.generation,
            "branch": self.branch,
            "event": "generation_summary",
            "mean_accuracy": sum(accs) / len(accs),
            "best_accuracy": max(accs),
            "worst_accuracy": min(accs),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "best_fitness": max(fitnesses),
            "mean_C_mu": sum(C_vals) / len(C_vals),
            "mean_h_mu": sum(h_vals) / len(h_vals),
            "best_genome": max(zip(accs, self.agents), key=lambda x: x[0])[1].genome,
            "alphabet_diversity": len(
                set(s for a in self.agents for s in a.symbols)
            ),
            "elapsed_s": time.time() - t0,
        }
        self.history.append(summary)

        # 5. 选择 + 变异
        new_agents = []

        # 精英保留
        ranked = sorted(
            zip(fitnesses, self.agents), key=lambda x: x[0], reverse=True
        )
        for _, elite_agent in ranked[:elitism]:
            new_agents.append(elite_agent)

        # 锦标赛选择 + LLM变异
        while len(new_agents) < len(self.agents):
            parent = self._tournament_select(fitnesses, tournament_k)
            new_genome = llm_mutate(mutate_client, mutate_model, parent.genome)
            child = CMDGMAgent(agent_id=self._next_id, genome=new_genome)
            child.parent_id = parent.agent_id
            child.generation = self.generation + 1
            new_agents.append(child)
            self._next_id += 1

        self.agents = new_agents[:len(self.agents)]  # 截断到种群大小
        self.generation += 1

    def _tournament_select(self, fitnesses: list[float], k: int = 3) -> CMDGMAgent:
        """锦标赛选择：随机选k个，返回适应度最高的。"""
        indices = random.sample(range(len(self.agents)), min(k, len(self.agents)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return self.agents[best_idx]

    def get_stats(self) -> dict:
        """获取当前种群统计。"""
        accs = [a.accuracy for a in self.agents if a.n_total > 0]
        return {
            "generation": self.generation,
            "n_agents": len(self.agents),
            "mean_accuracy": sum(accs) / len(accs) if accs else 0,
            "best_accuracy": max(accs) if accs else 0,
            "genomes": [a.genome[:50] for a in self.agents],
        }
