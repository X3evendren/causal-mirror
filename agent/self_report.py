"""自我认知报告生成器：将CSSR分析结果转换为自然语言自我报告。"""

from agent.prompts import SELF_REPORT_PROMPT


class SelfReportGenerator:
    """Generates natural-language self-cognition reports from CSSR analysis."""

    def __init__(self, client, model: str, temperature: float = 0.3,
                 max_tokens: int = 600):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, metrics: dict, machine) -> str:
        """Generate a self-cognition report from CSSR output.

        Args:
            metrics: Dict from compute_all_metrics().
            machine: EpsilonMachine instance.

        Returns:
            Natural language self-report string.
        """
        # Build state descriptions
        state_descriptions = self._describe_states(machine)

        # Build key findings
        key_findings = self._identify_key_findings(machine, metrics)

        # Use LLM to generate the report
        prompt = SELF_REPORT_PROMPT.format(
            C_mu=metrics["statistical_complexity"],
            E=metrics["excess_entropy"],
            h_mu=metrics["entropy_rate"],
            chi=metrics["predictive_information"],
            n_states=metrics["n_states"],
            state_descriptions=state_descriptions,
            key_findings=key_findings,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback: generate a structured report without LLM
            return self._fallback_report(metrics, state_descriptions, key_findings)

    def _describe_states(self, machine) -> str:
        """Build text descriptions of each causal state."""
        descriptions = []
        for s in machine.states:
            probs = ", ".join(
                f"{b}: {p:.2f}" for b, p in sorted(s.emission_probs.items())
            )
            n = len(s.histories)
            descriptions.append(
                f"状态{s.state_id}（{n}个历史模式）: 输出分布 {{{probs}}}"
            )
        if not descriptions:
            return "未检测到显著的因果状态"
        return "\n".join(descriptions)

    def _identify_key_findings(self, machine, metrics: dict) -> str:
        """Identify key behavioral patterns from the epsilon-machine."""
        findings = []

        # High error rate states
        for s in machine.states:
            error_prob = 0.0
            for sym, prob in s.emission_probs.items():
                if "_INCORRECT" in str(sym) or "_PARTIAL" in str(sym):
                    error_prob += prob
            if error_prob > 0.5:
                findings.append(
                    f"状态{s.state_id}中错误/偏差概率高达{error_prob:.1%}：此状态代表高失败风险模式"
                )

        # Low complexity
        if metrics["statistical_complexity"] < 0.2:
            findings.append("行为模式过于简单（C_μ < 0.2 bits）：可能缺乏多样化的推理策略")
        elif metrics["statistical_complexity"] > 2.0:
            findings.append("行为模式非常复杂（C_μ > 2.0 bits）：可能存在不必要的策略碎片化")

        # High entropy rate
        if metrics["entropy_rate"] > 2.0:
            findings.append("每步推理的不确定性很高（h_μ > 2.0 bits）：行为缺乏稳定性")

        # Predictive information
        if metrics["predictive_information"] > 1.0:
            findings.append("内部状态对下一步有较强预测力：行为模式高度结构化")

        if not findings:
            findings.append("未发现显著的风险模式")

        return "\n".join(f"- {f}" for f in findings)

    def _fallback_report(self, metrics, state_descriptions, key_findings) -> str:
        """Generate a structured report without LLM."""
        C, E, h, chi = (
            metrics["statistical_complexity"],
            metrics["excess_entropy"],
            metrics["entropy_rate"],
            metrics["predictive_information"],
        )
        return (
            f"# 自我认知报告\n\n"
            f"## 行为度量\n"
            f"- 统计复杂度 C_μ: {C:.4f} bits\n"
            f"- 超熵 E: {E:.4f} bits\n"
            f"- 熵率 h_μ: {h:.4f} bits\n"
            f"- 预测信息 χ: {chi:.4f} bits\n"
            f"- 因果状态数: {metrics['n_states']}\n\n"
            f"## 因果状态结构\n{state_descriptions}\n\n"
            f"## 关键发现\n{key_findings}\n\n"
            f"## 建议\n基于以上分析，在下一轮中应特别关注高失败率状态对应的行为模式。"
        )
