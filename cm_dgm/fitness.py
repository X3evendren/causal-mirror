"""CM-DGM适应度函数：CM加权 vs 纯准确率。

CM加权公式:
  fitness = accuracy * w_acc
          + C_μ_norm * w_C
          + E_norm * w_E
          - h_μ_norm * w_h
          + alphabet_norm * w_alpha

归一化到[0,1]范围，使指标贡献与准确率可比。

包含验证门：如果CSSR指标未通过完整性检查，拒绝因果奖励，
退回到纯准确率适应度。防止在损坏的指标上做出错误选择。
"""

import logging

logger = logging.getLogger(__name__)

# 归一化除数（基于典型观察范围）
C_MU_MAX = 3.0       # log2(8状态) ≈ 3
E_MAX = 2.0          # 超熵典型上限
H_MU_MAX = 3.0       # 熵率典型上限 (log2(8符号) ≈ 3)
ALPHABET_MAX = 16    # 最大符号种类

# 默认权重
DEFAULT_WEIGHTS = {
    "accuracy": 1.0,
    "C_mu": 0.3,
    "E": 0.2,
    "h_mu": -0.1,
    "alphabet": 0.1,
}


def validate_cssr_metrics(m: dict) -> list[str]:
    """检查CSSR指标是否通过完整性验证。

    返回失败原因列表。空列表表示通过。
    """
    failures = []
    n_states = m.get("n_states", 0)
    C_mu = m.get("statistical_complexity", 0.0)
    E = m.get("excess_entropy", 0.0)
    h_mu = m.get("entropy_rate", 0.0)
    chi = m.get("predictive_information", 0.0)

    # 无状态：CSSR未运行或失败
    if n_states == 0:
        failures.append("n_states=0 (CSSR did not run)")
        return failures

    # 多状态但零复杂度：稳态退化（旧Bug的症状）
    if n_states > 1 and C_mu <= 0.0:
        failures.append(
            f"n_states={n_states} but C_mu={C_mu:.4f} "
            f"(degenerate steady-state distribution)"
        )

    # 有结构但无超熵：转移矩阵可能退化
    if n_states > 1 and C_mu > 0.0 and E <= 0.0:
        failures.append(
            f"n_states={n_states}, C_mu={C_mu:.4f} but E={E:.4f} "
            f"(transition matrix may be degenerate)"
        )

    # 非零熵率是过程随机性的基本检查
    if n_states >= 1 and h_mu <= 0.0:
        failures.append(
            f"h_mu={h_mu:.4f} (zero entropy rate — deterministic or broken)"
        )

    return failures


def cm_weighted_fitness(agent_result: dict, weights: dict | None = None) -> float:
    """CM加权适应度：准确率 + 因果结构奖励。

    在应用因果奖励之前验证CSSR指标。如果验证失败，
    退回纯准确率适应度并记录警告。

    Args:
        agent_result: CMDGMAgent.evaluate() 返回的dict
        weights: 各指标权重，None则用默认

    Returns:
        非负浮点数适应度
    """
    w = weights or DEFAULT_WEIGHTS
    m = agent_result.get("metrics") or {}
    acc = agent_result.get("accuracy", 0.0)

    # 验证门：指标坏了就不用因果奖励
    failures = validate_cssr_metrics(m)
    if failures:
        logger.warning(
            "Agent %s: CSSR validation failed, falling back to accuracy-only. "
            "Failures: %s",
            agent_result.get("agent_id", "?"),
            "; ".join(failures),
        )
        return acc

    C = min(m.get("statistical_complexity", 0.0) / C_MU_MAX, 1.0)
    E = min(m.get("excess_entropy", 0.0) / E_MAX, 1.0)
    h = min(m.get("entropy_rate", 0.0) / H_MU_MAX, 1.0)
    alpha = len(agent_result.get("alphabet_used", [])) / ALPHABET_MAX

    total = (
        w.get("accuracy", 1.0) * acc
        + w.get("C_mu", 0.3) * C
        + w.get("E", 0.2) * E
        + w.get("h_mu", -0.1) * h
        + w.get("alphabet", 0.1) * alpha
    )
    return max(0.0, total)


def accuracy_only_fitness(agent_result: dict) -> float:
    """对照适应度：仅准确率。"""
    return agent_result.get("accuracy", 0.0)
