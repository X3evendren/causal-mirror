"""LLM驱动的策略提示词变异。

使用lfm2.5（快、免费、足够"笨"产生有趣的策略漂移）。
"""

MUTATION_PROMPT = (
    "你是一个策略优化器。下面是一个数学解题的策略提示词：\n\n"
    '"{genome}"\n\n'
    "请生成一个略微修改的版本。改动1-3个词，或者改写句式，"
    "或者添加/删除一个细节。保持大致相同的长度，用中文。"
    "只输出修改后的策略文本，不要加任何解释。"
)

NOVEL_PROMPT = (
    "你是一个策略设计器。请为数学解题创造一个简短有效的策略提示词"
    "（50-100字，中文）。策略应该有助于提高解题准确率。"
    "只输出策略文本，不要加任何解释。"
)

GENOME_MAX_LEN = 200


def llm_mutate(client, model: str, genome: str, temperature: float = 0.7,
               novel_prob: float = 0.2) -> str:
    """使用LLM变异基因组。

    Args:
        client: OpenAI兼容客户端
        model: 模型名
        genome: 当前策略文本（可能为空）
        temperature: 变异温度（越高越随机）
        novel_prob: 生成全新策略的概率

    Returns:
        变异后的策略文本
    """
    import random

    # 一定概率生成全新策略
    if not genome or random.random() < novel_prob:
        prompt = NOVEL_PROMPT
    else:
        prompt = MUTATION_PROMPT.format(genome=genome)

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150,
        )
        new_genome = r.choices[0].message.content or ""
        new_genome = new_genome.strip()

        # 清理LLM可能添加的格式标记
        for marker in ["策略：", "策略:", "新策略：", "修改后："]:
            if new_genome.startswith(marker):
                new_genome = new_genome[len(marker):].strip()

        if len(new_genome) < 5:
            return genome  # 太短，回退

        return new_genome[:GENOME_MAX_LEN]
    except Exception:
        return genome  # API错误，回退原基因组
