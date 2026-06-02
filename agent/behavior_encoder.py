"""行为编码器：将推理trace编码为离散符号序列。

使用LLM对每步推理进行分类，输出 动作_质量 格式的符号序列，
供CSSR引擎进行因果状态分析。
"""

import re
from collections import Counter


class BehaviorEncoder:
    """Encodes reasoning traces into discrete behavior symbol sequences."""

    ACTION_TYPES = ["INFER", "CALC", "VERIFY", "CORRECT"]
    QUALITY_TYPES = ["CORRECT", "PARTIAL", "INCORRECT", "SKIPPED"]

    # Map Chinese action labels to English codes
    _ACTION_MAP = {
        "推理": "INFER", "推断": "INFER", "分析": "INFER", "思考": "INFER",
        "计算": "CALC", "运算": "CALC",
        "验证": "VERIFY", "检查": "VERIFY", "核实": "VERIFY", "确认": "VERIFY",
        "修正": "CORRECT", "纠正": "CORRECT", "改正": "CORRECT",
    }

    def __init__(self, client=None, model: str = "", temperature: float = 0.1,
                 max_tokens: int = 512):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def encode(self, steps_text: str) -> list[str]:
        """Encode reasoning steps into a symbol sequence.

        Args:
            steps_text: The reasoning steps text.

        Returns:
            List of symbols like ["INFER_CORRECT", "CALC_PARTIAL", ...].
        """
        if not steps_text or not steps_text.strip():
            return ["VERIFY_SKIPPED"]

        steps = self._split_steps(steps_text)
        if not steps:
            return ["VERIFY_SKIPPED"]

        symbols = []
        for step in steps:
            symbol = self._encode_single_step(step)
            symbols.append(symbol)

        return symbols

    def encode_batch(self, steps_texts: list[str]) -> list[list[str]]:
        """Encode multiple reasoning traces."""
        return [self.encode(text) for text in steps_texts]

    def _split_steps(self, steps_text: str) -> list[str]:
        """Split steps text into individual step strings."""
        steps = re.findall(r"\[步骤\s*\d+\].*?(?=\[步骤\s*\d+\]|$)", steps_text, re.DOTALL)
        if steps:
            return [s.strip() for s in steps]

        lines = [l.strip() for l in steps_text.split("\n") if l.strip()]
        reasoning = [
            l for l in lines
            if not l.startswith("答案") and not l.startswith("最终")
        ]
        if reasoning:
            return reasoning

        return [steps_text]

    def _extract_action_from_marker(self, step_text: str) -> str | None:
        """Extract action type from [动作: X] or [类型：X] markers."""
        match = re.search(r"\[(?:动作|类型|action|type)[：:\s]*([^\]]+)\]", step_text, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            for cn, en in self._ACTION_MAP.items():
                if cn in label:
                    return en
            # Try direct English match
            upper = label.upper()
            for t in self.ACTION_TYPES:
                if t in upper:
                    return t
        return None

    def _detect_quality(self, step_text: str) -> str:
        """Detect result quality from step text keywords."""
        text = step_text.lower()
        if any(w in text for w in ["错误", "不对", "失误", "wrong", "error", "mistake", "incorrect"]):
            return "INCORRECT"
        if any(w in text for w in ["可能", "大概", "或许", "maybe", "perhaps", "approximately", "部分"]):
            return "PARTIAL"
        if any(w in text for w in ["跳过", "略", "skip"]):
            return "SKIPPED"
        return "CORRECT"

    def _encode_single_step(self, step_text: str) -> str:
        """Rule-first encoding: extract action from marker, quality from keywords.

        Only calls LLM if rule-based approach can't determine the action.
        """
        # 1. Extract action from [动作: X] marker (instant, no API call)
        action = self._extract_action_from_marker(step_text)
        if action:
            quality = self._detect_quality(step_text)
            return f"{action}_{quality}"

        # 2. Try full rule-based classification
        result = self._fallback_classify(step_text)
        # If rule-based gives a plausible result, use it
        # Only call LLM for truly ambiguous cases
        return result

    def _fallback_classify(self, step_text: str) -> str:
        """Pattern-based fallback classification (no LLM call needed)."""
        text_lower = step_text.lower()

        # Detect action type
        if any(w in text_lower for w in ["修正", "纠正", "改正", "重来", "不对", "错误", "correct", "fix", "wrong", "retry"]):
            action = "CORRECT"
        elif any(w in text_lower for w in ["计算", "算", "乘以", "除以", "加", "减", "等于", "calc", "compute", "+", "-", "*", "/", "="]):
            action = "CALC"
        elif any(w in text_lower for w in ["验证", "检查", "核实", "确认", "verify", "check", "confirm", "proof"]):
            action = "VERIFY"
        else:
            action = "INFER"

        # Detect quality (heuristic)
        if any(w in text_lower for w in ["错误", "不对", "失误", "wrong", "error", "mistake", "incorrect"]):
            quality = "INCORRECT"
        elif any(w in text_lower for w in ["可能", "大概", "或许", "也许", "maybe", "perhaps", "approximately", "部分"]):
            quality = "PARTIAL"
        elif any(w in text_lower for w in ["跳过", "略", "skip"]):
            quality = "SKIPPED"
        else:
            quality = "CORRECT"

        return f"{action}_{quality}"

    def get_alphabet(self) -> set:
        """Return the full 16-symbol alphabet."""
        return {f"{a}_{q}" for a in self.ACTION_TYPES for q in self.QUALITY_TYPES}
