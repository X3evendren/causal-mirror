"""LLM 任务求解器：调用LLM解答数学题并记录推理trace。"""

import re
import json
import time
from typing import Any


class TaskSolver:
    """LLM-based task solver with trace recording."""

    def __init__(self, client, model: str, system_prompt: str = "",
                 temperature: float = 0.3, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def solve(self, problem: str, inject_text: str = "") -> dict[str, Any]:
        """Solve a single math problem.

        Args:
            problem: The problem text.
            inject_text: Optional text to inject into system prompt (self-report).

        Returns:
            dict with keys: problem, full_response, final_answer, is_correct,
            steps_text, trace_timestamp.
        """
        system_msg = self.system_prompt
        if inject_text:
            system_msg = system_msg + "\n\n" + inject_text

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": problem},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            msg = response.choices[0].message
            # 思考模型 (如 lfm2.5-thinking) 的推理放在 reasoning 字段，答案在 content
            reasoning = getattr(msg, "reasoning", "") or ""
            full_response = (reasoning + "\n" + (msg.content or "")).strip()
        except Exception as e:
            return {
                "problem": problem,
                "full_response": f"ERROR: {e}",
                "final_answer": None,
                "is_correct": False,
                "steps_text": "",
                "error": str(e),
            }

        # Extract final answer
        final_answer = self._extract_final_answer(full_response)

        # Extract reasoning steps
        steps_text = self._extract_steps(full_response)

        return {
            "problem": problem,
            "full_response": full_response,
            "final_answer": final_answer,
            "is_correct": None,  # Will be checked against ground truth
            "steps_text": steps_text,
        }

    def _extract_final_answer(self, text: str):
        """Extract the final numeric answer from the response."""
        # Pattern: "答案: <number>" or "答案：<number>" or "#### <number>"
        patterns = [
            r"答案[：:]\s*([\d,]+\.?\d*)",
            r"####\s*([\d,]+\.?\d*)",
            r"answer\s*(?:is|:)?\s*([\d,]+\.?\d*)",
            # Last number in text as fallback
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                ans = matches[-1].replace(",", "")
                try:
                    return float(ans) if "." in ans else int(ans)
                except ValueError:
                    continue

        # Fallback: try to find the last number
        all_numbers = re.findall(r"[\d,]+\.?\d*", text)
        if all_numbers:
            try:
                ans = all_numbers[-1].replace(",", "")
                return float(ans) if "." in ans else int(ans)
            except ValueError:
                pass
        return None

    def _extract_steps(self, text: str) -> str:
        """Extract reasoning steps from the response."""
        # Match [步骤 N] through to next [步骤 M] or end of text
        steps = re.findall(r"\[步骤\s*\d+\].*?(?=\[步骤\s*\d+\]|$)", text, re.DOTALL)
        if steps:
            return "\n".join(step.strip() for step in steps)

        # If no explicit step markers, split by newlines and filter
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        reasoning_lines = [
            l for l in lines
            if not l.startswith("答案") and not l.startswith("####")
        ]
        return "\n".join(reasoning_lines)
