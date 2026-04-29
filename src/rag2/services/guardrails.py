from __future__ import annotations

from dataclasses import dataclass
import re


PROMPT_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"override\s+safety", re.IGNORECASE),
)


@dataclass(slots=True)
class GuardrailResult:
    status: str
    reason: str


def preflight_guardrail(question: str) -> GuardrailResult:
    normalized = question.strip()
    if not normalized:
        return GuardrailResult(status="blocked", reason="Question cannot be empty.")
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(normalized):
            return GuardrailResult(
                status="blocked",
                reason="The request looks like a prompt-injection or system-bypass attempt.",
            )
    return GuardrailResult(status="ok", reason="Question passed the preflight guardrail.")


def evidence_guardrail(best_relevance: float, hit_count: int, threshold: float) -> GuardrailResult:
    if hit_count == 0 or best_relevance < threshold:
        return GuardrailResult(
            status="insufficient_evidence",
            reason="Retrieved context is too weak to answer this question reliably.",
        )
    return GuardrailResult(status="ok", reason="Retrieved context is strong enough to answer.")
