def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def confidence_score(relevances: list[float], citation_count: int, guardrail_status: str) -> tuple[float, str]:
    if guardrail_status == "blocked":
        return 0.0, "The question was blocked by guardrails."
    if not relevances:
        return 0.05, "No supporting chunks were retrieved."

    top_three = relevances[:3]
    retrieval_strength = sum(top_three) / len(top_three)
    citation_bonus = min(citation_count / 4, 1.0) * 0.15
    evidence_bonus = min(len(relevances) / 6, 1.0) * 0.1
    penalty = 0.25 if guardrail_status == "insufficient_evidence" else 0.0
    score = clamp(retrieval_strength * 0.75 + citation_bonus + evidence_bonus - penalty)

    if score >= 0.8:
        reason = "Strong retrieval matches and enough supporting citations were found."
    elif score >= 0.55:
        reason = "The answer is grounded, but the retrieved evidence is only moderately strong."
    else:
        reason = "The answer relies on weak or limited supporting evidence."

    return round(score, 3), reason