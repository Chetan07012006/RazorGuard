from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    score: float
    level: str
    action: str


def decide(score: float) -> RiskDecision:
    if not 0 <= score <= 1:
        raise ValueError("Risk score must be between 0 and 1.")
    if score >= 0.80:
        return RiskDecision(score, "high", "review")
    if score >= 0.50:
        return RiskDecision(score, "medium", "verify")
    return RiskDecision(score, "low", "allow")
