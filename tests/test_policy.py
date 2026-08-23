import pytest

from razorguard.policy import decide


def test_policy_boundaries() -> None:
    assert decide(0.49).action == "allow"
    assert decide(0.50).action == "verify"
    assert decide(0.80).action == "review"


def test_policy_rejects_invalid_score() -> None:
    with pytest.raises(ValueError):
        decide(1.01)
