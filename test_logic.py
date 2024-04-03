import pytest

from app import compound_interest, investment_evolution_breakdown


def test_compound_interest_basic():
    """Test basic compound interest calculation without contributions."""
    periods, f = compound_interest(
        initial_amount=1000, interest_rate=0.05, periods=10, contributions=0
    )
    assert f[-1] == pytest.approx(1628.89, 0.01)


def test_investment_evolution_breakdown_basic():
    """Test investment evolution with contributions."""
    df = investment_evolution_breakdown(
        initial_amount=1000, interest_rate=0.05, periods=10, contributions=100
    )
    assert df.iloc[-1]["value"] == pytest.approx(2949.57, 0.01)
    assert df.iloc[-1]["value_from_contributions"] == pytest.approx(2000, 0.01)
    assert df.iloc[-1]["value_from_interest"] == pytest.approx(
        df.iloc[-1]["value"] - df.iloc[-1]["value_from_contributions"]
    )
