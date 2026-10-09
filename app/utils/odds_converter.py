"""
OmniPicks — Utilities: Odds Conversion
"""

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds == 0:
        return 1.0
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def decimal_to_american(odds: float) -> float:
    """Convert decimal odds to American odds."""
    if odds <= 1.0:
        return 0.0
    if odds >= 2.0:
        return (odds - 1) * 100
    else:
        return -100 / (odds - 1)

def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        return 0.0
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def decimal_to_implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        return 0.0
    return 1 / odds

def devig_odds(probs: list[float]) -> list[float]:
    """
    Remove the vigorish (bookmaker margin) from a list of implied probabilities.
    Returns fair (true) probabilities that sum to 1.0.
    """
    total_implied = sum(probs)
    if total_implied <= 0:
        return probs
    return [p / total_implied for p in probs]

def format_american_odds(odds: float) -> str:
    """Format American odds as string (e.g., '+150' or '-110')."""
    if odds > 0:
        return f"+{int(odds)}"
    return str(int(odds))
