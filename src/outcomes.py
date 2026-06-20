"""
Canonical 1X2 outcome logic for prediction evaluation.

This module owns the definition of:
- What the aggregate predicted outcome is (from probabilities)
- What the actual outcome is (from goals)
- Whether a prediction hit the 1X2 outcome

All other modules should import from here rather than duplicating the rules.
"""

def get_predicted_1x2_outcome(
    team_a_win_probability: float,
    draw_probability: float,
    team_b_win_probability: float,
) -> str:
    """
    Determine the predicted 1X2 outcome from the three aggregate probabilities.

    Returns:
        "1" if Team A (first listed) is predicted to win
        "X" if draw is predicted
        "2" if Team B (second listed) is predicted to win

    Tie-breaker:
        When two or more probabilities are exactly equal (after float comparison),
        "X" (draw) is chosen deterministically. This preserves historical behavior
        of the original inline implementations.
    """
    wa = float(team_a_win_probability or 0.0)
    d = float(draw_probability or 0.0)
    wb = float(team_b_win_probability or 0.0)

    if wa > d and wa > wb:
        return "1"
    elif d > wa and d > wb:
        return "X"
    elif wb > wa and wb > d:
        return "2"
    else:
        # Tie-breaker: favor X (draw) on exact equality, matching prior logic
        if wa == d and wa > wb:
            return "X"
        elif wb == d and wb > wa:
            return "X"
        else:
            return "X"


def get_actual_1x2_outcome(team_a_goals: int, team_b_goals: int) -> str:
    """
    Determine the actual 1X2 outcome from final scores.

    Returns:
        "1" if team_a scored more
        "X" if equal
        "2" if team_b scored more
    """
    a = int(team_a_goals)
    b = int(team_b_goals)
    if a > b:
        return "1"
    elif a == b:
        return "X"
    else:
        return "2"


def is_1x2_hit(
    team_a_win_probability: float,
    draw_probability: float,
    team_b_win_probability: float,
    team_a_goals: int,
    team_b_goals: int,
) -> bool:
    """Return True iff the aggregate predicted 1X2 matches the actual result."""
    predicted = get_predicted_1x2_outcome(
        team_a_win_probability, draw_probability, team_b_win_probability
    )
    actual = get_actual_1x2_outcome(team_a_goals, team_b_goals)
    return predicted == actual
