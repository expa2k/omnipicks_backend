"""
OmniPicks — ML: Poisson Model Stub
"""
class DixonColesModel:
    def __init__(self):
        pass
    def predict_match(self, home_team, away_team):
        return {
            "home_win_prob": 0.5,
            "draw_prob": 0.25,
            "away_win_prob": 0.25,
            "expected_home_goals": 1.5,
            "expected_away_goals": 1.0,
            "over_under_probs": {"over_2.5": 0.45, "under_2.5": 0.55}
        }
