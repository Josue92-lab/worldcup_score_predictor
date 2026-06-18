import numpy as np
import pandas as pd
from scipy.stats import poisson
from src.config import PROCESSED_DIR

def calculate_match_probabilities(lambda_a: float, lambda_b: float, max_goals: int = 7, rho: float = -0.13):
    """
    Calculates the probability matrix for a match given xG for Team A and Team B.
    Applies the Dixon-Coles adjustment for low-scoring outcomes.
    Returns:
    - Prob matrix (A goals x B goals)
    - Top 5 scorelines
    - Win/Draw/Loss probs
    """
    prob_a = [poisson.pmf(i, lambda_a) for i in range(max_goals + 1)]
    prob_b = [poisson.pmf(i, lambda_b) for i in range(max_goals + 1)]
    
    # Probability matrix (independent Poisson)
    prob_matrix = np.outer(prob_a, prob_b)
    
    # Apply Dixon-Coles adjustment (rho parameter)
    # tau(0, 0) = 1 - (lambda_a * lambda_b * rho)
    # tau(1, 0) = 1 + (lambda_b * rho)
    # tau(0, 1) = 1 + (lambda_a * rho)
    # tau(1, 1) = 1 - rho
    if rho != 0.0:
        # Ensure probabilities don't become negative due to large lambdas or rho
        tau_00 = max(0, 1 - (lambda_a * lambda_b * rho))
        tau_10 = max(0, 1 + (lambda_b * rho))
        tau_01 = max(0, 1 + (lambda_a * rho))
        tau_11 = max(0, 1 - rho)
        
        prob_matrix[0, 0] *= tau_00
        prob_matrix[1, 0] *= tau_10
        prob_matrix[0, 1] *= tau_01
        prob_matrix[1, 1] *= tau_11
    
    # Normalize if truncation or adjustment loses/gains too much probability
    prob_matrix = prob_matrix / prob_matrix.sum()
    
    scorelines = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            scorelines.append({
                "scoreline": f"{i}-{j}",
                "probability": float(prob_matrix[i, j])
            })
            
    # Sort and get top 5
    scorelines.sort(key=lambda x: x['probability'], reverse=True)
    top_5 = scorelines[:5]
    
    # W/D/L
    win_a = np.tril(prob_matrix, -1).sum()
    draw = np.trace(prob_matrix)
    win_b = np.triu(prob_matrix, 1).sum()
    
    return {
        "matrix": prob_matrix,
        "top_5": top_5,
        "win_a": float(win_a),
        "draw": float(draw),
        "win_b": float(win_b)
    }

def fit_poisson_model(historical_features_df):
    """
    Empirical calculation of attack and defense strengths.
    Attack Strength = Average goals scored by team / Average goals scored by all teams
    Defense Weakness = Average goals conceded by team / Average goals conceded by all teams
    """
    model = {
        "attack": {},
        "defense": {},
        "base_rate": 1.2
    }
    
    if historical_features_df.empty:
        return model
        
    # We expect historical_features_df to have rows like: home_team, away_team, home_score, away_score
    if 'home_score' not in historical_features_df.columns:
        return model
        
    avg_home_goals = historical_features_df['home_score'].mean()
    avg_away_goals = historical_features_df['away_score'].mean()
    overall_avg = (avg_home_goals + avg_away_goals) / 2.0
    
    model["base_rate"] = float(overall_avg)
    
    # Calculate for each team
    teams = set(historical_features_df['home_team']).union(set(historical_features_df['away_team']))
    
    for team in teams:
        home_matches = historical_features_df[historical_features_df['home_team'] == team]
        away_matches = historical_features_df[historical_features_df['away_team'] == team]
        
        total_matches = len(home_matches) + len(away_matches)
        if total_matches == 0:
            continue
            
        goals_scored = home_matches['home_score'].sum() + away_matches['away_score'].sum()
        goals_conceded = home_matches['away_score'].sum() + away_matches['home_score'].sum()
        
        team_avg_scored = goals_scored / total_matches
        team_avg_conceded = goals_conceded / total_matches
        
        model["attack"][team] = float(team_avg_scored / overall_avg) if overall_avg > 0 else 1.0
        model["defense"][team] = float(team_avg_conceded / overall_avg) if overall_avg > 0 else 1.0

    return model
    
def predict_xg(team_a, team_b, model):
    base = model.get('base_rate', 1.2)
    # Defaults to 1.0 if not found
    att_a = model['attack'].get(team_a, 1.0)
    def_a = model['defense'].get(team_a, 1.0)
    att_b = model['attack'].get(team_b, 1.0)
    def_b = model['defense'].get(team_b, 1.0)
    
    xg_a = base * att_a * def_b
    xg_b = base * att_b * def_a
    
    return xg_a, xg_b
