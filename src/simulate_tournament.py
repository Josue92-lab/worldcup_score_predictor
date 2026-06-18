import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIXTURE_PATH, PROCESSED_DIR, OUTPUTS_DIR, PREDICTIONS_JSON_PATH
from src.model_poisson import predict_xg, calculate_match_probabilities

MODEL_PARAMS_PATH = OUTPUTS_DIR / "model_params.json"

def load_data():
    with open(PREDICTIONS_JSON_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    with open(MODEL_PARAMS_PATH, "r", encoding="utf-8") as f:
        model = json.load(f)
        
    fixtures = pd.read_csv(FIXTURE_PATH)
    # Create match number mapping
    fixtures["match_num"] = range(1, len(fixtures) + 1)
    
    return predictions, model, fixtures

def sample_match_outcome(prob_matrix):
    """Samples exact goals for Team A and Team B based on prob matrix."""
    flat_probs = prob_matrix.flatten()
    flat_probs = flat_probs / flat_probs.sum()
    sampled_idx = np.random.choice(len(flat_probs), p=flat_probs)
    goals_a, goals_b = np.unravel_index(sampled_idx, prob_matrix.shape)
    return int(goals_a), int(goals_b)

def simulate_tournament(runs=10000):
    print(f"Running Monte Carlo tournament simulation with {runs} runs...")
    
    if not PREDICTIONS_JSON_PATH.exists() or not MODEL_PARAMS_PATH.exists():
        print("Error: Run 'predict' first to generate model_params.json and predictions.json")
        return
        
    predictions, model, fixtures = load_data()
    
    match_num_to_id = {str(row["match_num"]): row["match_id"] for _, row in fixtures.iterrows()}
    
    gs_preds = {p["match_id"]: p for p in predictions if p["match_id"].startswith("GS")}
    gs_fixtures = fixtures[fixtures["stage"].str.startswith("Group")]
    ko_fixtures = fixtures[~fixtures["stage"].str.startswith("Group")].copy()
    
    champion_counts = defaultdict(int)
    
    # Pre-calculate base matrices for group stage to speed up simulation
    gs_matrices = {}
    for match_id, p in gs_preds.items():
        xg_a = p["expected_goals_team_a"]
        xg_b = p["expected_goals_team_b"]
        res = calculate_match_probabilities(xg_a, xg_b)
        gs_matrices[match_id] = res["matrix"]
    
    print("Starting simulation loop...")
    for r in range(runs):
        if (r+1) % 100 == 0:
            print(f"\r  Run {r+1}/{runs} ...", end="", flush=True)
            
        # --- GROUP STAGE ---
        group_standings = defaultdict(lambda: defaultdict(lambda: {"pts": 0, "gd": 0, "gf": 0, "group": ""}))
        
        for _, row in gs_fixtures.iterrows():
            m_id = row["match_id"]
            team_a = row["home_team"]
            team_b = row["away_team"]
            group = row["stage"]
            
            if m_id not in gs_matrices:
                continue
            
            ga, gb = sample_match_outcome(gs_matrices[m_id])
            
            group_standings[group][team_a]["group"] = group
            group_standings[group][team_b]["group"] = group
            group_standings[group][team_a]["gf"] += ga
            group_standings[group][team_b]["gf"] += gb
            group_standings[group][team_a]["gd"] += (ga - gb)
            group_standings[group][team_b]["gd"] += (gb - ga)
            
            if ga > gb:
                group_standings[group][team_a]["pts"] += 3
            elif gb > ga:
                group_standings[group][team_b]["pts"] += 3
            else:
                group_standings[group][team_a]["pts"] += 1
                group_standings[group][team_b]["pts"] += 1

        # Rank groups
        group_winners = {}
        group_runners = {}
        third_places = []
        
        for group, teams in group_standings.items():
            # Tiebreaker: Pts -> GD -> GF -> Random coin flip
            sorted_teams = sorted(teams.items(), key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], random.random()), reverse=True)
            group_winners[group] = sorted_teams[0][0]
            group_runners[group] = sorted_teams[1][0]
            third_places.append((sorted_teams[2][0], sorted_teams[2][1]))
            
        # Top 8 third-placed teams advance
        third_places.sort(key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], random.random()), reverse=True)
        advancing_thirds = [t[0] for t in third_places[:8]]
        random.shuffle(advancing_thirds) # Heuristic random mapping for R32
        
        # --- KNOCKOUT STAGE ---
        match_results = {} # m_id -> {"winner": team, "loser": team}
        
        def resolve_team(name):
            if name.startswith("Winner Group "):
                grp = "Group " + name[-1]
                return group_winners[grp]
            if name.startswith("Runner-up Group "):
                grp = "Group " + name[-1]
                return group_runners[grp]
            if name.startswith("Third place"):
                return advancing_thirds.pop() if advancing_thirds else "Unknown"
            if name.startswith("Winner Match "):
                m_num = name.split("Winner Match ")[1]
                m_id = match_num_to_id[m_num]
                return match_results[m_id]["winner"]
            if name.startswith("Loser Match "):
                m_num = name.split("Loser Match ")[1]
                m_id = match_num_to_id[m_num]
                return match_results[m_id]["loser"]
            return name
        
        # Process chronologically
        for _, row in ko_fixtures.iterrows():
            m_id = row["match_id"]
            team_a = resolve_team(row["home_team"])
            team_b = resolve_team(row["away_team"])
            
            # Predict
            xg_a, xg_b = predict_xg(team_a, team_b, model)
            res = calculate_match_probabilities(xg_a, xg_b)
            ga, gb = sample_match_outcome(res["matrix"])
            
            if ga > gb:
                winner, loser = team_a, team_b
            elif gb > ga:
                winner, loser = team_b, team_a
            else:
                # Penalty shootout (50/50 for simplicity)
                if random.random() < 0.5:
                    winner, loser = team_a, team_b
                else:
                    winner, loser = team_b, team_a
                    
            match_results[m_id] = {"winner": winner, "loser": loser}
            
            if m_id == "FINAL_01":
                champion_counts[winner] += 1
                
    print() # clear progress line
    
    # --- AGGREGATION ---
    champion_probs = {k: v / runs for k, v in champion_counts.items()}
    # Sort descending
    champion_probs = dict(sorted(champion_probs.items(), key=lambda item: item[1], reverse=True))
    
    results = {
        "runs": runs,
        "champion_probabilities": champion_probs
    }
    
    out_path = PROCESSED_DIR / "tournament_simulation_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved simulation results to {out_path}")
    
    print("\nTop 5 Most Probable Champions:")
    for t, p in list(champion_probs.items())[:5]:
        print(f"  {t}: {p:.1%}")

if __name__ == "__main__":
    simulate_tournament(runs=10000)
