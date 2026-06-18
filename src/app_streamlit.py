import streamlit as st
import pandas as pd
import json
import altair as alt
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIT_REPORT_PATH, FIXTURE_PATH, OUTPUTS_DIR

def get_first(row: dict, *keys: str, default=None):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default

def normalize_prediction(row: dict) -> dict:
    normalized = dict(row)
    normalized["home_team"] = get_first(row, "home_team", "team_a", "team1", "team_home", "home")
    normalized["away_team"] = get_first(row, "away_team", "team_b", "team2", "team_away", "away")
    
    normalized["team_a"] = get_first(row, "team_a", "home_team", "team1", "team_home", "home")
    normalized["team_b"] = get_first(row, "team_b", "away_team", "team2", "team_away", "away")
    normalized["team_a_label"] = get_first(row, "team_a_label", default="Team A / Listed first")
    normalized["team_b_label"] = get_first(row, "team_b_label", default="Team B / Listed second")
    normalized["neutral"] = get_first(row, "neutral", default=True)
    normalized["venue"] = get_first(row, "venue", "venue_code", default="Unknown")

    normalized["top_5_scorelines"] = get_first(row, "top_5_scorelines", "top_scores", "scorelines", default=[])
    normalized["home_win_probability"] = get_first(row, "home_win_probability", "team_a_win_probability")
    normalized["away_win_probability"] = get_first(row, "away_win_probability", "team_b_win_probability")
    return normalized

def main():
    st.set_page_config(page_title="World Cup Score Predictor", layout="wide")

    st.title("🏆 World Cup Score Predictor")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Audit Report", "Fixtures"])

    with tab1:
        st.header("Match Predictions")
        
        available_reports = list(OUTPUTS_DIR.glob("*.json"))
        available_reports = [f for f in available_reports if f.name not in ["model_params.json", "evaluation_report.json"]]
        
        if available_reports:
            selected_file = st.selectbox("Select Prediction Report", [f.name for f in available_reports])
            file_path = OUTPUTS_DIR / selected_file
            
            with open(file_path, "r", encoding="utf-8") as f:
                raw_predictions = json.load(f)
                
            if isinstance(raw_predictions, dict):
                metadata = raw_predictions.get("metadata", {})
                predictions = (
                    raw_predictions.get("predictions")
                    or raw_predictions.get("matches")
                    or raw_predictions.get("data")
                    or []
                )
            elif isinstance(raw_predictions, list):
                metadata = {}
                predictions = raw_predictions
            else:
                metadata = {}
                predictions = []

            if not isinstance(predictions, list):
                st.error("Unsupported JSON format: 'predictions' is not a list.")
                predictions = []
                
            if metadata:
                st.info(f"**Mode:** {metadata.get('mode')} | **As Of Date:** {metadata.get('as_of_date')} | **Train Cutoff:** {metadata.get('train_cutoff')}")

            predictions = [normalize_prediction(p) for p in predictions]
            valid_predictions = [p for p in predictions if p.get("team_a") and p.get("team_b")]
            
            skipped = len(predictions) - len(valid_predictions)
            if skipped > 0:
                st.warning(f"Skipped {skipped} prediction rows due to missing team fields.")
                
            predictions = valid_predictions

            if not predictions:
                st.warning("No valid predictions available.")
            else:
                # Let user filter by team
                all_teams = set()
                for p in predictions:
                    all_teams.add(p['team_a'])
                    all_teams.add(p['team_b'])
                
                selected_team = st.selectbox("Filter by Team", ["All"] + sorted(list(all_teams)))
                
                for p in predictions:
                    if selected_team != "All" and selected_team not in (p['team_a'], p['team_b']):
                        continue
                        
                    st.subheader(f"Match: {p['team_a']} vs {p['team_b']}")
                    st.write(f"**{p['team_a_label']}:** {p['team_a']}  \n"
                             f"**{p['team_b_label']}:** {p['team_b']}  \n"
                             f"**Venue:** {p['venue']}  \n"
                             f"**Neutral:** {p['neutral']}")
                    st.write(f"**Match ID:** {p.get('match_id', 'N/A')} | **Phase:** {p.get('phase', 'N/A')} | **Date:** {p.get('date', 'N/A')}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"{p['team_a']} Win Prob", f"{p['home_win_probability']:.1%}")
                    col2.metric("Draw Prob", f"{p['draw_probability']:.1%}")
                    col3.metric(f"{p['team_b']} Win Prob", f"{p['away_win_probability']:.1%}")
                    
                    st.write(f"**Scoreline probability: {p['team_a']} vs {p['team_b']}**")
                    scoreline_data = pd.DataFrame(p['top_5_scorelines'])
                    if not scoreline_data.empty:
                        if 'display_scoreline' not in scoreline_data.columns:
                            # Backward compatibility for bare scoreline format
                            scoreline_data['display_scoreline'] = scoreline_data.apply(
                                lambda r: f"{p['team_a']} {r.get('scoreline', '?-?').split('-')[0]} - {r.get('scoreline', '?-?').split('-')[-1]} {p['team_b']}", axis=1
                            )
                        
                        scoreline_data["probability_pct"] = scoreline_data["probability"].astype(float) * 100
                        
                        # Dataframe presentation
                        display_df = scoreline_data[["display_scoreline", "probability_pct"]].rename(columns={
                            "display_scoreline": "Scoreline",
                            "probability_pct": "Probability"
                        })
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Probability": st.column_config.NumberColumn(
                                    "Probability",
                                    format="%.2f%%"
                                )
                            }
                        )
                        
                        # Plot displaying probabilities vs scorelines
                        chart_df = scoreline_data.copy()
                        chart_df["Probability (%)"] = chart_df["probability_pct"]
                        chart_df["Scoreline"] = chart_df["display_scoreline"]

                        chart = (
                            alt.Chart(chart_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("Probability (%):Q", title="Probability (%)"),
                                y=alt.Y("Scoreline:N", sort="-x", title="Scoreline"),
                                tooltip=[
                                    alt.Tooltip("Scoreline:N", title="Scoreline"),
                                    alt.Tooltip("Probability (%):Q", title="Probability", format=".2f")
                                ],
                            )
                            .properties(height=260)
                        )

                        st.altair_chart(chart, use_container_width=True)
                    
                    with st.expander("Explanation & Data Quality"):
                        st.json(p['explanation'])
                        if p['data_quality']['warnings']:
                            for w in p['data_quality']['warnings']:
                                st.warning(w)
                    st.divider()
        else:
            st.info("Run `python -m src.cli backtest` to generate predictions.")

    with tab2:
        st.header("Data Audit Report")
        if AUDIT_REPORT_PATH.exists():
            with open(AUDIT_REPORT_PATH, 'r', encoding='utf-8') as f:
                audit = json.load(f)
            
            status_color = "green" if audit.get("status") == "pass" else "orange" if audit.get("status") == "needs_review" else "red"
            st.markdown(f"**Status:** <span style='color:{status_color}'>{audit.get('status', 'unknown').upper()}</span>", unsafe_allow_html=True)
            
            st.write("### Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Fixtures", audit.get("fixture_rows", 0))
            col2.metric("Venues", audit.get("venues_rows", 0))
            col3.metric("Squad Teams", audit.get("squad_teams", 0))
            col4.metric("Squad Players", audit.get("squad_players", 0))
            
            if audit.get("warnings"):
                st.error("Warnings detected:")
                for w in audit["warnings"]:
                    st.write(f"- {w}")
                    
            st.write("### Full JSON")
            st.json(audit)
        else:
            st.info("Run `python -m src.cli audit` to generate the audit report.")

    with tab3:
        st.header("Raw Fixtures")
        if FIXTURE_PATH.exists():
            fixtures = pd.read_csv(FIXTURE_PATH)
            st.dataframe(fixtures)
        else:
            st.warning("No fixture.csv found.")

if __name__ == "__main__":
    main()
