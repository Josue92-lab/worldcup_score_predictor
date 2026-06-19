import streamlit as st
import pandas as pd
import json
import altair as alt
from pathlib import Path
import sys
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIT_REPORT_PATH, FIXTURE_PATH, OUTPUTS_DIR
from src.normalize_teams import normalize_team_name

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

def load_evaluation_report():
    eval_path = OUTPUTS_DIR / "evaluation_report.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        matches = eval_data.get("evaluated_matches", [])
        
        eval_map = {}
        for m in matches:
            if m.get("match_id"):
                eval_map[str(m.get("match_id"))] = m
            
            # Fallback key
            d = str(m.get("date", "")).strip()
            ta = normalize_team_name(m.get("team_a", ""))
            tb = normalize_team_name(m.get("team_b", ""))
            fallback_key = f"{d}_{ta}_{tb}"
            eval_map[fallback_key] = m
            fallback_key_rev = f"{d}_{tb}_{ta}"
            eval_map[fallback_key_rev] = m
            
        return eval_map
    return {}

def main():
    st.set_page_config(page_title="World Cup Score Predictor", layout="wide")

    st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    h1 {
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        font-size: 1.45rem !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
    }
    h3 {
        font-size: 1.15rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    .stDataFrame {
        font-size: 0.85rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏆 World Cup Score Predictor")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Audit Report", "Fixtures"])

    with tab1:
        st.header("Match Predictions")
        
        available_reports = list(OUTPUTS_DIR.glob("*.json"))
        available_reports = [f for f in available_reports if f.name not in ["model_params.json", "evaluation_report.json", "calibration_report.json"]]
        
        if available_reports:
            # Filters block
            col_rep, col_view = st.columns([2, 2])
            with col_rep:
                selected_file = st.selectbox("Select Prediction Report", [f.name for f in available_reports])
            with col_view:
                view_mode = st.radio("View Mode", ["Compact", "Detailed"], horizontal=True)

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

                calib_factor = metadata.get("calibration_factor", 1.0)
                calib_active = metadata.get("calibrated", False)
                
                # Check for calibration_report.json
                calib_path = OUTPUTS_DIR / "calibration_report.json"
                if calib_path.exists():
                    try:
                        with open(calib_path, "r", encoding="utf-8") as f:
                            calib_data = json.load(f)
                            
                        with st.expander("Model Calibration Panel", expanded=calib_active):
                            st.write(f"**Model Mode:** {metadata.get('mode')}")
                            st.write(f"**Average Predicted Goals:** {calib_data.get('average_predicted_total_goals', 0):.2f}")
                            st.write(f"**Average Actual Goals:** {calib_data.get('average_actual_total_goals', 0):.2f}")
                            st.write(f"**Calibration Factor:** {calib_factor:.3f}")
                            
                            if calib_active:
                                st.warning(f"⚠️ **Underestimating Goals:** The base model was underestimating goal volume. Lambdas inflated by {calib_factor:.3f}x.")
                            else:
                                st.success("Model goal volume is within expected range or calibration is inactive.")
                    except:
                        pass

            predictions = [normalize_prediction(p) for p in predictions]
            valid_predictions = [p for p in predictions if p.get("team_a") and p.get("team_b")]
            
            skipped = len(predictions) - len(valid_predictions)
            if skipped > 0:
                st.warning(f"Skipped {skipped} prediction rows due to missing team fields.")
                
            # Load evaluation data to match
            eval_map = load_evaluation_report()
            actuals_loaded = len({id(m) for m in eval_map.values()})
            actuals_matched = 0

            # Assign Status & Dates
            available_dates = set()
            all_teams = set()
            all_phases = set()

            for p in valid_predictions:
                match_id = str(p.get("match_id"))
                eval_match = eval_map.get(match_id)
                
                if not eval_match:
                    p_date_str = str(p.get("date")) if p.get("date") else ""
                    ta = normalize_team_name(p.get("team_a", ""))
                    tb = normalize_team_name(p.get("team_b", ""))
                    eval_match = eval_map.get(f"{p_date_str}_{ta}_{tb}")
                    
                if eval_match:
                    actuals_matched += 1

                p_date_str = str(p.get("date")) if p.get("date") else None
                
                # Assign actuals from eval_match if available
                p["_actual_score"] = eval_match.get("actual_scoreline") if eval_match else None
                p["_top_5_hit"] = eval_match.get("top5_correct") if eval_match else None
                p["_1x2_hit"] = eval_match.get("outcome_1x2_correct") if eval_match else None

                # Infer status
                if p["_actual_score"] is not None:
                    p["_status"] = "Played"
                else:
                    if p_date_str and p_date_str == str(date.today()):
                        p["_status"] = "Today"
                    else:
                        p["_status"] = "Upcoming"

                if p_date_str:
                    available_dates.add(p_date_str)
                all_teams.add(p['team_a'])
                all_teams.add(p['team_b'])
                if p.get('phase'):
                    all_phases.add(p['phase'])

            # Debug Panel
            with st.expander("Debug: Actual Results Merge", expanded=False):
                st.write(f"**Actual results loaded:** {actuals_loaded}")
                st.write(f"**Actuals matched to selected report:** {actuals_matched}")
                st.write(f"**Actuals unmatched:** {actuals_loaded - actuals_matched}")
                st.write(f"**Selected report rows:** {len(valid_predictions)}")

            # Add more filters
            col_date, col_team, col_phase, col_status = st.columns(4)
            with col_date:
                preset_dates = st.selectbox("Date Range", ["All dates", "Today", "Next 3 days", "Played only", "Upcoming only", "Custom range"])
            with col_team:
                selected_team = st.selectbox("Filter by Team", ["All"] + sorted(list(all_teams)))
            with col_phase:
                selected_phase = st.selectbox("Phase / Group", ["All"] + sorted(list(all_phases)))
            with col_status:
                selected_status = st.selectbox("Match Status", ["All", "Played", "Upcoming", "Today"])

            # Filter Logic
            filtered_predictions = []
            today_str = str(date.today())
            next_3_str = str(date.today() + timedelta(days=3))

            custom_dates = None
            if preset_dates == "Custom range":
                custom_dates = st.date_input("Select Dates", [])

            for p in valid_predictions:
                # Team
                if selected_team != "All" and selected_team not in (p['team_a'], p['team_b']):
                    continue
                # Phase
                if selected_phase != "All" and str(p.get('phase')) != selected_phase:
                    continue
                # Status
                if selected_status != "All" and p.get("_status") != selected_status:
                    continue
                # Dates
                p_date = str(p.get('date')) if p.get('date') else ""
                
                if preset_dates == "Today" and p_date != today_str:
                    continue
                elif preset_dates == "Next 3 days" and not (today_str <= p_date <= next_3_str):
                    continue
                elif preset_dates == "Played only" and p.get("_status") != "Played":
                    continue
                elif preset_dates == "Upcoming only" and p.get("_status") != "Upcoming":
                    continue
                elif preset_dates == "Custom range" and custom_dates:
                    if len(custom_dates) == 1:
                        if p_date != str(custom_dates[0]):
                            continue
                    elif len(custom_dates) == 2:
                        if not (str(custom_dates[0]) <= p_date <= str(custom_dates[1])):
                            continue

                filtered_predictions.append(p)

            st.write(f"**Showing {len(filtered_predictions)} matches**")

            if not filtered_predictions:
                st.info("No matches match the selected filters.")
            else:
                # Summary Table
                if view_mode == "Compact":
                    summary_rows = []
                    for p in filtered_predictions:
                        top_pred = ""
                        top_prob = ""
                        if p.get("top_5_scorelines"):
                            top = p["top_5_scorelines"][0]
                            top_pred = f"{p['team_a']} {top.get('scoreline', '?-?').split('-')[0]} - {top.get('scoreline', '?-?').split('-')[-1]} {p['team_b']}"
                            top_prob = f"{float(top.get('probability', 0)) * 100:.2f}%"

                        summary_rows.append({
                            "Date": p.get("date", ""),
                            "Phase / Group": p.get("phase", ""),
                            "Match": f"{p['team_a']} vs {p['team_b']}",
                            "Team A Win %": f"{float(p.get('home_win_probability', 0)):.1%}",
                            "Draw %": f"{float(p.get('draw_probability', 0)):.1%}",
                            "Team B Win %": f"{float(p.get('away_win_probability', 0)):.1%}",
                            "Top Prediction": top_pred,
                            "Top Prediction %": top_prob,
                            "Actual Score": p.get("_actual_score") or "Pending",
                            "Top 5 Hit": "✅" if p.get("_top_5_hit") else ("❌" if p.get("_top_5_hit") is False else ""),
                            "1X2 Hit": "✅" if p.get("_1x2_hit") else ("❌" if p.get("_1x2_hit") is False else "")
                        })

                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

                # Expanders (or detailed blocks)
                for idx, p in enumerate(filtered_predictions):
                    expand_by_default = (idx == 0 and len(filtered_predictions) <= 5)
                    top_pred = ""
                    if p.get("top_5_scorelines"):
                        top = p["top_5_scorelines"][0]
                        top_pred = f"{p['team_a']} {top.get('scoreline', '?-?').split('-')[0]} - {top.get('scoreline', '?-?').split('-')[-1]} {p['team_b']}"

                    expander_title = f"{p.get('date', '')} | {p['team_a']} vs {p['team_b']} | Top: {top_pred}"
                    if p.get("_actual_score"):
                        expander_title += f" | Actual: {p['_actual_score']}"

                    if view_mode == "Compact":
                        container = st.expander(expander_title, expanded=expand_by_default)
                    else:
                        st.subheader(f"Match: {p['team_a']} vs {p['team_b']}")
                        container = st.container()

                    with container:
                        st.write(f"**{p['team_a_label']}:** {p['team_a']}  \n"
                                 f"**{p['team_b_label']}:** {p['team_b']}  \n"
                                 f"**Venue:** {p['venue']}  \n"
                                 f"**Neutral:** {p['neutral']}")
                        st.write(f"**Match ID:** {p.get('match_id', 'N/A')} | **Phase:** {p.get('phase', 'N/A')} | **Date:** {p.get('date', 'N/A')}")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"{p['team_a']} Win Prob", f"{float(p.get('home_win_probability', 0)):.1%}")
                        col2.metric("Draw Prob", f"{float(p.get('draw_probability', 0)):.1%}")
                        col3.metric(f"{p['team_b']} Win Prob", f"{float(p.get('away_win_probability', 0)):.1%}")
                        
                        st.write(f"**Scoreline probability: {p['team_a']} vs {p['team_b']}**")
                        scoreline_data = pd.DataFrame(p['top_5_scorelines'])
                        if not scoreline_data.empty:
                            actual_score = p.get("_actual_score")
                            actual_in_top5 = False
                            
                            if 'display_scoreline' not in scoreline_data.columns:
                                # Backward compatibility for bare scoreline format
                                scoreline_data['display_scoreline'] = scoreline_data.apply(
                                    lambda r: f"{p['team_a']} {r.get('scoreline', '?-?').split('-')[0]} - {r.get('scoreline', '?-?').split('-')[-1]} {p['team_b']}", axis=1
                                )
                            
                            scoreline_data["probability_pct"] = scoreline_data["probability"].astype(float) * 100
                            
                            if actual_score:
                                scoreline_data["Result"] = scoreline_data["scoreline"].apply(
                                    lambda x: "Actual result" if x == actual_score else ""
                                )
                                if "Actual result" in scoreline_data["Result"].values:
                                    actual_in_top5 = True
                            else:
                                scoreline_data["Result"] = ""
                                
                            # Dataframe presentation
                            display_df = scoreline_data[["display_scoreline", "probability_pct", "Result"]].rename(columns={
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
                                .properties(height=190)
                            )

                            st.altair_chart(chart, use_container_width=True)
                            
                            if actual_score and not actual_in_top5:
                                actual_score_display = f"{p['team_a']} {actual_score.split('-')[0]} - {actual_score.split('-')[1]} {p['team_b']}"
                                st.info(f"Actual result: {actual_score_display} was not in the model's top 5 scorelines.")
                        
                        with st.expander("Explanation & Data Quality", expanded=False):
                            st.json(p.get('explanation', {}))
                            if p.get('data_quality', {}).get('warnings'):
                                for w in p.get('data_quality', {}).get('warnings', []):
                                    st.warning(w)
                        if view_mode == "Detailed":
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
