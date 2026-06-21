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


def extract_predictions_from_report(raw: dict | list) -> list:
    """Centralize the fragile report shape handling used by the dashboard."""
    if isinstance(raw, dict):
        return (
            raw.get("predictions")
            or raw.get("matches")
            or raw.get("data")
            or []
        )
    elif isinstance(raw, list):
        return raw
    return []


def load_selected_predictions(selected_file: str) -> tuple[dict, list]:
    """Load one of the generated report files and return (metadata, predictions_list)."""
    file_path = OUTPUTS_DIR / selected_file
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    predictions = extract_predictions_from_report(raw)
    if not isinstance(predictions, list):
        st.error("Unsupported JSON format: 'predictions' is not a list.")
        predictions = []
    return metadata, predictions


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


def load_full_evaluation_report():
    """Load the full evaluation report safely. Return {} if missing or invalid."""
    eval_path = OUTPUTS_DIR / "evaluation_report.json"
    if not eval_path.exists():
        return {}
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_calibration_report():
    """Load the full calibration report safely. Return {} if missing or invalid."""
    calib_path = OUTPUTS_DIR / "calibration_report.json"
    if not calib_path.exists():
        return {}
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_calibration_factor(metadata: dict, calib_data: dict = None) -> float:
    """Resolve single factor from metadata (new key or fav/und avg) or calib global. Default 1.0.
    Used for display and warning guards.
    """
    m = metadata or {}
    cf = m.get("calibration_factor")
    if cf is not None:
        try:
            return round(float(cf), 3)
        except Exception:
            pass
    fav = m.get("calibration_factor_fav")
    und = m.get("calibration_factor_und")
    if fav is not None and und is not None:
        try:
            return round((float(fav) + float(und)) / 2.0, 3)
        except Exception:
            pass
    if calib_data:
        g = calib_data.get("global_calibration_factor")
        if g is not None:
            try:
                return round(float(g), 3)
            except Exception:
                pass
    return 1.0


def load_diagnostics_summary():
    """Return a safe summary dict with hit rates, goal stats, concentration info.
    Never crashes on missing fields.
    """
    eval_report = load_full_evaluation_report()
    calib_report = load_calibration_report()

    diag = {
        "top1_rate": None,
        "top5_rate": None,
        "one_x2_rate": None,
        "matches_evaluated": 0,
        "avg_predicted_goals": None,
        "avg_actual_goals": None,
        "calibration_factor": None,
        "top1_distribution": {},
        "high_score_miss_count": None,
        "one_one_count": 0,
        "one_one_pct": None,
        "source": "evaluation_report",
    }

    if eval_report:
        diag["matches_evaluated"] = eval_report.get("matches_evaluated", 0) or 0
        diag["top1_rate"] = eval_report.get("exact_score_top1_rate")
        diag["top5_rate"] = eval_report.get("exact_score_top5_rate")
        diag["one_x2_rate"] = eval_report.get("outcome_1x2_rate")

        d = eval_report.get("diagnostics", {}) or {}
        diag["avg_predicted_goals"] = d.get("predicted_total_goals_avg")
        diag["avg_actual_goals"] = d.get("actual_total_goals_avg")
        diag["calibration_factor"] = d.get("calibration_factor")
        diag["top1_distribution"] = d.get("top1_scoreline_distribution", {}) or {}
        diag["high_score_miss_count"] = d.get("high_score_miss_count")

        top1_dist = diag["top1_distribution"]
        total_top1 = sum(top1_dist.values()) if top1_dist else 0
        matches_eval = diag.get("matches_evaluated", 0) or 0
        one_one = top1_dist.get("1-1", 0) or 0
        # Prevent impossible: count cannot exceed evaluated matches (e.g. dist over full report vs matched)
        if matches_eval > 0 and total_top1 > matches_eval:
            # Inconsistent source data; fall back to safe
            diag["top1_distribution"] = {}
            diag["one_one_count"] = 0
            diag["one_one_pct"] = None
        else:
            diag["one_one_count"] = one_one
            if total_top1 > 0:
                diag["one_one_pct"] = one_one / total_top1

    # Prefer calibration numbers if they are more recent / relevant
    if calib_report:
        if diag["avg_predicted_goals"] is None:
            diag["avg_predicted_goals"] = calib_report.get("average_predicted_total_goals")
        if diag["avg_actual_goals"] is None:
            diag["avg_actual_goals"] = calib_report.get("average_actual_total_goals")
        if diag.get("calibration_factor") in (None, 1.0):
            gcf = calib_report.get("global_calibration_factor")
            if gcf is not None:
                try:
                    if float(gcf) != 1.0:
                        diag["calibration_factor"] = float(gcf)
                except Exception:
                    if diag["calibration_factor"] is None:
                        diag["calibration_factor"] = gcf

    return diag


def get_educational_markdown() -> str:
    """Concise plain-language explanation of the predictions."""
    return """
**This is a probabilistic educational model, not a betting tool or oracle.**

- **Exact scoreline vs outcome**: A scoreline like `1-1` is one specific final score. The three percentages (`Team A Win %`, `Draw %`, `Team B Win %`) are sums across *all* possible scorelines that produce that broad result.
- **1-1 can be most likely exact score even when one team is favored**: Many small probabilities for different winning scores can add up to more total probability than the single most likely draw.
- **Team A / Team B**: These are simply the first-listed and second-listed teams in the fixture (neutral venue for World Cup matches unless noted). No home advantage is assumed.
- **Top Prediction**: The single most probable exact scoreline according to the model.
- **Top 5 Hit**: ✅ if the actual final score was among the model's top 5 most probable exact scorelines.
- **1X2 Hit**: ✅ if the model correctly predicted the broad outcome (Team A win, draw, or Team B win) by taking the highest of the three aggregated percentages.
- **Pending**: No actual result has been matched yet from the results feed.
"""


def get_mode_banner(metadata: dict) -> str:
    """Return a clear banner message for backtest vs live, or empty string."""
    mode = (metadata or {}).get("mode") or "unknown"
    as_of = (metadata or {}).get("as_of_date")
    train_cutoff = (metadata or {}).get("train_cutoff")

    if mode == "backtest":
        cutoff = train_cutoff or "Unknown"
        return (
            f"**Backtest mode**: This evaluates what the model would have predicted using only data before the training cutoff ({cutoff}). "
            "It must not use later World Cup results for training or calibration. Use this to assess historical model skill honestly."
        )
    elif mode == "live":
        asof = as_of or "Unknown"
        return (
            f"**Live mode** (as of {asof}): This may use already-played World Cup results up to the as-of date to detect goal-volume trends and apply a calibration factor to future predictions. "
            "Live numbers should not be interpreted as a retroactive claim of past predictive performance."
        )
    else:
        return ""


def format_explanation_bullets(p: dict, report_metadata: dict = None) -> str:
    """Return human-readable markdown bullets for explanation & data quality.
    Safe for any missing keys.
    """
    lines = []
    exp = p.get("explanation") or {}
    dq = p.get("data_quality") or {}
    meta = report_metadata or {}

    # Expected goals (top level on prediction)
    xg_a = p.get("expected_goals_team_a")
    xg_b = p.get("expected_goals_team_b")
    if xg_a is not None and xg_b is not None:
        lines.append(f"- **Expected goals**: {float(xg_a):.2f} – {float(xg_b):.2f}")

    # Strengths from explanation
    att_a = exp.get("team_a_attack_strength")
    att_b = exp.get("team_b_attack_strength")
    def_a = exp.get("team_a_defense_strength")
    def_b = exp.get("team_b_defense_strength")
    if att_a is not None:
        lines.append(f"- **Team A attack strength**: {float(att_a):.2f}× historical average")
    if att_b is not None:
        lines.append(f"- **Team B attack strength**: {float(att_b):.2f}× historical average")
    if def_a is not None:
        lines.append(f"- **Team A defense strength**: {float(def_a):.2f}× (lower is better)")
    if def_b is not None:
        lines.append(f"- **Team B defense strength**: {float(def_b):.2f}× (lower is better)")

    # Squad
    sq_a = exp.get("team_a_squad_strength") or exp.get("team_a_sqi")
    sq_b = exp.get("team_b_squad_strength") or exp.get("team_b_sqi")
    if sq_a is not None and sq_b is not None:
        try:
            lines.append(f"- **Squad strength (A vs B)**: {float(sq_a):,.0f} vs {float(sq_b):,.0f}")
        except Exception:
            pass

    # Main factors
    factors = exp.get("main_factors") or []
    if factors:
        lines.append("- **Main factors**: " + " | ".join(str(f) for f in factors))

    # Data quality
    status = dq.get("squad_match_status")
    rate = dq.get("player_kaggle_match_rate")
    if status:
        rate_str = f" (match rate ~{float(rate):.0%})" if rate is not None else ""
        lines.append(f"- **Squad data status**: {status}{rate_str}")
    if rate is not None and rate < 0.6:
        lines.append("- **Note**: Low player data coverage for one or both teams — squad context is limited.")

    # Calibration status from report metadata
    if meta.get("calibrated"):
        cf = meta.get("calibration_factor_fav") or meta.get("calibration_factor")
        lines.append(f"- **Live calibration active**: goal rates adjusted (factor ≈ {cf:.3f})" if cf else "- **Live calibration active**")

    # Warnings
    warnings = dq.get("warnings") or []
    if warnings:
        for w in warnings:
            lines.append(f"- ⚠️ {w}")

    if not lines:
        lines.append("- Baseline Poisson model (historical attack/defense strengths). Limited additional context available.")

    return "\n".join(lines)


def get_1_1_concentration_note(diag: dict, predictions: list) -> str:
    """Return a warning string if 1-1 concentration is high, else ''."""
    one_one_pct = diag.get("one_one_pct")
    one_one_count = diag.get("one_one_count", 0)
    matches = diag.get("matches_evaluated", 0) or len(predictions) or 0

    # Guard against any residual inconsistent count
    if matches > 0 and one_one_count > matches:
        one_one_count = 0
        one_one_pct = None

    # Also compute from current predictions if eval dist looks inflated
    if not predictions:
        pred_one_one = 0
    else:
        pred_one_one = sum(
            1 for p in predictions
            if p.get("top_5_scorelines") and p["top_5_scorelines"][0].get("scoreline") == "1-1"
        )

    total_for_pct = matches or len(predictions) or 1
    effective_pct = one_one_pct if (one_one_pct is not None and one_one_pct > 0.3) else (pred_one_one / total_for_pct if total_for_pct else 0)

    if effective_pct and effective_pct > 0.4:
        pct_str = f"{effective_pct*100:.0f}%"
        return (
            f"⚠️ The model is currently heavily concentrated on **1-1** as the top prediction ({pct_str} of recent top-1s). "
            "This is typical of a conservative Poisson baseline on lower-scoring historical international data and often underestimates high-scoring matches. "
            "Live calibration can partially compensate for future matches but should never be applied retroactively to backtest results to claim better historical accuracy."
        )
    if one_one_count > 20 and matches and one_one_count / matches > 0.5:
        return (
            "⚠️ Very high 1-1 concentration detected in evaluation diagnostics. "
            "The baseline model may be under-dispersed relative to actual tournament scoring."
        )
    return ""


def get_data_credits_markdown() -> str:
    return """
**Data credits**

- Historical international results & live actuals: [martj42/international_results](https://github.com/martj42/international_results)
- Player valuations, appearances & squad context: [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- Official national team squads: FIFA SquadLists PDF (parsed)
- Fixtures & venues: project `fixture.csv` / `venues.csv`

Dataset freshness varies. Player-level data can lag real-world transfers/injuries. Match results are updated independently.
"""


def _render_calibration_panel(metadata: dict, calib_factor: float, calib_active: bool) -> None:
    """Render the optional calibration expander (internal helper)."""
    calib_path = OUTPUTS_DIR / "calibration_report.json"
    if calib_path.exists():
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)

            effective_factor = _resolve_calibration_factor(metadata, calib_data)
            mode = (metadata or {}).get("mode") or "unknown"
            with st.expander("Model Calibration Panel", expanded=calib_active):
                st.write(f"**Model Mode:** {mode}")
                st.write(f"**Average Predicted Goals:** {calib_data.get('average_predicted_total_goals', 0):.2f}")
                st.write(f"**Average Actual Goals:** {calib_data.get('average_actual_total_goals', 0):.2f}")
                st.write(f"**Calibration Factor:** {effective_factor:.3f}")

                if effective_factor > 1.001:
                    st.warning(f"⚠️ **Underestimating Goals:** The base model was underestimating goal volume. Lambdas inflated by {effective_factor:.3f}x.")
                elif calib_active:
                    st.info("The model is underestimating goal volume, but no effective lambda inflation is currently applied.")
                else:
                    st.success("Model goal volume is within expected range or calibration is inactive.")

                st.caption("Diagnostics source: calibration_report.json (backtest predictions matched to actuals using rolling window).")
                if mode == "live":
                    st.caption("Selected report is live_predictions.json; calibration (if >1) applies only to future-match lambdas.")
        except:
            pass


def render_model_diagnostics(metadata: dict, diag: dict, selected_mode: str) -> None:
    """Render the model diagnostics panel (extracted for readability)."""
    base_title = "📊 Model diagnostics (from latest evaluation/calibration)"
    if selected_mode == "live":
        base_title += " — backtest evaluation (independent of selected live report)"
    with st.expander(base_title, expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches evaluated", diag.get("matches_evaluated") or "Not available")
        top1 = diag.get("top1_rate")
        c2.metric("Top-1 exact hit rate", f"{top1:.1%}" if isinstance(top1, (int, float)) else "Not available")
        top5 = diag.get("top5_rate")
        c3.metric("Top-5 exact hit rate", f"{top5:.1%}" if isinstance(top5, (int, float)) else "Not available")
        one_x2 = diag.get("one_x2_rate")
        c4.metric("1X2 hit rate", f"{one_x2:.1%}" if isinstance(one_x2, (int, float)) else "Not available")

        c5, c6, c7 = st.columns(3)
        pred_g = diag.get("avg_predicted_goals")
        c5.metric("Avg predicted total goals", f"{float(pred_g):.2f}" if isinstance(pred_g, (int, float)) else "Not available")
        act_g = diag.get("avg_actual_goals")
        c6.metric("Avg actual total goals", f"{float(act_g):.2f}" if isinstance(act_g, (int, float)) else "Not available")
        cf = diag.get("calibration_factor")
        c7.metric("Calibration factor", f"{float(cf):.3f}" if isinstance(cf, (int, float)) else "Not available")

        # Most common top prediction + concentration
        top1_dist = diag.get("top1_distribution") or {}
        matches_eval = diag.get("matches_evaluated", 0) or 0
        if top1_dist:
            most_common, count = max(top1_dist.items(), key=lambda kv: kv[1])
            if matches_eval > 0 and count > matches_eval:
                st.write("**Most common top-1 prediction**: Not available")
                st.caption("⚠️ Diagnostics data inconsistent (count exceeds evaluated matches); using latest evaluation only.")
            else:
                st.write(f"**Most common top-1 prediction**: {most_common} ({count}×)")
        else:
            st.write("**Most common top-1 prediction**: Not available")

        st.caption("See warning below the filters for 1-1 concentration note when applicable.")
        if selected_mode == "live":
            st.caption("Note: Hit rates / top-1 dist / avgs / calibration factor shown here are from backtest evaluation (base model, for honesty). Live report applies calibration only to future predictions. See Model Calibration Panel for windowed actual/predicted avgs and applied factor.")


def render_educational_panels() -> None:
    """Render the static educational / trust expanders (kept at top of Dashboard)."""
    with st.expander("📖 How to read these predictions", expanded=True):
        st.markdown(get_educational_markdown())

    with st.expander("Trust and limitations", expanded=False):
        st.markdown("""
This app is a reproducible educational probability model, not a betting tool or oracle. Predictions are generated from historical international match rates using a Poisson model with optional Dixon-Coles adjustment and live goal-volume calibration. The model exposes its own performance through backtesting and actual-result comparison.

It does not know current injuries, confirmed starting lineups, tactical plans, weather, motivation, or last-minute news. Player and squad data may lag behind real-world changes. Use the app to understand probability, uncertainty, and model calibration — not as a guarantee of match results.
""")


def inject_custom_css() -> None:
    """Inject custom CSS using Barlow (Google Fonts) for a professional, analytical dashboard feel.
    Icons (Material Symbols etc.) are protected so they do not render as text labels.
    Typography hierarchy is normalized for readability.
    Monospace for code/JSON. Dataframe internals left mostly default.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=swap');

        /* Base font for app container (avoid blanket application to icons/spans) */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            font-family: 'Barlow', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 16px;
        }

        /* Headings use Barlow with controlled weights */
        h1, h2, h3, h4, h5, h6,
        .stTitle, .stHeader {
            font-family: 'Barlow', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.005em;
        }

        h2, h3 {
            font-weight: 500 !important;
        }

        /* Educational / markdown text (selective) */
        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li {
            font-family: 'Barlow', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-weight: 400;
            line-height: 1.55;
        }

        /* Sidebar text */
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] label {
            font-family: 'Barlow', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Controls: buttons, selects, tabs, metrics, expanders (labels only where needed) */
        .stButton button,
        .stSelectbox label,
        .stMultiSelect label,
        .stRadio label,
        .stCheckbox label,
        .stTabs [role="tab"],
        .stMetric label,
        [data-testid="stExpander"] summary {
            font-family: 'Barlow', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            font-weight: 500;
        }

        /* Metric values keep a clean look (size already controlled) */
        div[data-testid="stMetricValue"] {
            font-family: 'Barlow', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Protect icon fonts (Material Symbols / icons used by Streamlit) */
        .material-icons,
        .material-symbols-outlined,
        .material-symbols-rounded,
        .material-symbols-sharp,
        [class*="material-icons"],
        [class*="material-symbols"],
        [data-testid*="Icon"],
        [data-testid*="icon"],
        svg,
        i,
        .st-emotion-cache * [class*="icon"] {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', 'Segoe UI Symbol' !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-feature-settings: normal !important;
        }

        /* Code / raw JSON / preformatted must stay monospace */
        code, pre, kbd, samp,
        .stCodeBlock,
        .stJson,
        [data-testid="stJson"] {
            font-family: 'Source Code Pro', Consolas, Monaco, 'Courier New', monospace !important;
        }

        /* DataFrame: keep size but do not force Barlow on table cells for readability */
        .stDataFrame {
            font-size: 0.85rem !important;
        }
        .stDataFrame table,
        .stDataFrame td,
        .stDataFrame th,
        .stDataFrame * {
            font-family: inherit;
        }

        /* Existing size/layout rules preserved */
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="World Cup Score Predictor", layout="wide")

    # Inject Barlow font (styling only)
    inject_custom_css()

    st.title("🏆 World Cup Score Predictor")

    # Sidebar controls (debug + credits access)
    with st.sidebar:
        st.header("Controls")
        show_debug = st.checkbox("Show debug diagnostics", value=False, help="For developers: shows internal actuals merge stats.")
        st.markdown("---")
        st.markdown(get_data_credits_markdown())

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Audit Report", "Fixtures"])

    with tab1:
        st.header("Match Predictions")

        render_educational_panels()
        
        available_reports = list(OUTPUTS_DIR.glob("*.json"))
        available_reports = [f for f in available_reports if f.name not in ["model_params.json", "evaluation_report.json", "calibration_report.json"]]
        
        if available_reports:
            # Filters block
            col_rep, col_view = st.columns([2, 2])
            with col_rep:
                selected_file = st.selectbox("Select Prediction Report", [f.name for f in available_reports])
            with col_view:
                view_mode = st.radio("View Mode", ["Compact", "Detailed"], horizontal=True)

            metadata, predictions = load_selected_predictions(selected_file)
                
            if metadata:
                st.info(f"**Mode:** {metadata.get('mode')} | **As Of Date:** {metadata.get('as_of_date')} | **Train Cutoff:** {metadata.get('train_cutoff')}")

                # 2. Visible backtest vs live banner
                banner = get_mode_banner(metadata)
                if banner:
                    if metadata.get("mode") == "backtest":
                        st.success(banner)
                    else:
                        st.info(banner)

                calib_active = metadata.get("calibrated", False) if metadata else False
                calib_report_for_factor = load_calibration_report()
                calib_factor = _resolve_calibration_factor(metadata, calib_report_for_factor)
                _render_calibration_panel(metadata, float(calib_factor), calib_active)

            # 6. Compact Model diagnostics panel (always safe)
            diag = load_diagnostics_summary()
            selected_mode = (metadata or {}).get("mode", "unknown")
            render_model_diagnostics(metadata, diag, selected_mode)

            predictions = [normalize_prediction(p) for p in predictions]
            valid_predictions = [p for p in predictions if p.get("team_a") and p.get("team_b")]
            
            skipped = len(predictions) - len(valid_predictions)
            if skipped > 0:
                st.warning(f"Skipped {skipped} prediction rows due to missing team fields.")

            # 7. 1-1 concentration note (computed from actual predictions + diagnostics)
            conc_note = get_1_1_concentration_note(diag, valid_predictions)
            if conc_note:
                st.warning(conc_note)
                
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

            # Debug Panel - hidden by default, controlled from sidebar
            if show_debug:
                with st.expander("Debug: Actual Results Merge", expanded=False):
                    st.write(f"**Actual results loaded:** {actuals_loaded}")
                    st.write(f"**Actuals matched to selected report:** {actuals_matched}")
                    st.write(f"**Actuals unmatched:** {actuals_loaded - actuals_matched}")
                    st.write(f"**Selected report rows:** {len(valid_predictions)}")
                # Also surface concentration from the actual loaded predictions
                conc_note2 = get_1_1_concentration_note(diag, valid_predictions)
                if conc_note2:
                    st.caption(conc_note2)

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
                            # Readable version (primary)
                            bullets = format_explanation_bullets(p, metadata)
                            st.markdown(bullets)

                            # Warnings already in bullets, but surface again if present
                            dq = p.get('data_quality', {}) or {}
                            if dq.get('warnings'):
                                for w in dq.get('warnings', []):
                                    st.warning(w)

                            # 5. Raw JSON hidden by default
                            if st.checkbox("Show raw JSON", key=f"show_raw_{p.get('match_id', idx)}"):
                                st.json({"explanation": p.get('explanation', {}), "data_quality": dq})
                        if view_mode == "Detailed":
                            st.divider()

            # Data credits footer inside dashboard (once, after match list)
            st.markdown("---")
            st.caption("Data sources: martj42/international_results • davidcariboo/player-scores • FIFA SquadLists PDF • project fixtures. See sidebar for full credits.")
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
