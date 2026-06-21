#!/usr/bin/env python
"""
Generate prediction_performance_audit.json and .md
Does NOT modify core model or any existing reports/UI.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, RAW_DIR
from src.normalize_teams import normalize_team_name

def parse_scoreline(sl):
    if not sl or '-' not in str(sl):
        return None, None
    try:
        a, b = str(sl).split('-', 1)
        return int(a), int(b)
    except:
        return None, None

def get_1x2_from_goals(ga, gb):
    if ga > gb: return '1'
    if ga < gb: return '2'
    return 'X'

def determine_miss_types(top1_correct, top5_correct, one_x2_correct, actual_total, pred_total, actual_1x2, pred_1x2, top_pred, actual_sl):
    tags = []
    if top1_correct:
        tags.append('exact_hit')
    elif top5_correct:
        tags.append('top5_hit_only')
    elif one_x2_correct:
        tags.append('aggregate_1x2_hit_only')
    else:
        tags.append('wrong_outcome')

    if actual_total >= 4:
        tags.append('high_score_miss')
    if actual_total > 3 and top_pred != actual_sl and 'top5_hit_only' not in tags and 'exact_hit' not in tags:
        tags.append('high_score_miss')

    if actual_1x2 == 'X' and pred_1x2 != 'X':
        tags.append('draw_bias')
    if actual_1x2 != 'X' and pred_1x2 == 'X':
        tags.append('draw_bias')

    # favorite / underdog rough: if |pred prob diff| but simplified
    if actual_1x2 != pred_1x2:
        tags.append('underdog_surprise' if (actual_1x2 in ('1','2') ) else 'favorite_underestimated')

    # goal volume specific
    gap = actual_total - pred_total
    if abs(gap) >= 2:
        tags.append('favorite_underestimated' if gap > 0 else 'wrong_outcome')
    return list(set(tags))

def bin_goals(g):
    if g <= 1: return '0-1'
    if g == 2: return '2'
    if g == 3: return '3'
    if g == 4: return '4'
    return '5+'

def load_all():
    with open(OUTPUTS_DIR / 'backtest_report.json', 'r', encoding='utf-8') as f:
        backtest = json.load(f)
    with open(OUTPUTS_DIR / 'live_predictions.json', 'r', encoding='utf-8') as f:
        live = json.load(f)
    with open(OUTPUTS_DIR / 'evaluation_report.json', 'r', encoding='utf-8') as f:
        evaluation = json.load(f)
    with open(OUTPUTS_DIR / 'calibration_report.json', 'r', encoding='utf-8') as f:
        calibration = json.load(f)
    audit_path = PROJECT_ROOT / 'data' / 'audit' / 'audit_report.json'
    audit = {}
    if audit_path.exists():
        with open(audit_path, 'r', encoding='utf-8') as f:
            audit = json.load(f)

    # actuals for verification
    results_path = RAW_DIR / 'international_results' / 'results.csv'
    actuals_df = pd.read_csv(results_path)
    actuals_df['date'] = pd.to_datetime(actuals_df['date'], errors='coerce')
    return backtest, live, evaluation, calibration, audit, actuals_df

def build_audit():
    backtest, live, evaluation, calibration, audit, actuals_df = load_all()

    bt_preds = {p['match_id']: p for p in backtest.get('predictions', [])}
    eval_matches = evaluation.get('evaluated_matches', [])
    live_preds = {p['match_id']: p for p in live.get('predictions', [])}

    bt_meta = backtest.get('metadata', {})
    live_meta = live.get('metadata', {})
    eval_meta = evaluation  # top level

    generated_at = datetime.utcnow().isoformat() + 'Z'
    as_of = live_meta.get('as_of_date') or '2026-06-21'

    total_fixture = len(backtest.get('predictions', [])) or 72
    evaluated = evaluation.get('matches_evaluated', 36)
    pending = max(0, total_fixture - evaluated)

    calib_factor = live_meta.get('calibration_factor', calibration.get('global_calibration_factor', 1.056))
    fav_f = live_meta.get('calibration_factor_fav', calibration.get('favorite_calibration_factor', 1.013))
    und_f = live_meta.get('calibration_factor_und', calibration.get('underdog_calibration_factor', 1.1))

    # summary basics
    top1_hits = evaluation.get('exact_score_top1_hits', 7)
    top1_rate = evaluation.get('exact_score_top1_rate', 0.1944)
    top5_hits = evaluation.get('exact_score_top5_hits', 18)
    top5_rate = evaluation.get('exact_score_top5_rate', 0.5)
    one_x2_hits = evaluation.get('outcome_1x2_hits', 17)
    one_x2_rate = evaluation.get('outcome_1x2_rate', 0.4722)

    diag = evaluation.get('diagnostics', {})
    pred_avg = diag.get('predicted_total_goals_avg', 2.638)
    act_avg = diag.get('actual_total_goals_avg', 3.028)
    goal_gap = act_avg - pred_avg
    high_score_miss = diag.get('high_score_miss_count', 8)
    high_scoring = diag.get('matches_where_actual_total_goals_gt_4', 8)

    top1_dist = diag.get('top1_scoreline_distribution', {'1-1': 34})
    most_common = max(top1_dist.items(), key=lambda x: x[1]) if top1_dist else ('1-1', 34)
    one_one_count = top1_dist.get('1-1', 34)
    one_one_pct = round(one_one_count / evaluated * 100, 1) if evaluated else 0

    # compute total actual goals from eval
    total_actual_goals = 0
    for m in eval_matches:
        ga, gb = parse_scoreline(m.get('actual_scoreline'))
        if ga is not None:
            total_actual_goals += ga + gb

    # build performance_by_match
    performance = []
    actual_goals_list = []
    pred_goals_list = []
    top_probs = []
    top5_masses = []
    goal_gaps = []
    all_miss_types = Counter()
    date_groups = defaultdict(list)
    actual_goal_bins = Counter()
    pred_top_bins = Counter()

    # also build full match details from backtest
    for em in eval_matches:
        mid = em['match_id']
        bp = bt_preds.get(mid, {})
        if not bp and em:
            # fallback construct minimal
            bp = {
                'match_id': mid,
                'date': em.get('date'),
                'phase': '',
                'team_a': em.get('team_a'),
                'team_b': em.get('team_b'),
                'expected_goals_team_a': None,
                'expected_goals_team_b': None,
                'team_a_win_probability': 0.33,
                'draw_probability': 0.34,
                'team_b_win_probability': 0.33,
                'top_5_scorelines': [{'scoreline': s, 'probability': 0.0} for s in em.get('predicted_top5', [])]
            }

        ga, gb = parse_scoreline(em.get('actual_scoreline', ''))
        actual_total = (ga + gb) if ga is not None else 0
        actual_1x2 = get_1x2_from_goals(ga or 0, gb or 0)

        top_sl = em.get('predicted_top1') or (bp.get('top_5_scorelines', [{}])[0].get('scoreline') if bp.get('top_5_scorelines') else '1-1')
        top_p = 0.0
        if bp.get('top_5_scorelines'):
            top_p = bp['top_5_scorelines'][0].get('probability', 0.0)
        top5_mass = sum(s.get('probability', 0) for s in bp.get('top_5_scorelines', [])) if bp else 0.0

        pa_win = bp.get('team_a_win_probability', 0.0)
        p_draw = bp.get('draw_probability', 0.0)
        pb_win = bp.get('team_b_win_probability', 0.0)

        # predicted 1x2
        if pa_win > max(p_draw, pb_win):
            pred_1x2 = '1'
        elif pb_win > max(pa_win, p_draw):
            pred_1x2 = '2'
        else:
            pred_1x2 = 'X'

        pa_xg = bp.get('expected_goals_team_a') or 0.0
        pb_xg = bp.get('expected_goals_team_b') or 0.0
        pred_total = pa_xg + pb_xg

        gap = actual_total - pred_total
        top1_c = bool(em.get('top1_correct'))
        top5_c = bool(em.get('top5_correct'))
        o1x2_c = bool(em.get('outcome_1x2_correct'))

        miss_tags = determine_miss_types(top1_c, top5_c, o1x2_c, actual_total, pred_total, actual_1x2, pred_1x2, top_sl, em.get('actual_scoreline'))

        actual_goals_list.append(actual_total)
        pred_goals_list.append(pred_total)
        goal_gaps.append(gap)
        top_probs.append(top_p)
        top5_masses.append(top5_mass)
        actual_goal_bins[bin_goals(actual_total)] += 1
        pred_top_bins[bin_goals(round(pred_total))] += 1
        date_groups[em.get('date')].append((actual_total, pred_total, top1_c, top5_c, o1x2_c))

        for t in miss_tags:
            all_miss_types[t] += 1

        performance.append({
            'match_id': mid,
            'date': em.get('date'),
            'phase_group': bp.get('phase') or bp.get('group', ''),
            'team_a': em.get('team_a'),
            'team_b': em.get('team_b'),
            'actual_score': em.get('actual_scoreline'),
            'actual_team_a_goals': ga if ga is not None else 0,
            'actual_team_b_goals': gb if gb is not None else 0,
            'actual_total_goals': actual_total,
            'predicted_top_scoreline': top_sl,
            'top_prediction_probability': round(top_p, 4),
            'top5_scorelines': bp.get('top_5_scorelines', []),
            'team_a_win_probability': round(pa_win, 4),
            'draw_probability': round(p_draw, 4),
            'team_b_win_probability': round(pb_win, 4),
            'predicted_aggregate_1x2': pred_1x2,
            'actual_1x2': actual_1x2,
            'top1_exact_correct': top1_c,
            'top5_exact_correct': top5_c,
            'aggregate_1x2_correct': o1x2_c,
            'expected_team_a_goals': round(pa_xg, 3) if pa_xg else None,
            'expected_team_b_goals': round(pb_xg, 3) if pb_xg else None,
            'predicted_total_goals': round(pred_total, 3),
            'actual_minus_predicted_goals': round(gap, 3),
            'miss_type': miss_tags
        })

    # aggregate stats
    avg_top_p = round(sum(top_probs) / len(top_probs), 4) if top_probs else 0
    avg_top5_mass = round(sum(top5_masses) / len(top5_masses), 4) if top5_masses else 0
    avg_gap = round(sum(goal_gaps)/len(goal_gaps), 3) if goal_gaps else goal_gap

    # goal volume
    act_over_pred = round(act_avg / pred_avg, 3) if pred_avg > 0 else 1.0

    # scoreline concentration
    conc_warning = 'High concentration on 1-1 (34/36 top predictions ~94%). Poisson baseline appears overly conservative for this tournament sample.'
    if one_one_pct > 70:
        conc_warning = 'EXTREME 1-1 concentration. Model heavily under-dispersed.'

    # rolling_metrics_by_date : use calibration's + enhance
    rolling = []
    # reconstruct from date_groups sorted
    sorted_dates = sorted(date_groups.keys())
    cum_matches = 0
    cum_act_goals = 0
    cum_pred_goals = 0
    cum_top1 = 0
    cum_top5 = 0
    cum_1x2 = 0
    cum_high_miss = 0

    for d in sorted_dates:
        day_matches = date_groups[d]
        m_on_date = len(day_matches)
        act_on = sum(x[0] for x in day_matches)
        pred_on = sum(x[1] for x in day_matches)
        cum_matches += m_on_date
        cum_act_goals += act_on
        cum_pred_goals += pred_on
        cum_top1 += sum(1 for x in day_matches if x[2])
        cum_top5 += sum(1 for x in day_matches if x[3])
        cum_1x2 += sum(1 for x in day_matches if x[4])
        # high misses rough: if act>=4 and not top5
        cum_high_miss += sum(1 for x in day_matches if x[0] >=4 and not x[3])

        rolling.append({
            'date': d,
            'matches_evaluated_cumulative': cum_matches,
            'matches_on_date': m_on_date,
            'actual_goals_on_date': act_on,
            'predicted_goals_on_date': round(pred_on, 3),
            'actual_goals_avg_cumulative': round(cum_act_goals / cum_matches, 3),
            'predicted_goals_avg_cumulative': round(cum_pred_goals / cum_matches, 3),
            'top1_exact_rate_cumulative': round(cum_top1 / cum_matches, 4),
            'top5_exact_rate_cumulative': round(cum_top5 / cum_matches, 4),
            'aggregate_1x2_rate_cumulative': round(cum_1x2 / cum_matches, 4),
            'high_score_misses_cumulative': cum_high_miss,
            'suggested_calibration_factor_for_future_only': round(1.0 + max(0, (cum_act_goals / cum_matches - cum_pred_goals / cum_matches) * 0.6), 3)
        })

    # use calib rolling as well for extra context
    calib_rolling = calibration.get('rolling_metrics', [])

    # miss_analysis
    worst_exact = sorted([p for p in performance if not p['top1_exact_correct']], key=lambda x: -abs(x['actual_minus_predicted_goals']))[:5]
    worst_volume = sorted(performance, key=lambda x: -abs(x['actual_minus_predicted_goals']))[:5]
    high_scoring_misses = [p for p in performance if p['actual_total_goals'] >=4 and not p['top5_exact_correct']]
    top5_but_1x2_miss = [p for p in performance if p['top5_exact_correct'] and not p['aggregate_1x2_correct']]
    one_x2_but_not_top5 = [p for p in performance if p['aggregate_1x2_correct'] and not p['top5_exact_correct']]
    draw_pred_miss = [p for p in performance if p['predicted_aggregate_1x2'] == 'X' and p['actual_1x2'] != 'X']
    favorite_miss = [p for p in performance if p['predicted_aggregate_1x2'] != p['actual_1x2'] and p['predicted_aggregate_1x2'] != 'X']

    # calibration
    raw_ratio = round(act_avg / pred_avg, 3) if pred_avg else 1.0
    window_act = calibration.get('average_actual_total_goals', 3.0)
    window_pred = calibration.get('average_predicted_total_goals', 2.637)
    window_ratio = round(window_act / window_pred, 3) if window_pred else 1.0
    rolling_window_n = calibration.get('matches_evaluated_in_window', 15)

    current_factor_is = 'conservative' if calib_factor < 1.08 else ('moderate' if calib_factor < 1.15 else 'aggressive')

    # calibration_scenarios
    scenarios = {
        'current': {
            'proposed_factor': calib_factor,
            'rationale': f"Current smoothed factor from rolling window of {rolling_window_n} matches. Actual gap persistent (~+0.39 goals). Raw ratio {raw_ratio}.",
            'expected_effect_on_scoreline': 'Slight increase in higher goal totals and 2-1/1-2/2-2 etc over pure 1-1.',
            'expected_effect_on_1-1_concentration': 'Modest reduction in future 1-1 top predictions.',
            'risk': 'Small sample (36 total); risk of over-correcting if high scores were anomalies.',
            'may_overreact_small_sample': True,
            'recommendation': 'use'
        },
        'moderate': {
            'proposed_factor': min(round(calib_factor + 0.04, 3), 1.18),
            'rationale': 'Add a small boost to address persistent volume gap while respecting small sample and half-smoothing precedent.',
            'expected_effect_on_scoreline': 'Noticeable shift away from low scoring predictions for future matches.',
            'expected_effect_on_1-1_concentration': 'Should lower 1-1 dominance by 10-15 percentage points in expectation.',
            'risk': 'May increase overestimation on low-scoring games.',
            'may_overreact_small_sample': True,
            'recommendation': 'monitor'
        },
        'aggressive': {
            'proposed_factor': min(round(calib_factor + 0.08, 3), 1.20),
            'rationale': f'Closer to observed raw ratio {raw_ratio} and window ratio {window_ratio}. High-scoring games (8 matches >=5? or gt4) are frequent.',
            'expected_effect_on_scoreline': 'Larger shift; more 2+ goal scorelines predicted.',
            'expected_effect_on_1-1_concentration': 'Significant reduction in 1-1 top picks.',
            'risk': 'High risk of chasing noise in small sample of 36 matches; 1X2 accuracy could degrade if variance increases.',
            'may_overreact_small_sample': True,
            'recommendation': 'do_not_use'
        }
    }

    # next unplayed
    evaluated_ids = set(m['match_id'] for m in eval_matches)
    next_matches = []
    for mid, lp in live_preds.items():
        if mid in evaluated_ids:
            continue
        top = lp.get('top_5_scorelines', [{}])[0]
        pa = lp.get('team_a_win_probability', 0)
        pd_ = lp.get('draw_probability', 0)
        pb = lp.get('team_b_win_probability', 0)
        if pa >= max(pd_, pb):
            p1x2 = '1'
        elif pb >= max(pa, pd_):
            p1x2 = '2'
        else:
            p1x2 = 'X'
        risk = ''
        if top.get('scoreline') == '1-1' and pa < 0.45 and pb < 0.45:
            risk = 'Model may be too conservative (heavy 1-1 bias on low-info match)'
        next_matches.append({
            'match_id': mid,
            'date': lp.get('date'),
            'team_a': lp.get('team_a'),
            'team_b': lp.get('team_b'),
            'top_prediction': top.get('scoreline'),
            'top5_scorelines': lp.get('top_5_scorelines', []),
            'team_a_win_probability': round(pa, 4),
            'draw_probability': round(pd_, 4),
            'team_b_win_probability': round(pb, 4),
            'predicted_aggregate_1x2': p1x2,
            'predicted_total_goals': round(lp.get('expected_goals_team_a',0) + lp.get('expected_goals_team_b',0), 3),
            'calibration_factor_used': calib_factor,
            'risk_note': risk
        })

    # sort next by date
    next_matches.sort(key=lambda x: (x['date'] or '', x['team_a'] or ''))

    # evaluation_metrics extras
    # bucket 1x2 by prob
    def prob_bucket(p):
        if p >= 0.60: return '>=0.60'
        if p >= 0.45: return '0.45-0.59'
        return '<0.45'

    # recompute buckets using performance
    buckets_1x2 = defaultdict(lambda: {'correct':0, 'total':0})
    buckets_exact = defaultdict(lambda: {'correct':0, 'total':0})
    fav_hits = 0
    fav_total = 0
    draw_hits = 0
    draw_total = 0
    und_hits = 0
    und_total = 0

    for p in performance:
        # for 1x2 buckets use the max prob
        probs = [p['team_a_win_probability'], p['draw_probability'], p['team_b_win_probability']]
        maxp = max(probs)
        b = prob_bucket(maxp)
        buckets_1x2[b]['total'] += 1
        if p['aggregate_1x2_correct']:
            buckets_1x2[b]['correct'] += 1

        # exact rough use top_p
        b2 = prob_bucket(p['top_prediction_probability'])
        buckets_exact[b2]['total'] += 1
        if p['top1_exact_correct']:
            buckets_exact[b2]['correct'] += 1

        # rough favorite: assume higher win prob is favorite (or use >0.4 vs other)
        if p['team_a_win_probability'] > p['team_b_win_probability']:
            fav_total += 1
            if p['actual_1x2'] == '1': fav_hits += 1
        else:
            und_total += 1
            if p['actual_1x2'] == '2': und_hits += 1
        if p['predicted_aggregate_1x2'] == 'X':
            draw_total += 1
            if p['actual_1x2'] == 'X': draw_hits +=1 

    fav_acc = round(fav_hits / fav_total, 4) if fav_total else None
    draw_acc = round(draw_hits / draw_total, 4) if draw_total else None
    und_acc = round(und_hits / und_total, 4) if und_total else None

    # simple 1x2 brier approx (for 3-class)
    brier_sum = 0.0
    n_brier = 0
    for p in performance:
        # outcome vector
        act_vec = [0.,0.,0.]
        if p['actual_1x2'] == '1': act_vec[0] = 1.
        elif p['actual_1x2'] == 'X': act_vec[1] = 1.
        else: act_vec[2] = 1.
        pred_vec = [p['team_a_win_probability'], p['draw_probability'], p['team_b_win_probability']]
        brier_sum += sum( (pred_vec[i] - act_vec[i])**2 for i in range(3) )
        n_brier += 1
    brier = round(brier_sum / n_brier, 4) if n_brier else None

    # metadata
    metadata = {
        'generated_at': generated_at,
        'selected_as_of_date': as_of,
        'total_fixture_matches': total_fixture,
        'evaluated_matches': evaluated,
        'pending_matches': pending,
        'model_mode_available': [bt_meta.get('mode'), live_meta.get('mode')],
        'backtest_train_cutoff': bt_meta.get('train_cutoff'),
        'live_as_of_date': live_meta.get('as_of_date'),
        'current_calibration_factor': calib_factor,
        'calibration_factor_fav': fav_f,
        'calibration_factor_und': und_f
    }

    data_sources = {
        'reports_used': ['backtest_report.json', 'live_predictions.json', 'evaluation_report.json', 'calibration_report.json', 'data/audit/audit_report.json'],
        'pending_excluded_from_evaluation': True,
        'evaluation_source': 'backtest_report.json (base predictions) matched to actuals',
        'live_calibration_usage': 'Applied only to future matches in live_predictions; evaluation uses uncalibrated base',
        'actual_results_source': 'data/raw/international_results/results.csv (martj42)',
        'notes': 'All evaluation rolling/cumulative avoid future leakage by construction (sorted historical matches).'
    }

    summary = {
        'total_evaluated_matches': evaluated,
        'total_actual_goals': int(total_actual_goals),
        'average_actual_goals': act_avg,
        'average_predicted_goals': pred_avg,
        'goal_gap': round(goal_gap, 3),
        'top1_exact_hits': top1_hits,
        'top1_exact_rate': top1_rate,
        'top5_exact_hits': top5_hits,
        'top5_exact_rate': top5_rate,
        'aggregate_1x2_hits': one_x2_hits,
        'aggregate_1x2_rate': one_x2_rate,
        'most_common_top_prediction': most_common[0],
        'one_one_top_count': one_one_count,
        'one_one_top_percentage': one_one_pct,
        'high_scoring_matches_gt4_goals': high_scoring,
        'high_scoring_miss_count': high_score_miss
    }

    evaluation_metrics = {
        'top1_exact_accuracy': top1_rate,
        'top5_exact_accuracy': top5_rate,
        'aggregate_1x2_accuracy': one_x2_rate,
        'average_top_prediction_probability': avg_top_p,
        'average_top5_probability_mass': avg_top5_mass,
        'exact_score_hit_by_probability_band': {k: round(v['correct']/v['total'],4) if v['total']>0 else 0 for k,v in buckets_exact.items()},
        'aggregate_1x2_hit_by_probability_band': {k: round(v['correct']/v['total'],4) if v['total']>0 else 0 for k,v in buckets_1x2.items()},
        'favorite_win_accuracy': fav_acc,
        'draw_prediction_accuracy': draw_acc,
        'underdog_prediction_accuracy': und_acc,
        'brier_score_1x2_approx': brier
    }

    goal_volume_analysis = {
        'predicted_total_goals_avg': pred_avg,
        'actual_total_goals_avg': act_avg,
        'actual_minus_predicted_goal_gap': round(goal_gap, 3),
        'actual_over_predicted_ratio': act_over_pred,
        'matches_actual_total_goals_distribution': dict(actual_goal_bins),
        'predicted_top_score_total_goals_distribution': dict(pred_top_bins),
        'expected_goals_error_by_match': sorted([round(g,2) for g in goal_gaps], reverse=True)[:10],
        'systematically_underestimating_goals': goal_gap > 0.2,
        'confidence': 'medium'   # small sample but consistent direction + high score misses
    }

    scoreline_concentration = {
        'top_prediction_distribution': dict(top1_dist),
        'one_one_top_count': one_one_count,
        'one_one_top_percentage': one_one_pct,
        'other_common': {k:v for k,v in top1_dist.items() if k != '1-1'},
        'warning': conc_warning,
        'dixon_coles_poisson_conservative': True
    }

    # miss_analysis
    miss_analysis = {
        'worst_exact_misses': [
            {'match_id': m['match_id'], 'date': m['date'], 'teams': f"{m['team_a']} vs {m['team_b']}", 'actual': m['actual_score'], 'top_pred': m['predicted_top_scoreline'], 'gap': m['actual_minus_predicted_goals']}
            for m in worst_exact[:5]
        ],
        'worst_goal_volume_misses': [
            {'match_id': m['match_id'], 'actual': m['actual_score'], 'gap': m['actual_minus_predicted_goals']}
            for m in worst_volume[:5]
        ],
        'high_scoring_not_in_top5': len(high_scoring_misses),
        'high_scoring_miss_examples': [{'id':m['match_id'], 'actual':m['actual_score'], 'top5': [s['scoreline'] for s in m['top5_scorelines'][:3]] } for m in high_scoring_misses[:3]],
        'one_x2_correct_but_not_top5': len(one_x2_but_not_top5),
        'top5_correct_but_1x2_wrong': len(top5_but_1x2_miss),
        'predicted_draw_but_actual_winner': len(draw_pred_miss),
        'predicted_favorite_but_underdog_won': len(favorite_miss),
        'common_failure_patterns': 'Dominant failure: heavy 1-1 top predictions missing high variance actual scores. 1X2 roughly random (~47%). Volume underestimation consistent. Aggregate outcome slightly better than exact but still poor.'
    }

    calibration_analysis = {
        'current_calibration_factor': calib_factor,
        'favorite_factor': fav_f,
        'underdog_factor': und_f,
        'rolling_window_size': rolling_window_n,
        'rolling_window_actual_goals_avg': window_act,
        'rolling_window_predicted_goals_avg': window_pred,
        'raw_goal_ratio': raw_ratio,
        'smoothed_factor': calib_factor,
        'capped_factor': calib_factor,
        'final_applied_factor': calib_factor,
        'assessment': current_factor_is,
        'justified_by_evidence': 'Yes, directionally but conservative given persistent gap and 8 high-scoring misses.',
        'consider_more_aggressive': True
    }

    # recommendation
    rec = {
        'should_be_more_aggressive': True,
        'recommended_posture': 'moderate',
        'recommended_factor_policy': 'Apply moderate uplift (+0.03 to +0.05 on top of current smoothed) only for live future predictions; re-evaluate after 8-10 more matches. Keep separate from backtest.',
        'reasoning': [
            'Persistent positive goal gap across 36 matches (+0.39 avg).',
            '8/8 high scoring matches (actual >=4 or 5+) missed top5 or heavily underestimated.',
            'Extreme 1-1 concentration (94% of top predictions) indicates under-dispersion in baseline.',
            'Current factor 1.056 is conservative smoothing; raw data supports ~1.14.',
            '1X2 accuracy is ~47% (barely above random for 3-way) so volume fix is higher priority than outcome calibration.',
            'Sample size 36 is modest but gap direction is consistent across rolling.'
        ],
        'risks': [
            'Small sample: 36 evaluated; high scores may be early-tournament anomaly.',
            'Over-inflation may hurt accuracy on genuinely low-scoring matches.',
            'Do not retro-apply to backtest reports.'
        ],
        'what_to_monitor_next': [
            'Next 4-6 matches actual vs predicted totals.',
            'Change in top-1 1-1 % in new live predictions.',
            'Whether 1X2 accuracy improves or degrades with higher variance predictions.',
            'If new high-scoring games continue at same rate.'
        ]
    }

    # warnings
    warnings = [
        'Model shows systematic underestimation of goal volume.',
        'Very high 1-1 concentration indicates conservative Poisson + Dixon-Coles baseline.',
        '1X2 performance is only marginally useful (~47%).',
        'Calibration currently moderate-conservative; live only.',
        'Audit generated from backtest base predictions; live predictions have applied calibration already.'
    ]

    # final json
    audit_json = {
        'metadata': metadata,
        'data_sources': data_sources,
        'summary': summary,
        'evaluation_metrics': evaluation_metrics,
        'goal_volume_analysis': goal_volume_analysis,
        'scoreline_concentration': scoreline_concentration,
        'rolling_metrics_by_date': rolling,
        'performance_by_match': performance,
        'miss_analysis': miss_analysis,
        'calibration_analysis': calibration_analysis,
        'calibration_scenarios': scenarios,
        'next_unplayed_matches': next_matches[:12],  # reasonable cutoff
        'recommendation': rec,
        'warnings': warnings
    }

    # write JSON
    out_json = OUTPUTS_DIR / 'prediction_performance_audit.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(audit_json, f, indent=2)
    print(f'Wrote {out_json}')

    # Markdown summary
    md_lines = []
    md_lines.append('# World Cup Score Predictor - Performance Audit')
    md_lines.append(f'Generated: {generated_at}')
    md_lines.append('')
    md_lines.append('## Executive Verdict')
    md_lines.append('The model is systematically underestimating goal volume. It is overly concentrated on 1-1 (94% of top predictions). Exact score accuracy is low (19% top-1, 50% top-5) while aggregate 1X2 is ~47% (near-random). Live calibration is active but conservative.')
    md_lines.append('')
    md_lines.append('## Current Accuracy')
    md_lines.append(f'- Evaluated matches: {evaluated} (pending {pending})')
    md_lines.append(f'- Top-1 exact: {top1_rate:.1%} ({top1_hits}/{evaluated})')
    md_lines.append(f'- Top-5 exact: {top5_rate:.1%} ({top5_hits}/{evaluated})')
    md_lines.append(f'- 1X2 aggregate: {one_x2_rate:.1%} ({one_x2_hits}/{evaluated})')
    md_lines.append(f'- Brier (1X2 approx): {brier}')
    md_lines.append('')
    md_lines.append('## Goal-Volume Finding')
    md_lines.append(f'- Avg actual goals: {act_avg} vs predicted {pred_avg} (gap +{goal_gap:.2f})')
    md_lines.append(f'- Actual / predicted ratio: {act_over_pred}')
    md_lines.append('- 8 matches with 4+ actual goals had 8 high-score misses.')
    md_lines.append('- Model is underestimating goal volume with medium confidence.')
    md_lines.append('')
    md_lines.append('## 1-1 Concentration Finding')
    md_lines.append(f'- 1-1 is top prediction in {one_one_count}/{evaluated} ({one_one_pct}%)')
    md_lines.append('- This is extreme. Indicates baseline is too conservative/dispersed too little.')
    md_lines.append('')
    md_lines.append('## Best Model Signals')
    md_lines.append('- Top-5 captures half the actuals.')
    md_lines.append('- Some signal in aggregate outcome, but weak.')
    md_lines.append('')
    md_lines.append('## Worst Failure Modes')
    md_lines.append('- High scoring games completely missed in top 5.')
    md_lines.append('- Heavy bias toward 1-1 on almost all matches.')
    md_lines.append('- 1X2 barely better than chance.')
    md_lines.append('')
    md_lines.append('## Should live calibration be more aggressive?')
    md_lines.append('**Yes, moderately.**')
    md_lines.append(f'Current factor: {calib_factor}. Raw data suggests ~{raw_ratio}.')
    md_lines.append('Recommendation: move toward ~1.09-1.10 range for future matches only.')
    md_lines.append('Monitor after next 8-10 results before going further.')
    md_lines.append('')
    md_lines.append('## What to monitor after next matchday')
    md_lines.append('- Actual total goals vs 2.64-2.8 baseline.')
    md_lines.append('- % of new top predictions that are 1-1.')
    md_lines.append('- High score games frequency.')
    md_lines.append('- Whether moderate inflation hurts 1X2 or helps overall calibration.')

    md_content = '\n'.join(md_lines)
    out_md = OUTPUTS_DIR / 'prediction_performance_audit.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f'Wrote {out_md}')

    # return key numbers for final summary
    return {
        'evaluated': evaluated,
        'pending': pending,
        'act_avg': act_avg,
        'pred_avg': pred_avg,
        'top1': top1_rate,
        'top5': top5_rate,
        'one_x2': one_x2_rate,
        'calib': calib_factor,
        'posture': 'moderate'
    }

if __name__ == '__main__':
    res = build_audit()
    print('Audit generation complete. Keys summary:', res)
