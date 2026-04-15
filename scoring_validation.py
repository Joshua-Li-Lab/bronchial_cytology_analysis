"""
scoring_validation.py
Performs internal validation of the trained scoring model on a held-out set.
Scores each specimen using the final model weights, classifies into risk
groups (high / intermediate / low), and evaluates stratification performance
with chi-squared testing and score-level summary statistics.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")

SCORE_COLUMNS = [
    'AGE55ormore',
    'APTT_Flagging', 'Albumin_Flagging', 'Basophil, absolute_Flagging',
    'C-Reactive Protein_Flagging', 'Creatinine_Flagging',
    'Eosinophil, absolute_Flagging', 'Haemoglobin, Blood_Flagging',
    'Lactate Dehydrogenase_Flagging', 'Lymphocyte, absolute_Flagging',
    'MCH_Flagging', 'MCHC_Flagging', 'MCV_Flagging',
    'Neutrophil, absolute_Flagging', 'Platelet_Flagging',
    'Protein, Total_Flagging', 'Prothrombin Time_Flagging',
    'WBC_Flagging'
]


def _score_specimen(row, weights, base_score):
    """Compute a single specimen's risk score."""
    score = base_score
    for col in SCORE_COLUMNS:
        try:
            specimen_val = int(row[col])
        except (ValueError, TypeError):
            continue
        key = (col, specimen_val)
        if key in weights:
            score += weights[key]
    return score


def _classify_risk(score, high_cutoff, low_cutoff):
    """Assign a risk group label based on score cutoffs."""
    if score > high_cutoff:
        return 'High-Risk'
    elif score <= low_cutoff:
        return 'Low-Risk'
    return 'Intermediate'


def run_validation(val_df, weights, base_score, high_cutoff, low_cutoff, output_dir):
    """Score validation specimens, classify risk groups, and save results.

    Args:
        val_df: Validation DataFrame (encoded, same format as training).
        weights: dict of (variable, flag_value) → weight.
        base_score: Integer added to every specimen's raw score.
        high_cutoff: Score above which a specimen is High-Risk.
        low_cutoff: Score at or below which a specimen is Low-Risk.
        output_dir: Directory for output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Internal Validation")
    print("=" * 70)
    print(f"Validation set:  {len(val_df)}")
    print(f"  Positives (cancer): {int(val_df['ANY_LUNG'].sum())}")
    print(f"  Negatives (normal): {int((val_df['ANY_LUNG'] == 0).sum())}")

    # --- Score specimens ---
    val_scored = val_df.copy()
    val_scored['Risk_Score'] = val_scored.apply(
        lambda row: _score_specimen(row, weights, base_score), axis=1
    )

    print(f"\nScore range:  {val_scored['Risk_Score'].min()} to {val_scored['Risk_Score'].max()}")
    print(f"Score mean:   {val_scored['Risk_Score'].mean():.2f}")
    print(f"Score median: {val_scored['Risk_Score'].median():.1f}")

    # Score distribution
    print(f"\nScore distribution:")
    print(f"  {'Score':>6s} | {'Total':>6s} | {'Cancer':>7s} | {'Normal':>7s} | {'Cancer%':>8s}")
    print("  " + "-" * 48)
    for score_val in sorted(val_scored['Risk_Score'].unique()):
        subset = val_scored[val_scored['Risk_Score'] == score_val]
        total = len(subset)
        cancer = int(subset['ANY_LUNG'].sum())
        normal = total - cancer
        pct = cancer / total * 100 if total > 0 else 0
        print(f"  {score_val:>6d} | {total:>6d} | {cancer:>7d} | {normal:>7d} | {pct:>7.1f}%")

    # --- Risk group classification ---
    val_scored['Risk_Group'] = val_scored['Risk_Score'].apply(
        lambda s: _classify_risk(s, high_cutoff, low_cutoff)
    )

    print(f"\nRisk groups (High: >{high_cutoff}, Low: <={low_cutoff}):\n")
    print(f"  {'Group':>15s} | {'Total':>6s} | {'Cancer':>7s} | {'Normal':>7s} | {'Cancer%':>8s}")
    print("  " + "-" * 56)
    for group in ['High-Risk', 'Intermediate', 'Low-Risk']:
        subset = val_scored[val_scored['Risk_Group'] == group]
        n = len(subset)
        cancer = int(subset['ANY_LUNG'].sum())
        normal = n - cancer
        pct = cancer / n * 100 if n > 0 else 0
        print(f"  {group:>15s} | {n:>6d} | {cancer:>7d} | {normal:>7d} | {pct:>7.1f}%")

    total_all = len(val_scored)
    cancer_all = int(val_scored['ANY_LUNG'].sum())
    print("  " + "-" * 56)
    print(f"  {'Total':>15s} | {total_all:>6d} | {cancer_all:>7d} | "
          f"{total_all - cancer_all:>7d} | {cancer_all / total_all * 100:>7.1f}%")

    # --- Chi-squared test across risk groups ---
    chi2_groups = ['High-Risk', 'Intermediate', 'Low-Risk']
    chi2_observed, chi2_groups_present = [], []
    for group in chi2_groups:
        sub = val_scored[val_scored['Risk_Group'] == group]
        if len(sub) == 0:
            continue
        ca = int(sub['ANY_LUNG'].sum())
        chi2_observed.append([ca, len(sub) - ca])
        chi2_groups_present.append(group)

    chi2_stat, chi2_p, chi2_dof = np.nan, np.nan, np.nan
    if len(chi2_groups_present) >= 2:
        chi2_table = np.array(chi2_observed)
        chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(chi2_table)
        p_str = f"{chi2_p:.2e}" if chi2_p < 0.001 else f"{chi2_p:.6f}"
        print(f"\n  Chi-squared test: statistic={chi2_stat:.4f}, df={chi2_dof}, p={p_str}")

    # --- Summary table by score threshold ---
    summary_rows = []
    for score_val in sorted(val_scored['Risk_Score'].unique()):
        subset = val_scored[val_scored['Risk_Score'] == score_val]
        cancer = int(subset['ANY_LUNG'].sum())
        normal = len(subset) - cancer
        total = len(subset)
        cancer_pct = cancer / total * 100 if total > 0 else 0

        leq = val_scored[val_scored['Risk_Score'] <= score_val]
        gt = val_scored[val_scored['Risk_Score'] > score_val]

        cancer_leq, total_leq = int(leq['ANY_LUNG'].sum()), len(leq)
        cancer_gt, total_gt = int(gt['ANY_LUNG'].sum()), len(gt)
        pct_leq = cancer_leq / total_leq * 100 if total_leq > 0 else 0
        pct_gt = cancer_gt / total_gt * 100 if total_gt > 0 else 0
        delta = abs(pct_leq - pct_gt)
        over100 = int(total_leq > 100 and total_gt > 100)

        summary_rows.append({
            'Score': score_val, 'Cancer': cancer, 'Normal': normal,
            'Total': total, 'Cancer%': round(cancer_pct, 2),
            'Cancer<=Score': cancer_leq, 'Total<=Score': total_leq,
            'Cancer<=Score%': round(pct_leq, 2),
            'Cancer>Score': cancer_gt, 'Total>Score': total_gt,
            'Cancer>Score%': round(pct_gt, 2),
            'Delta': round(delta, 2), 'Over100': over100,
        })
    summary_df = pd.DataFrame(summary_rows)

    # --- Save outputs ---
    val_scored.to_excel(
        os.path.join(output_dir, 'validation_scored_specimens.xlsx'), index=False)
    summary_df.to_excel(
        os.path.join(output_dir, 'validation_summary_by_score.xlsx'), index=False)

    # Risk groups + chi-squared result
    risk_rows = []
    for group in ['High-Risk', 'Intermediate', 'Low-Risk']:
        subset = val_scored[val_scored['Risk_Group'] == group]
        n = len(subset)
        cancer = int(subset['ANY_LUNG'].sum())
        risk_rows.append({
            'Risk_Group': group, 'Total': n,
            'Cancer': cancer, 'Normal': n - cancer,
            'Cancer%': round(cancer / n * 100 if n > 0 else 0, 2)
        })
    risk_df = pd.DataFrame(risk_rows)
    if len(chi2_groups_present) >= 2:
        p_str = f"{chi2_p:.2e}" if chi2_p < 0.001 else f"{chi2_p:.6f}"
        chi2_row = pd.DataFrame([{
            'Risk_Group': 'Chi2_Overall', 'Total': '',
            'Cancer': '', 'Normal': f'chi2={chi2_stat:.4f}, df={chi2_dof}',
            'Cancer%': f'p={p_str}'
        }])
        risk_df = pd.concat([risk_df, chi2_row], ignore_index=True)
    risk_df.to_excel(
        os.path.join(output_dir, 'validation_risk_groups.xlsx'), index=False)

    # Weights used
    weight_rows = []
    for (var, val), w in sorted(weights.items(), key=lambda x: (-x[1], x[0])):
        if var == 'AGE55ormore':
            meaning = '>=55' if val == 1 else '<55'
        else:
            meaning = {1: 'Low', 2: 'Normal', 3: 'High'}.get(val, str(val))
        weight_rows.append({
            'Variable': var, 'Flag_Value': val,
            'Flag_Meaning': meaning, 'Weight': w
        })
    weights_df = pd.DataFrame(weight_rows)
    weights_df.loc[len(weights_df)] = {
        'Variable': '(BASE_SCORE)', 'Flag_Value': '-',
        'Flag_Meaning': 'All specimens', 'Weight': base_score
    }
    weights_df.to_excel(
        os.path.join(output_dir, 'validation_weights_used.xlsx'), index=False)

    print(f"\nAll validation outputs saved to: {os.path.abspath(output_dir)}")
