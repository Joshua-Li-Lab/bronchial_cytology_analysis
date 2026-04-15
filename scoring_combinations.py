"""
scoring_combinations.py
Implements the two-phase combinatorial scoring model training pipeline.
Phase 1 optimises weights for the top 8 chi-square features via exhaustive
enumeration. Phase 2 extends to 16 features with Phase 1 weights fixed.
The best model is selected by maximising Delta2 (stratification gap) on
the training set.
"""

import os
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# =========================================================================
# Helper Functions
# =========================================================================

def generate_combinations(score_df):
    """Generate all weight combinations from the score assignments."""
    score_dict = {}
    for _, row in score_df.iterrows():
        key = (row['Flagging'], row['Value'])
        if isinstance(row['Score'], list):
            score_dict[key] = row['Score']
        else:
            score_dict[key] = [row['Score']]
    all_keys = list(score_dict.keys())
    combos = list(itertools.product(*(score_dict[k] for k in all_keys)))
    results_list = []
    for combo in combos:
        formatted = {f"{f} comparison {c}": s for (f, c), s in zip(all_keys, combo)}
        results_list.append(formatted)
    return pd.DataFrame(results_list)


def compute_patient_scores(patient_df, combination_df):
    """Score every patient under every weight combination."""
    results_list = []
    for _, row in tqdm(patient_df.iterrows(), total=patient_df.shape[0],
                       desc="Computing patient scores"):
        row_results = {'Number': row['Number'], 'ANY_LUNG': row['ANY_LUNG']}
        for combo_idx in range(len(combination_df)):
            score = 0
            for col in patient_df.columns[2:]:
                comp_val = row[col]
                comp_col = f'{col} comparison {comp_val}'
                if comp_col in combination_df.columns:
                    score += combination_df.at[combo_idx, comp_col]
            row_results[f'Combo_{combo_idx}'] = score
        results_list.append(row_results)
    return pd.DataFrame(results_list)


def compute_summary(combination_result_df):
    """Compute cancer-rate stratification summary for each combination."""
    result_summary = []
    for combo in tqdm(combination_result_df.columns[2:], desc="Processing combos"):
        unique_scores = sorted(combination_result_df[combo].unique())
        for score in unique_scores:
            subset = combination_result_df[combination_result_df[combo] == score]
            count_cancer = subset[subset['ANY_LUNG'].isin([1, 2])].shape[0]
            count_normal = subset[subset['ANY_LUNG'] == 0].shape[0]
            total_count = count_cancer + count_normal

            leq = combination_result_df[combination_result_df[combo] <= score]
            gt = combination_result_df[combination_result_df[combo] > score]

            cancer_leq = leq['ANY_LUNG'].sum()
            cancer_gt = gt['ANY_LUNG'].sum()
            total_leq = leq['ANY_LUNG'].count()
            total_gt = gt['ANY_LUNG'].count()

            pct = count_cancer / total_count if total_count > 0 else 0
            pct_leq = cancer_leq / total_leq if total_leq > 0 else 0
            pct_gt = cancer_gt / total_gt if total_gt > 0 else 0
            delta = abs(pct_leq - pct_gt)
            over_100 = int(total_leq > 100 and total_gt > 100)

            result_summary.append({
                'Combo': combo, 'Score': score,
                'Cancer Cases': count_cancer, 'Normal Cases': count_normal,
                'Total Patients': total_count, 'Cancer %': pct * 100,
                'Cancer <= Score': cancer_leq, 'Cancer > Score': cancer_gt,
                'Total <= Score': total_leq, 'Total > Score': total_gt,
                'Cancer <= Score %': pct_leq * 100, 'Cancer > Score %': pct_gt * 100,
                'Delta': delta * 100, 'Over 100': over_100
            })
    return pd.DataFrame(result_summary)


def compute_delta2(summary_df):
    """Compute Delta2: max cross-threshold gap among valid score pairs."""
    filtered = summary_df[summary_df['Over 100'] == 1]
    results_list = []
    for combo, group in filtered.groupby('Combo'):
        for idx, row in group.iterrows():
            cancer_less = row['Cancer <= Score %']
            max_diff = 0
            comp_score = None
            comp_cancer = None
            for comp_idx, comp_row in group.iterrows():
                if comp_idx != idx:
                    diff = abs(cancer_less - comp_row['Cancer > Score %'])
                    if diff > max_diff:
                        max_diff = diff
                        comp_score = comp_row['Score']
                        comp_cancer = comp_row['Cancer > Score %']
            results_list.append({
                'Combo': combo, 'Score': row['Score'],
                'Cancer <= Score %': cancer_less,
                'Compared Score': comp_score,
                'Compared Cancer > Score %': comp_cancer,
                'Delta2': max_diff
            })
    return pd.DataFrame(results_list)


def merge_summary_delta2(summary_df, delta2_df):
    """Merge summary with Delta2 results."""
    return pd.merge(
        summary_df,
        delta2_df[['Combo', 'Score', 'Compared Score',
                    'Compared Cancer > Score %', 'Delta2']],
        on=['Combo', 'Score'], how='left'
    )


def sort_summary_by_combo(summary_df):
    """Sort summary by combo index then score."""
    summary_df = summary_df.copy()
    summary_df['Combo_digits'] = summary_df['Combo'].str.split('_').str[1].astype(int)
    summary_df.sort_values(by=['Combo_digits', 'Score'], inplace=True)
    summary_df.drop(columns=['Combo_digits'], inplace=True)
    return summary_df


def extract_weights(combination_df, combo_idx):
    """Extract (flagging, flag_value) → weight dict from a combination row."""
    row = combination_df.iloc[combo_idx]
    weights = {}
    for col_name, weight in row.items():
        parts = col_name.rsplit(' comparison ', 1)
        if len(parts) == 2:
            flagging = parts[0]
            try:
                flag_value = int(float(parts[1]))
            except (ValueError, TypeError):
                flag_value = parts[1]
            weights[(flagging, flag_value)] = int(weight)
    return weights


# =========================================================================
# Training Pipeline
# =========================================================================

def _run_phase(variables_df, train_df, output_dir, phase_name):
    """Run a single phase: generate combinations, score patients, evaluate."""
    combination_df = generate_combinations(variables_df)
    combination_df.to_excel(os.path.join(output_dir, f'{phase_name}_combinations.xlsx'))
    print(f"{phase_name} combinations: {len(combination_df)}")

    result_df = compute_patient_scores(train_df, combination_df)
    result_df.to_excel(os.path.join(output_dir, f'{phase_name}_patient_scores.xlsx'))

    summary_df = compute_summary(result_df)
    summary_df = sort_summary_by_combo(summary_df)
    summary_df.to_excel(os.path.join(output_dir, f'{phase_name}_summary.xlsx'))

    delta2_df = compute_delta2(summary_df)
    delta2_df.to_excel(os.path.join(output_dir, f'{phase_name}_delta2.xlsx'))

    merged_df = merge_summary_delta2(summary_df, delta2_df)
    merged_df.to_excel(os.path.join(output_dir, f'{phase_name}_summary_merged.xlsx'))

    # Select best combination by Delta2
    valid = merged_df[merged_df['Over 100'] == 1]
    metrics = valid.groupby('Combo').agg(
        Best_Delta=('Delta', 'max'), Best_Delta2=('Delta2', 'max')
    ).reset_index()
    metrics['Best_Delta2'] = metrics['Best_Delta2'].fillna(0)
    metrics.sort_values(['Best_Delta2', 'Best_Delta'], ascending=[False, False], inplace=True)
    metrics.reset_index(drop=True, inplace=True)

    return combination_df, merged_df, metrics


def run_training(ranked_results_df, train_df, output_dir):
    """Execute the full two-phase training pipeline.

    Phase 1: optimise weights for the top 8 chi-square features.
    Phase 2: extend to 16 features with Phase 1 weights locked.

    Returns:
        final_weights: dict of (variable, flag_value) → weight
        best_weights_df: DataFrame describing each weight entry
        pipeline_summary: dict with training metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    ranked_results_df.to_excel(os.path.join(output_dir, 'results_df_direction.xlsx'))

    sorted_df = ranked_results_df.sort_values(
        by='Chi2 Statistic', ascending=False
    ).reset_index(drop=True)

    # --- Phase 1: Top 8 variables ---
    print("\n" + "=" * 70)
    print("Phase 1: Top 8 Variables")
    print("=" * 70)
    row_0_to_7 = sorted_df.iloc[:8].reset_index(drop=True)
    print(row_0_to_7[['Flagging', 'Value', 'Chi2 Statistic', 'P-Value', 'Score']])

    p1_combo_df, p1_merged, p1_metrics = _run_phase(
        row_0_to_7, train_df, output_dir, 'phase1')

    best_p1 = p1_metrics.iloc[0]
    best_p1_combo = best_p1['Combo']
    best_p1_idx = int(best_p1_combo.split('_')[1])
    phase1_weights = extract_weights(p1_combo_df, best_p1_idx)

    print(f"\nPhase 1 best: {best_p1_combo} "
          f"(Delta={best_p1['Best_Delta']:.2f}%, Delta2={best_p1['Best_Delta2']:.2f}%)")

    # --- Phase 2: Top 16 variables (Phase 1 weights fixed) ---
    print("\n" + "=" * 70)
    print("Phase 2: Top 16 Variables")
    print("=" * 70)
    row_8_to_15 = sorted_df.iloc[8:16].reset_index(drop=True)
    row_0_to_15 = pd.concat([row_0_to_7, row_8_to_15], ignore_index=True)

    # Lock Phase 1 weights
    for (flagging, val), score_val in phase1_weights.items():
        mask = ((row_0_to_15['Flagging'] == flagging) &
                (row_0_to_15['Value'] == val))
        if mask.any():
            row_0_to_15.loc[mask, 'Score'] = score_val

    print(row_0_to_15[['Flagging', 'Value', 'Chi2 Statistic', 'Score']])

    p2_combo_df, p2_merged, p2_metrics = _run_phase(
        row_0_to_15, train_df, output_dir, 'phase2')
    p2_metrics.to_excel(os.path.join(output_dir, 'training_ranking.xlsx'), index=False)

    # --- Final model selection ---
    print("\n" + "=" * 70)
    print("Final Model Selection")
    print("=" * 70)
    print("\nTraining ranking (sorted by Best_Delta2):")
    print(p2_metrics.head(10).to_string(index=False))

    best_combo_name = p2_metrics.iloc[0]['Combo']
    best_combo_idx = int(best_combo_name.split('_')[1])
    final_weights = extract_weights(p2_combo_df, best_combo_idx)

    print(f"\n*** BEST MODEL: {best_combo_name} ***")
    print(f"    Delta={p2_metrics.iloc[0]['Best_Delta']:.2f}%, "
          f"Delta2={p2_metrics.iloc[0]['Best_Delta2']:.2f}%")

    # Build weight summary table
    weight_rows = []
    for (flagging, flag_val), weight in final_weights.items():
        weight_rows.append({
            'Variable': flagging,
            'Flag_Value': flag_val,
            'Flag_Meaning': {1: 'Low', 2: 'Normal', 3: 'High', 0: 'Ref'}.get(flag_val, str(flag_val)),
            'Weight': weight,
        })
    best_weights_df = pd.DataFrame(weight_rows)
    best_weights_df.sort_values('Weight', ascending=False, inplace=True)
    best_weights_df.to_excel(
        os.path.join(output_dir, 'FINAL_best_model_weights.xlsx'), index=False)
    print("\nFinal model weights:")
    print(best_weights_df.to_string(index=False))

    # Pipeline summary
    pipeline_summary = {
        'Phase1_Best_Combo': best_p1_combo,
        'Phase1_Delta': best_p1['Best_Delta'],
        'Phase1_Delta2': best_p1['Best_Delta2'],
        'Phase2_Best_Combo': best_combo_name,
        'Phase2_Train_Delta': p2_metrics.iloc[0]['Best_Delta'],
        'Phase2_Train_Delta2': p2_metrics.iloc[0]['Best_Delta2'],
    }
    pd.DataFrame([pipeline_summary]).to_excel(
        os.path.join(output_dir, 'FINAL_pipeline_summary.xlsx'), index=False)

    return final_weights, best_weights_df, pipeline_summary
