"""
scoring_combinations.py
=======================
Combinatorial scoring model training (Phase 1 and Phase 2).

Enumerates all weight combinations for the top chi-square-selected
variables, scores training patients under each combination, and selects
the model that maximises Delta and Delta2 (risk stratification metrics).

Phase 1: Top 8 variables — find best weight combination.
Phase 2: Fix Phase 1 weights, add next 8 variables, optimise again.
"""

import os
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm


# ===================================================================
# Helper functions
# ===================================================================

def generate_combinations(score_df: pd.DataFrame) -> pd.DataFrame:
    """Enumerate all weight combinations from a score specification table.

    Each row in score_df has a Flagging column, Value, and Score field
    that is either a scalar (fixed weight) or a list of candidate weights.

    Returns a DataFrame where each row is one combination of weights.
    """
    score_dict = {}
    for _, row in score_df.iterrows():
        key = (row["Flagging"], row["Value"])
        if isinstance(row["Score"], list):
            score_dict[key] = row["Score"]
        else:
            score_dict[key] = [row["Score"]]

    all_keys = list(score_dict.keys())
    combos = list(itertools.product(*(score_dict[k] for k in all_keys)))

    results_list = []
    for combo in combos:
        formatted = {
            f"{f} comparison {c}": s for (f, c), s in zip(all_keys, combo)
        }
        results_list.append(formatted)

    return pd.DataFrame(results_list)


def compute_patient_scores(
    patient_df: pd.DataFrame,
    combination_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every patient under every weight combination.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Must contain 'HN Number', 'ANY_LUNG', and flagging columns.
    combination_df : pd.DataFrame
        Output of generate_combinations().

    Returns
    -------
    pd.DataFrame
        Columns: HN Number, ANY_LUNG, Combo_0, Combo_1, ...
    """
    results_list = []
    for _, row in tqdm(
        patient_df.iterrows(),
        total=patient_df.shape[0],
        desc="Computing patient scores",
    ):
        row_results = {
            "HN Number": row["HN Number"],
            "ANY_LUNG": row["ANY_LUNG"],
        }
        for combo_idx in range(len(combination_df)):
            score = 0
            for col in patient_df.columns[2:]:
                comp_val = row[col]
                comp_col = f"{col} comparison {comp_val}"
                if comp_col in combination_df.columns:
                    score += combination_df.at[combo_idx, comp_col]
            row_results[f"Combo_{combo_idx}"] = score
        results_list.append(row_results)

    return pd.DataFrame(results_list)


def compute_summary(combination_result_df: pd.DataFrame) -> pd.DataFrame:
    """Compute score-level summary statistics for every combination.

    For each combination and each unique score threshold, computes:
    - Cancer rate in patients <= threshold vs > threshold
    - Delta = absolute difference in cancer rates
    - Over 100 flag (both groups must have > 100 patients)
    """
    result_summary = []
    for combo in tqdm(
        combination_result_df.columns[2:], desc="Processing combos"
    ):
        unique_scores = sorted(combination_result_df[combo].unique())
        for score in unique_scores:
            subset = combination_result_df[
                combination_result_df[combo] == score
            ]
            count_cancer = subset[
                subset["ANY_LUNG"].isin([1, 2])
            ].shape[0]
            count_normal = subset[subset["ANY_LUNG"] == 0].shape[0]
            total_count = count_cancer + count_normal

            leq = combination_result_df[
                combination_result_df[combo] <= score
            ]
            gt = combination_result_df[
                combination_result_df[combo] > score
            ]

            cancer_leq = leq["ANY_LUNG"].sum()
            cancer_gt = gt["ANY_LUNG"].sum()
            total_leq = leq["ANY_LUNG"].count()
            total_gt = gt["ANY_LUNG"].count()

            pct = count_cancer / total_count if total_count > 0 else 0
            pct_leq = cancer_leq / total_leq if total_leq > 0 else 0
            pct_gt = cancer_gt / total_gt if total_gt > 0 else 0
            delta = abs(pct_leq - pct_gt)
            over_100 = int(total_leq > 100 and total_gt > 100)

            result_summary.append({
                "Combo": combo, "Score": score,
                "Cancer Cases": count_cancer,
                "Normal Cases": count_normal,
                "Total Patients": total_count,
                "Cancer %": pct * 100,
                "Cancer <= Score": cancer_leq,
                "Cancer > Score": cancer_gt,
                "Total <= Score": total_leq,
                "Total > Score": total_gt,
                "Cancer <= Score %": pct_leq * 100,
                "Cancer > Score %": pct_gt * 100,
                "Delta": delta * 100,
                "Over 100": over_100,
            })

    return pd.DataFrame(result_summary)


def compute_delta2(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Delta2: max |cancer_leq%(s1) - cancer_gt%(s2)| across pairs.

    Only considers score thresholds where Over 100 == 1.
    """
    filtered = summary_df[summary_df["Over 100"] == 1]
    results_list = []
    for combo, group in filtered.groupby("Combo"):
        for idx, row in group.iterrows():
            cancer_less = row["Cancer <= Score %"]
            max_diff = 0
            comp_score = None
            comp_cancer = None
            for comp_idx, comp_row in group.iterrows():
                if comp_idx != idx:
                    diff = abs(cancer_less - comp_row["Cancer > Score %"])
                    if diff > max_diff:
                        max_diff = diff
                        comp_score = comp_row["Score"]
                        comp_cancer = comp_row["Cancer > Score %"]
            results_list.append({
                "Combo": combo, "Score": row["Score"],
                "Cancer <= Score %": cancer_less,
                "Compared Score": comp_score,
                "Compared Cancer > Score %": comp_cancer,
                "Delta2": max_diff,
            })

    return pd.DataFrame(results_list)


def merge_summary_delta2(
    summary_df: pd.DataFrame,
    delta2_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge summary with Delta2 values."""
    return pd.merge(
        summary_df,
        delta2_df[["Combo", "Score", "Compared Score",
                    "Compared Cancer > Score %", "Delta2"]],
        on=["Combo", "Score"],
        how="left",
    )


def sort_summary_by_combo(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Sort summary by combo index then score."""
    df = summary_df.copy()
    df["Combo_digits"] = df["Combo"].str.split("_").str[1].astype(int)
    df.sort_values(by=["Combo_digits", "Score"], inplace=True)
    df.drop(columns=["Combo_digits"], inplace=True)
    return df


def extract_weights(
    combination_df: pd.DataFrame,
    combo_idx: int,
) -> dict:
    """Extract {(flagging, flag_value): weight} from a combination row."""
    row = combination_df.iloc[combo_idx]
    weights = {}
    for col_name, weight in row.items():
        parts = col_name.rsplit(" comparison ", 1)
        if len(parts) == 2:
            flagging = parts[0]
            try:
                flag_value = int(float(parts[1]))
            except (ValueError, TypeError):
                flag_value = parts[1]
            weights[(flagging, flag_value)] = int(weight)
    return weights


# ===================================================================
# Phase 1 and Phase 2 pipeline
# ===================================================================

def run_phase1(
    ranked_results_df: pd.DataFrame,
    train_df: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Phase 1: Select top 8 variables and find best combination.

    Returns dict with keys: best_combo, best_delta, best_delta2,
    phase1_weights, combination_df, row_0_to_7.
    """
    print("\n" + "=" * 70)
    print("Phase 1 — Top 8 Variables")
    print("=" * 70)

    sorted_df = ranked_results_df.sort_values(
        by="Chi2 Statistic", ascending=False
    ).reset_index(drop=True)

    row_0_to_7 = sorted_df.iloc[:8].reset_index(drop=True)
    print("\nPhase 1 variables:")
    print(row_0_to_7[["Flagging", "Value", "Chi2 Statistic", "P-Value", "Score"]])

    combination_df = generate_combinations(row_0_to_7)
    combination_df.to_excel(
        os.path.join(output_dir, "phase1_combinations.xlsx")
    )
    print(f"Phase 1 combinations: {len(combination_df)}")

    result_df = compute_patient_scores(train_df, combination_df)
    result_df.to_excel(
        os.path.join(output_dir, "phase1_patient_scores.xlsx")
    )

    summary_df = compute_summary(result_df)
    summary_df = sort_summary_by_combo(summary_df)
    summary_df.to_excel(os.path.join(output_dir, "phase1_summary.xlsx"))

    delta2_df = compute_delta2(summary_df)
    delta2_df.to_excel(os.path.join(output_dir, "phase1_delta2.xlsx"))

    merged_df = merge_summary_delta2(summary_df, delta2_df)
    merged_df.to_excel(
        os.path.join(output_dir, "phase1_summary_merged.xlsx")
    )

    # Select best by Delta2
    valid = merged_df[merged_df["Over 100"] == 1]
    metrics = (
        valid.groupby("Combo")
        .agg(Best_Delta=("Delta", "max"), Best_Delta2=("Delta2", "max"))
        .reset_index()
    )
    metrics["Best_Delta2"] = metrics["Best_Delta2"].fillna(0)
    metrics.sort_values(
        ["Best_Delta2", "Best_Delta"], ascending=[False, False], inplace=True
    )

    best = metrics.iloc[0]
    best_combo = best["Combo"]
    best_idx = int(best_combo.split("_")[1])
    phase1_weights = extract_weights(combination_df, best_idx)

    print(
        f"\nPhase 1 best: {best_combo} "
        f"(Delta={best['Best_Delta']:.2f}%, Delta2={best['Best_Delta2']:.2f}%)"
    )

    return {
        "best_combo": best_combo,
        "best_delta": best["Best_Delta"],
        "best_delta2": best["Best_Delta2"],
        "phase1_weights": phase1_weights,
        "combination_df": combination_df,
        "row_0_to_7": row_0_to_7,
        "sorted_df": sorted_df,
    }


def run_phase2(
    train_df: pd.DataFrame,
    phase1_results: dict,
    output_dir: str,
) -> dict:
    """Phase 2: Fix Phase 1 weights, add next 8 variables, re-optimise.

    Returns dict with keys: best_combo, best_delta, best_delta2,
    best_weights, combination_df, p2_metrics.
    """
    print("\n" + "=" * 70)
    print("Phase 2 — Top 16 Variables")
    print("=" * 70)

    sorted_df = phase1_results["sorted_df"]
    row_0_to_7 = phase1_results["row_0_to_7"]
    phase1_weights = phase1_results["phase1_weights"]
    best_p1_combo = phase1_results["best_combo"]

    row_8_to_15 = sorted_df.iloc[8:16].reset_index(drop=True)
    row_8_to_15 = pd.concat([row_0_to_7, row_8_to_15], ignore_index=True)

    # Fix Phase 1 weights
    for (flagging, val), score_val in phase1_weights.items():
        mask = (
            (row_8_to_15["Flagging"] == flagging)
            & (row_8_to_15["Value"] == val)
        )
        if mask.any():
            row_8_to_15.loc[mask, "Score"] = score_val

    print(
        f"\nPhase 2 variables "
        f"(with Phase 1 weights fixed from {best_p1_combo}):"
    )
    print(row_8_to_15[["Flagging", "Value", "Chi2 Statistic", "Score"]])

    combination_df = generate_combinations(row_8_to_15)
    combination_df.to_excel(
        os.path.join(output_dir, "phase2_combinations.xlsx")
    )
    print(f"Phase 2 combinations: {len(combination_df)}")

    result_df = compute_patient_scores(train_df, combination_df)
    result_df.to_excel(
        os.path.join(output_dir, "phase2_patient_scores.xlsx")
    )

    summary_df = compute_summary(result_df)
    summary_df = sort_summary_by_combo(summary_df)
    summary_df.to_excel(os.path.join(output_dir, "phase2_summary.xlsx"))

    delta2_df = compute_delta2(summary_df)
    merged_df = merge_summary_delta2(summary_df, delta2_df)
    merged_df.to_excel(
        os.path.join(output_dir, "phase2_summary_merged.xlsx")
    )

    # Top results display
    top_delta = merged_df[merged_df["Over 100"] == 1].nlargest(10, "Delta")
    top_delta2 = merged_df[merged_df["Over 100"] == 1].nlargest(10, "Delta2")
    print("\nPhase 2 — Top 10 by Delta (Over 100 == 1):")
    print(top_delta[["Combo", "Delta", "Over 100"]])
    print("\nPhase 2 — Top 10 by Delta2 (Over 100 == 1):")
    print(top_delta2[["Combo", "Delta2", "Over 100"]])

    # Final model selection
    valid = merged_df[merged_df["Over 100"] == 1]
    p2_metrics = (
        valid.groupby("Combo")
        .agg(Best_Delta=("Delta", "max"), Best_Delta2=("Delta2", "max"))
        .reset_index()
    )
    p2_metrics["Best_Delta2"] = p2_metrics["Best_Delta2"].fillna(0)
    p2_metrics.sort_values(
        ["Best_Delta2", "Best_Delta"], ascending=[False, False], inplace=True
    )
    p2_metrics.reset_index(drop=True, inplace=True)
    p2_metrics.to_excel(
        os.path.join(output_dir, "training_ranking.xlsx"), index=False
    )

    print("\nTRAINING RANKING (sorted by Best_Delta2):")
    print(p2_metrics.head(10).to_string(index=False))

    best_combo_name = p2_metrics.iloc[0]["Combo"]
    best_combo_idx = int(best_combo_name.split("_")[1])
    best_weights = extract_weights(combination_df, best_combo_idx)

    print(f"\n*** BEST MODEL: {best_combo_name} ***")
    print(
        f"    Training: Delta={p2_metrics.iloc[0]['Best_Delta']:.2f}%, "
        f"Delta2={p2_metrics.iloc[0]['Best_Delta2']:.2f}%"
    )

    # Save final weight table
    weight_rows = []
    for (flagging, flag_val), weight in best_weights.items():
        weight_rows.append({
            "Variable": flagging,
            "Flag_Value": flag_val,
            "Flag_Meaning": {1: "Low", 2: "Normal", 3: "High", 0: "Ref"}.get(
                flag_val, str(flag_val)
            ),
            "Weight": weight,
        })
    best_weights_df = pd.DataFrame(weight_rows)
    best_weights_df.sort_values("Weight", ascending=False, inplace=True)
    print("\nFinal model weights:")
    print(best_weights_df.to_string(index=False))
    best_weights_df.to_excel(
        os.path.join(output_dir, "FINAL_best_model_weights.xlsx"), index=False
    )

    # Pipeline summary
    summary_out = {
        "Phase1_Best_Combo": best_p1_combo,
        "Phase1_Delta": phase1_results["best_delta"],
        "Phase1_Delta2": phase1_results["best_delta2"],
        "Phase2_Best_Combo": best_combo_name,
        "Phase2_Train_Delta": p2_metrics.iloc[0]["Best_Delta"],
        "Phase2_Train_Delta2": p2_metrics.iloc[0]["Best_Delta2"],
    }
    pd.DataFrame([summary_out]).to_excel(
        os.path.join(output_dir, "FINAL_pipeline_summary.xlsx"), index=False
    )

    return {
        "best_combo": best_combo_name,
        "best_delta": p2_metrics.iloc[0]["Best_Delta"],
        "best_delta2": p2_metrics.iloc[0]["Best_Delta2"],
        "best_weights": best_weights,
        "combination_df": combination_df,
        "p2_metrics": p2_metrics,
    }


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from data_preprocessing import load_and_preprocess, split_train_validation
    from statistical_analysis import run_chi_square_analysis

    DATA_PATH = "Bronchial_cytology_CDARS_nosourcenoI_20250125_v1.xlsx"
    OUTPUT_DIR = "output_training"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    any_lung_df = load_and_preprocess(DATA_PATH)
    train_df, val_df = split_train_validation(any_lung_df, OUTPUT_DIR)
    ranked = run_chi_square_analysis(any_lung_df, OUTPUT_DIR)

    p1 = run_phase1(ranked, train_df, OUTPUT_DIR)
    p2 = run_phase2(train_df, p1, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("SCORING COMBINATION TRAINING COMPLETE")
    print("=" * 70)
