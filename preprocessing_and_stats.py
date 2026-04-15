"""
statistical_analysis.py
=======================
Chi-square feature selection with direction assignment.

For each laboratory flagging variable, performs chi-square tests
(Low vs Normal, High vs Normal) against the lung cancer outcome.
Assigns a direction (risk-increasing = 1, risk-decreasing = 0) based
on whether the cancer rate in the abnormal group exceeds the Normal group.
Assigns candidate score ranges for combinatorial search.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_FLAGGING_COLUMNS = [
    "Sex", "AGE55ormore",
    "APTT_Flagging", "Albumin_Flagging", "Basophil, absolute_Flagging",
    "C-Reactive Protein_Flagging", "Creatinine_Flagging",
    "Eosinophil, absolute_Flagging", "Haemoglobin, Blood_Flagging",
    "Lactate Dehydrogenase_Flagging", "Lymphocyte, absolute_Flagging",
    "MCH_Flagging", "MCHC_Flagging", "MCV_Flagging",
    "Neutrophil, absolute_Flagging", "Platelet_Flagging",
    "Protein, Total_Flagging", "Prothrombin Time_Flagging", "WBC_Flagging",
]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def run_chi_square_analysis(
    any_lung_df: pd.DataFrame,
    output_dir: str,
    p_threshold: float = 0.1,
) -> pd.DataFrame:
    """Perform chi-square analysis and return ranked results with scores.

    Parameters
    ----------
    any_lung_df : pd.DataFrame
        Full preprocessed dataset (used for chi-square on entire cohort).
    output_dir : str
        Directory to save results.
    p_threshold : float
        Significance threshold for candidate score assignment.

    Returns
    -------
    pd.DataFrame
        ranked_results_df with columns: Flagging, Value, Comparison,
        Chi2 Statistic, P-Value, Direction, Score.
    """
    lung_df = any_lung_df.copy()

    # ------------------------------------------------------------------
    # Step 1: Dependency table — cancer rate per (Flagging, Value)
    # ------------------------------------------------------------------
    dependency_results = []
    for flag in ALL_FLAGGING_COLUMNS:
        ct = pd.crosstab(lung_df[flag], lung_df["ANY_LUNG"])
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        for value in ct.index:
            dependency_results.append({
                "Flagging": flag,
                "Value": value,
                "Count_LUNG_0": ct.loc[value, 0] if 0 in ct.columns else 0,
                "Count_LUNG_1": ct.loc[value, 1] if 1 in ct.columns else 0,
                "Percentage_LUNG_0": pct.loc[value, 0] if 0 in pct.columns else 0,
                "Percentage_LUNG_1": pct.loc[value, 1] if 1 in pct.columns else 0,
            })
    dependency_df = pd.DataFrame(dependency_results)

    # Add missing CRP Low row (no Low observations for CRP)
    new_row = pd.DataFrame({
        "Flagging": ["C-Reactive Protein_Flagging"], "Value": [1],
        "Count_LUNG_0": [0], "Count_LUNG_1": [0],
        "Percentage_LUNG_0": [0], "Percentage_LUNG_1": [0],
    })
    dependency_df = pd.concat([dependency_df, new_row], ignore_index=True)

    # ------------------------------------------------------------------
    # Step 2: Direction — compare abnormal cancer rate vs Normal
    # ------------------------------------------------------------------
    dependency_df["Direction"] = 0
    for flagging in dependency_df["Flagging"].unique():
        sub = dependency_df[dependency_df["Flagging"] == flagging]
        if flagging in ("Sex", "AGE55ormore"):
            normal_pct = sub[sub["Value"] == 0]["Percentage_LUNG_1"].mean()
            for idx, row in sub.iterrows():
                if row["Value"] == 1 and row["Percentage_LUNG_1"] > normal_pct:
                    dependency_df.at[idx, "Direction"] = 1
        else:
            normal_vals = sub[sub["Value"] == 2]["Percentage_LUNG_1"].values
            if len(normal_vals) == 0:
                continue
            normal_pct = normal_vals[0]
            for idx, row in sub.iterrows():
                if row["Value"] in (1, 3):
                    if row["Percentage_LUNG_1"] > normal_pct:
                        dependency_df.at[idx, "Direction"] = 1

    # ------------------------------------------------------------------
    # Step 3: Chi-square tests
    # ------------------------------------------------------------------
    results = []
    for flag in ALL_FLAGGING_COLUMNS:
        if flag == "Sex":
            lung_df[f"{flag}_cmp"] = lung_df[flag].apply(
                lambda x: "Male" if x == 1 else "Female"
            )
            ct = pd.crosstab(lung_df[f"{flag}_cmp"], lung_df["ANY_LUNG"])
            chi2, p, _, _ = chi2_contingency(ct)
            results.append({
                "Flagging": flag, "Comparison": "Male vs Female",
                "Value": 1, "Chi2 Statistic": round(chi2, 5),
                "P-Value": round(p, 5),
            })
        elif flag == "AGE55ormore":
            lung_df[f"{flag}_cmp"] = lung_df[flag].apply(
                lambda x: "55OrMore" if x == 1 else "Younger"
            )
            ct = pd.crosstab(lung_df[f"{flag}_cmp"], lung_df["ANY_LUNG"])
            chi2, p, _, _ = chi2_contingency(ct)
            results.append({
                "Flagging": flag, "Comparison": "55OrMore vs Younger",
                "Value": 1, "Chi2 Statistic": round(chi2, 5),
                "P-Value": round(p, 5),
            })
        else:
            # High vs Normal
            lung_df[f"{flag}_HvN"] = lung_df[flag].apply(
                lambda x: "High" if x == 3 else ("Normal" if x == 2 else None)
            )
            ct_hn = pd.crosstab(lung_df[f"{flag}_HvN"], lung_df["ANY_LUNG"])
            chi2_hn, p_hn, _, _ = chi2_contingency(ct_hn)
            results.append({
                "Flagging": flag, "Comparison": "High vs Normal",
                "Value": 3, "Chi2 Statistic": round(chi2_hn, 5),
                "P-Value": round(p_hn, 5),
            })
            # Low vs Normal
            lung_df[f"{flag}_LvN"] = lung_df[flag].apply(
                lambda x: "Low" if x == 1 else ("Normal" if x == 2 else None)
            )
            ct_ln = pd.crosstab(lung_df[f"{flag}_LvN"], lung_df["ANY_LUNG"])
            chi2_ln, p_ln, _, _ = chi2_contingency(ct_ln)
            results.append({
                "Flagging": flag, "Comparison": "Low vs Normal",
                "Value": 1, "Chi2 Statistic": round(chi2_ln, 5),
                "P-Value": round(p_ln, 5),
            })

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Step 4: Merge dependency with chi-square results
    # ------------------------------------------------------------------
    merged_df = pd.merge(
        dependency_df,
        results_df[["Flagging", "Comparison", "Value", "Chi2 Statistic", "P-Value"]],
        on=["Flagging", "Value"],
        how="left",
    )

    # ------------------------------------------------------------------
    # Step 5: Score assignment based on p-value and direction
    # ------------------------------------------------------------------
    def _assign_score(row):
        if pd.isna(row["P-Value"]):
            return 0
        if row["P-Value"] < p_threshold and row["Direction"] == 1:
            return [0, 1, 2]
        elif row["P-Value"] < p_threshold and row["Direction"] == 0:
            return [-2, -1, 0]
        else:
            return 0

    merged_df["Score"] = merged_df.apply(_assign_score, axis=1)

    # Sort by Flagging name for consistent ordering
    ranked_results_df = (
        merged_df.sort_values(by="Flagging", ascending=False)
        .reset_index(drop=True)
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    ranked_results_df.to_excel(
        os.path.join(output_dir, "results_df_direction.xlsx")
    )
    print("Chi-square analysis complete.")

    return ranked_results_df


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess

    DATA_PATH = "Bronchial_cytology_CDARS_nosourcenoI_20250125_v1.xlsx"
    OUTPUT_DIR = "output_training"

    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    any_lung_df = load_and_preprocess(DATA_PATH)
    ranked = run_chi_square_analysis(any_lung_df, OUTPUT_DIR)
    print(f"Total variable-level rows: {len(ranked)}")
    print("Done.")
