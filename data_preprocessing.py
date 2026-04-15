"""
data_preprocessing.py
Loads bronchial cytology data, calculates age, encodes categorical variables,
splits into training/validation sets, and runs chi-square feature selection
with direction-aware score assignment for the combinatorial scoring pipeline.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

COLUMNS_TO_KEEP = [
    "Number", "Sex", "AGE55ormore",
    "APTT_Flagging", "Albumin_Flagging", "Basophil, absolute_Flagging",
    "C-Reactive Protein_Flagging", "Creatinine_Flagging",
    "Eosinophil, absolute_Flagging", "Haemoglobin, Blood_Flagging",
    "Lactate Dehydrogenase_Flagging", "Lymphocyte, absolute_Flagging",
    "MCH_Flagging", "MCHC_Flagging", "MCV_Flagging",
    "Neutrophil, absolute_Flagging", "Platelet_Flagging",
    "Protein, Total_Flagging", "Prothrombin Time_Flagging",
    "WBC_Flagging", "ANY_LUNG"
]

HIGH_NORMAL_LOW_COLUMNS = [
    'APTT_Flagging', 'Albumin_Flagging', 'Basophil, absolute_Flagging',
    'C-Reactive Protein_Flagging', 'Creatinine_Flagging',
    'Eosinophil, absolute_Flagging', 'Haemoglobin, Blood_Flagging',
    'Lactate Dehydrogenase_Flagging', 'Lymphocyte, absolute_Flagging',
    'MCH_Flagging', 'MCHC_Flagging', 'MCV_Flagging',
    'Neutrophil, absolute_Flagging', 'Platelet_Flagging',
    'Protein, Total_Flagging', 'Prothrombin Time_Flagging', 'WBC_Flagging'
]

FLAGGING_COLUMNS = [
    "Sex", "AGE55ormore",
    "APTT_Flagging", "Albumin_Flagging", "Basophil, absolute_Flagging",
    "C-Reactive Protein_Flagging", "Creatinine_Flagging",
    "Eosinophil, absolute_Flagging", "Haemoglobin, Blood_Flagging",
    "Lactate Dehydrogenase_Flagging", "Lymphocyte, absolute_Flagging",
    "MCH_Flagging", "MCHC_Flagging", "MCV_Flagging",
    "Neutrophil, absolute_Flagging", "Platelet_Flagging",
    "Protein, Total_Flagging", "Prothrombin Time_Flagging", "WBC_Flagging"
]


def load_and_preprocess(filepath):
    """Load Excel data, compute age, filter C3/C4, and encode variables."""
    df = pd.read_excel(filepath)

    df['Date of Birth (yyyy-mm-dd)'] = pd.to_datetime(df['Date of Birth (yyyy-mm-dd)'])
    df['Admission Date (yyyy-mm-dd)'] = pd.to_datetime(df['Admission Date (yyyy-mm-dd)'])
    df['age'] = (
        df['Admission Date (yyyy-mm-dd)'].dt.year
        - df['Date of Birth (yyyy-mm-dd)'].dt.year
    ) - (
        (df['Admission Date (yyyy-mm-dd)'].dt.month < df['Date of Birth (yyyy-mm-dd)'].dt.month) |
        ((df['Admission Date (yyyy-mm-dd)'].dt.month == df['Date of Birth (yyyy-mm-dd)'].dt.month) &
         (df['Admission Date (yyyy-mm-dd)'].dt.day < df['Date of Birth (yyyy-mm-dd)'].dt.day))
    )

    df['AGE55ormore'] = (df['age'] >= 55).astype(int)
    df = df[df['status'].isin(['C3', 'C4'])]
    df['ANY_LUNG'] = df['ANY_LUNG'].replace({1: 1, 2: 1})

    any_lung_df = df[COLUMNS_TO_KEEP].copy()

    any_lung_df[HIGH_NORMAL_LOW_COLUMNS] = any_lung_df[HIGH_NORMAL_LOW_COLUMNS].replace(
        {'H': 3, np.nan: 2, 'L': 1}
    )
    any_lung_df[['Sex']] = any_lung_df[['Sex']].replace({'M': 1, 'F': 0})

    return any_lung_df


def split_data(any_lung_df, test_size=0.5, random_state=42):
    """Split into training and validation sets with stratification by index."""
    train_df, val_df = train_test_split(
        any_lung_df, test_size=test_size, random_state=random_state
    )
    return train_df, val_df


def run_chi_square_analysis(any_lung_df):
    """Run chi-square tests on each flagging variable vs ANY_LUNG.

    Produces a ranked DataFrame with direction-aware score assignments:
      - p < 0.1 and Direction=1 (higher cancer%) → [0, 1, 2]
      - p < 0.1 and Direction=0 (lower cancer%)  → [-2, -1, 0]
      - otherwise → 0
    """
    lung_df = any_lung_df.copy()

    # --- Dependency table with direction ---
    dependency_results = []
    for flag in FLAGGING_COLUMNS:
        ct = pd.crosstab(lung_df[flag], lung_df['ANY_LUNG'])
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        for value in ct.index:
            dependency_results.append({
                'Flagging': flag, 'Value': value,
                'Count_LUNG_0': ct.loc[value, 0] if 0 in ct.columns else 0,
                'Count_LUNG_1': ct.loc[value, 1] if 1 in ct.columns else 0,
                'Percentage_LUNG_0': pct.loc[value, 0] if 0 in pct.columns else 0,
                'Percentage_LUNG_1': pct.loc[value, 1] if 1 in pct.columns else 0,
            })
    dependency_df = pd.DataFrame(dependency_results)

    # Add missing CRP Low row
    new_row = pd.DataFrame({
        "Flagging": ["C-Reactive Protein_Flagging"], "Value": [1],
        "Count_LUNG_0": [0], "Count_LUNG_1": [0],
        "Percentage_LUNG_0": [0], "Percentage_LUNG_1": [0]
    })
    dependency_df = pd.concat([dependency_df, new_row], ignore_index=True)

    # Compute direction: 1 if cancer% > Normal cancer%, else 0
    dependency_df['Direction'] = 0
    for flagging in dependency_df['Flagging'].unique():
        flagging_df = dependency_df[dependency_df['Flagging'] == flagging]
        if flagging in ['Sex', 'AGE55ormore']:
            normal_pct = flagging_df[flagging_df['Value'] == 0]['Percentage_LUNG_1'].mean()
            for idx, row in flagging_df.iterrows():
                if row['Value'] == 1:
                    if row['Percentage_LUNG_1'] > normal_pct:
                        dependency_df.at[idx, 'Direction'] = 1
        else:
            normal_vals = flagging_df[flagging_df['Value'] == 2]['Percentage_LUNG_1'].values
            if len(normal_vals) == 0:
                continue
            normal_pct = normal_vals[0]
            for idx, row in flagging_df.iterrows():
                if row['Value'] in [1, 3]:
                    if row['Percentage_LUNG_1'] > normal_pct:
                        dependency_df.at[idx, 'Direction'] = 1

    # --- Chi-square tests ---
    results = []
    for flag in FLAGGING_COLUMNS:
        if flag == 'Sex':
            lung_df[f'{flag}_cmp'] = lung_df[flag].apply(
                lambda x: 'Male' if x == 1 else 'Female')
            ct = pd.crosstab(lung_df[f'{flag}_cmp'], lung_df['ANY_LUNG'])
            chi2, p, _, _ = chi2_contingency(ct)
            results.append({'Flagging': flag, 'Comparison': 'Male vs Female',
                            'Value': 1, 'Chi2 Statistic': round(chi2, 5),
                            'P-Value': round(p, 5)})
        elif flag == 'AGE55ormore':
            lung_df[f'{flag}_cmp'] = lung_df[flag].apply(
                lambda x: '55OrMore' if x == 1 else 'Younger')
            ct = pd.crosstab(lung_df[f'{flag}_cmp'], lung_df['ANY_LUNG'])
            chi2, p, _, _ = chi2_contingency(ct)
            results.append({'Flagging': flag, 'Comparison': '55OrMore vs Younger',
                            'Value': 1, 'Chi2 Statistic': round(chi2, 5),
                            'P-Value': round(p, 5)})
        else:
            # High vs Normal
            lung_df[f'{flag}_HvN'] = lung_df[flag].apply(
                lambda x: 'High' if x == 3 else ('Normal' if x == 2 else None))
            ct_hn = pd.crosstab(lung_df[f'{flag}_HvN'], lung_df['ANY_LUNG'])
            chi2_hn, p_hn, _, _ = chi2_contingency(ct_hn)
            results.append({'Flagging': flag, 'Comparison': 'High vs Normal',
                            'Value': 3, 'Chi2 Statistic': round(chi2_hn, 5),
                            'P-Value': round(p_hn, 5)})
            # Low vs Normal
            lung_df[f'{flag}_LvN'] = lung_df[flag].apply(
                lambda x: 'Low' if x == 1 else ('Normal' if x == 2 else None))
            ct_ln = pd.crosstab(lung_df[f'{flag}_LvN'], lung_df['ANY_LUNG'])
            chi2_ln, p_ln, _, _ = chi2_contingency(ct_ln)
            results.append({'Flagging': flag, 'Comparison': 'Low vs Normal',
                            'Value': 1, 'Chi2 Statistic': round(chi2_ln, 5),
                            'P-Value': round(p_ln, 5)})

    results_df = pd.DataFrame(results)

    # --- Merge and assign scores ---
    merged_df = pd.merge(
        dependency_df,
        results_df[['Flagging', 'Comparison', 'Value', 'Chi2 Statistic', 'P-Value']],
        on=['Flagging', 'Value'], how='left'
    )

    def score_assign(row):
        if pd.isna(row['P-Value']):
            return 0
        if row['P-Value'] < 0.1 and row['Direction'] == 1:
            return [0, 1, 2]
        elif row['P-Value'] < 0.1 and row['Direction'] == 0:
            return [-2, -1, 0]
        else:
            return 0

    merged_df['Score'] = merged_df.apply(score_assign, axis=1)

    ranked_results_df = merged_df.sort_values(
        by='Flagging', ascending=False
    ).reset_index(drop=True)

    return ranked_results_df
