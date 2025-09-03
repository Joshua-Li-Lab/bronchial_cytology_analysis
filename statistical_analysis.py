import pandas as pd
from scipy.stats import chi2_contingency

def create_dependency_table(df, target_column='Lung cancer'):
    flagging_columns = [
        "Sex", "AGE55ormore", "APTT_Flagging", "Albumin_Flagging",
        "Basophil, absolute_Flagging", "C-Reactive Protein_Flagging", "Creatinine_Flagging",
        "Eosinophil, absolute_Flagging", "Haemoglobin, Blood_Flagging",
        "Lactate Dehydrogenase_Flagging", "Lymphocyte, absolute_Flagging",
        "MCH_Flagging", "MCHC_Flagging", "MCV_Flagging", "Neutrophil, absolute_Flagging",
        "Platelet_Flagging", "Protein, Total_Flagging", "Prothrombin Time_Flagging",
        "WBC_Flagging"
    ]
    
    dependency_results = []
    for flag in flagging_columns:
        contingency_table = pd.crosstab(df[flag], df[target_column])
        count_values = contingency_table
        percentage_values = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
        
        for value in count_values.index:
            dependency_results.append({
                'Flagging': flag,
                'Value': value,
                'Count_LUNG_0': count_values.loc[value, 0] if 0 in count_values.columns else 0,
                'Count_LUNG_1': count_values.loc[value, 1] if 1 in count_values.columns else 0,
                'Percentage_LUNG_0': percentage_values.loc[value, 0] if 0 in percentage_values.columns else 0,
                'Percentage_LUNG_1': percentage_values.loc[value, 1] if 1 in percentage_values.columns else 0,
            })
    
    dependency_df = pd.DataFrame(dependency_results)
    
    dependency_df['Direction'] = 0
    for flagging in dependency_df['Flagging'].unique():
        flagging_df = dependency_df[dependency_df['Flagging'] == flagging]
        if flagging in ['Sex', 'AGE55ormore']:
            normal_percentage = flagging_df[flagging_df['Value'] == 0]['Percentage_LUNG_1'][0]
            for index, row in flagging_df.iterrows():
                if row['Value'] == 1:
                    if row['Percentage_LUNG_1'] > normal_percentage:
                        dependency_df.at[index, 'Direction'] = 1
                elif row['Value'] == 0:
                    if row['Percentage_LUNG_1'] >= normal_percentage:
                        dependency_df.at[index, 'Direction'] = 1
        else:
            normal_percentage = flagging_df[flagging_df['Value'] == 2]['Percentage_LUNG_1'].values[0]
            for index, row in flagging_df.iterrows():
                if row['Value'] == 3:
                    if row['Percentage_LUNG_1'] > normal_percentage:
                        dependency_df.at[index, 'Direction'] = 1
                elif row['Value'] == 1:
                    if row['Percentage_LUNG_1'] > normal_percentage:
                        dependency_df.at[index, 'Direction'] = 1
    
    dependency_df['Total'] = dependency_df['Count_LUNG_0'] + dependency_df['Count_LUNG_1']
    return dependency_df

def perform_chi_square_tests(df, target_column='Lung cancer'):
    flagging_columns = [
        "Sex", "AGE55ormore", "APTT_Flagging", "Albumin_Flagging",
        "Basophil, absolute_Flagging", "C-Reactive Protein_Flagging", "Creatinine_Flagging",
        "Eosinophil, absolute_Flagging", "Haemoglobin, Blood_Flagging",
        "Lactate Dehydrogenase_Flagging", "Lymphocyte, absolute_Flagging",
        "MCH_Flagging", "MCHC_Flagging", "MCV_Flagging", "Neutrophil, absolute_Flagging",
        "Platelet_Flagging", "Protein, Total_Flagging", "Prothrombin Time_Flagging",
        "WBC_Flagging"
    ]
    
    results = []
    for flag in flagging_columns:
        if flag == 'Sex':
            df[f'{flag}_Male_vs_Female'] = df[flag].apply(lambda x: 'Male' if x == 1 else 'Female')
            contingency_table = pd.crosstab(df[f'{flag}_Male_vs_Female'], df[target_column])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append({
                'Flagging': flag,
                'Comparison': 'Male vs Female',
                'Value': 1,
                'Chi2 Statistic': round(chi2, 5),
                'P-Value': round(p, 5)
            })
        elif flag == 'AGE55ormore':
            df[f'{flag}_55OrMore_vs_Younger'] = df[flag].apply(lambda x: '55OrMore' if x == 1 else 'Younger')
            contingency_table = pd.crosstab(df[f'{flag}_55OrMore_vs_Younger'], df[target_column])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results.append({
                'Flagging': flag,
                'Comparison': '55OrMore vs Younger',
                'Value': 1,
                'Chi2 Statistic': round(chi2, 5),
                'P-Value': round(p, 5)
            })
        else:
            df[f'{flag}_High_vs_Normal'] = df[flag].apply(lambda x: 'High' if x == 3 else ('Normal' if x == 2 else None))
            df[f'{flag}_Low_vs_Normal'] = df[flag].apply(lambda x: 'Low' if x == 1 else ('Normal' if x == 2 else None))
            
            contingency_table_high = pd.crosstab(df[f'{flag}_High_vs_Normal'], df[target_column])
            chi2_high, p_high, _, _ = chi2_contingency(contingency_table_high)
            results.append({
                'Flagging': flag,
                'Comparison': 'High vs Normal',
                'Value': 3,
                'Chi2 Statistic': round(chi2_high, 5),
                'P-Value': round(p_high, 5)
            })
            
            contingency_table_low = pd.crosstab(df[f'{flag}_Low_vs_Normal'], df[target_column])
            chi2_low, p_low, _, _ = chi2_contingency(contingency_table_low)
            results.append({
                'Flagging': flag,
                'Comparison': 'Low vs Normal',
                'Value': 1,
                'Chi2 Statistic': round(chi2_low, 5),
                'P-Value': round(p_low, 5)
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def merge_and_assign_scores(dependency_df, results_df):
    merged_df = pd.merge(dependency_df, results_df[['Flagging', 'Comparison', 'Value', 'Chi2 Statistic', 'P-Value']],
                         on=['Flagging', 'Value'], how='left')
    
    def score_assign(row):
        if row['P-Value'] < 0.1 and row['Direction'] == 1:
            return [0, 1, 2]
        elif row['P-Value'] < 0.1 and row['Direction'] == 0:
            return [-2, -1, 0]
        else:
            return 0
    
    merged_df['Score'] = merged_df.apply(score_assign, axis=1)
    
    sorted_df = (
        merged_df[merged_df['P-Value'] < 0.1]
        .sort_values(by='Chi2 Statistic', ascending=False)
        .reset_index(drop=True)
        )
    
    return sorted_df

