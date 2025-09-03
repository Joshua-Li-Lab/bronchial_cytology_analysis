import pandas as pd
from tqdm import tqdm

def generate_summary(result_df):
    result_summary = []
    for combo in tqdm(result_df.columns[2:], desc= "Processing Combos"):
        unique_scores = sorted(result_df[combo].unique())
        for score in unique_scores:
            subset = result_df[result_df[combo] == score]
            count_cancer = subset[subset['Lung cancer'].isin([1, 2])].shape[0]
            count_normal = subset[subset['Lung cancer'] == 0].shape[0]
            total_count = count_cancer + count_normal
            
            cancer_leq_score = result_df[result_df[combo] <= score]['Lung cancer'].sum()
            cancer_gt_score = result_df[result_df[combo] > score]['Lung cancer'].sum()
            
            total_leq = result_df[result_df[combo] <= score]['Lung cancer'].count()
            total_gt = result_df[result_df[combo] > score]['Lung cancer'].count()
            
            cancer_percent = (count_cancer / total_count * 100) if total_count > 0 else 0
            cancer_leq_percent = (cancer_leq_score / total_leq * 100) if total_leq > 0 else 0
            cancer_gt_percent = (cancer_gt_score / total_gt * 100) if total_gt > 0 else 0
            
            delta = abs(cancer_leq_percent - cancer_gt_percent)
            over_100 = int(total_leq > 100 and total_gt > 100)
            
            result_summary.append({
                'Combo': combo,
                'Score': score,
                'Cancer Cases': count_cancer,
                'Normal Cases': count_normal,
                'Total Patients': total_count,
                'Cancer %': cancer_percent,
                'Cancer <= Score': cancer_leq_score,
                'Cancer > Score': cancer_gt_score,
                'Total <= Score': total_leq,
                'Total > Score': total_gt,
                'Cancer <= Score %': cancer_leq_percent,
                'Cancer > Score %': cancer_gt_percent,
                'Delta': delta,
                'Over 100': over_100
            })
    
    summary_df = pd.DataFrame(result_summary)
    return summary_df

def generate_delta2_summary(summary_df):
    delta2_summary_df = summary_df[summary_df['Over 100'] == 1]
    results = []
    for combo, group in delta2_summary_df.groupby('Combo'):
        for index, row in group.iterrows():
            cancer_less_score = row['Cancer <= Score %']
            score = row['Score']
            max_difference = 0
            compared_score = None
            compared_cancer_greater = None
            for comp_index, comp_row in group.iterrows():
                if comp_index != index:
                    cancer_greater_score = comp_row['Cancer > Score %']
                    difference = abs(cancer_less_score - cancer_greater_score)
                    if difference > max_difference:
                        max_difference = difference
                        compared_score = comp_row['Score']
                        compared_cancer_greater = cancer_greater_score
            results.append({
                'Combo': combo,
                'Score': score,
                'Cancer <= Score %': cancer_less_score,
                'Compared Score': compared_score,
                'Compared Cancer > Score %': compared_cancer_greater,
                'Delta2': max_difference
            })
    delta2_df = pd.DataFrame(results)
    return delta2_df

def merge_summaries(summary_df, delta2_df):
    merged_df = pd.merge(summary_df, delta2_df[['Combo', 'Score', 'Compared Score', 'Compared Cancer > Score %', 'Delta2']],
                         on=['Combo', 'Score'], how='left')
    return merged_df

