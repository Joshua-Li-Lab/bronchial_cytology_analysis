import itertools
import pandas as pd
from tqdm import tqdm

def generate_score_combinations(sorted_df):
    score_dict = {}
    for index, row in sorted_df.iterrows():
        key = (row['Flagging'], row['Value'])
        if isinstance(row['Score'], list):
            score_dict[key] = row['Score']
        else:
            score_dict[key] = [row['Score']]
    
    all_keys = list(score_dict.keys())
    score_combinations = list(itertools.product(*(score_dict[key] for key in all_keys)))
    
    results = []
    for combination in score_combinations:
        formatted_result = {
            f"{flagging} comparison {comparison}": score
            for (flagging, comparison), score in zip(all_keys, combination)
        }
        results.append(formatted_result)

    
    combination_df = pd.DataFrame(results)
    return combination_df

def apply_combinations_to_data(train_df, combination_df):
    results = []
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0], desc="Processing Rows"):
        row_results = {
            'Patient number': row['Patient number'],
            'Lung cancer': row['Lung cancer']
        }
        for combo_index in range(len(combination_df)):
            score = 0
            for column in train_df.columns[2:]:
                comparison_value = row[column]
                comparison_column = f'{column} comparison {comparison_value}'
                if comparison_column in combination_df.columns:
                    score += combination_df.at[combo_index, comparison_column]
            row_results[f'Combo_{combo_index}'] = score
        results.append(row_results)
    
    result_df = pd.DataFrame(results)
    return result_df