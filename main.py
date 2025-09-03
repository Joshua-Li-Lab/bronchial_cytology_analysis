from data_preprocessing import load_and_preprocess_data, split_and_export_data
from statistical_analysis import create_dependency_table, perform_chi_square_tests, merge_and_assign_scores
from scoring_combinations import generate_score_combinations, apply_combinations_to_data
from summary_generation import generate_summary, generate_delta2_summary, merge_summaries

def main():
    input_file = 'data.xlsx'

    merged_summary_output = 'summary_output.xlsx'
    
    # Preprocess data
    any_lung_df = load_and_preprocess_data(input_file)
    train_df, val_df = split_and_export_data(any_lung_df)
    
    # Statistical analysis
    dependency_df = create_dependency_table(any_lung_df)
    results_df = perform_chi_square_tests(any_lung_df)
    merged_df = merge_and_assign_scores(dependency_df, results_df)
    ranked_results_df = merged_df.sort_values(by='Flagging', ascending=False).reset_index(drop=True)
    
    combination_df = generate_score_combinations(ranked_results_df)
    
    combination_result = apply_combinations_to_data(train_df, combination_df)
    
    summary_df = generate_summary(combination_result)
    
    delta2_df = generate_delta2_summary(summary_df)

    merged_summary = merge_summaries(summary_df, delta2_df)
    
    merged_summary.to_excel(merged_summary_output, index=False)
    
    print("Pipeline completed. Outputs saved in current directory.")

if __name__ == "__main__":
    main()