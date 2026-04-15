"""
main.py
Orchestrates the full bronchial cytology scoring pipeline:
  1. Data preprocessing and chi-square feature selection
  2. Two-phase combinatorial model training
  3. Internal validation on a held-out set
"""

import os
from data_preprocessing import load_and_preprocess, split_data, run_chi_square_analysis
from scoring_combinations import run_training
from scoring_validation import run_validation

# =========================================================================
# Configuration
# =========================================================================
DATA_FILE = "data.xlsx"
OUTPUT_DIR = "output"
BASE_SCORE = 15
HIGH_RISK_CUTOFF = 17
LOW_RISK_CUTOFF = 8


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # Step 1: Data Preprocessing
    # -----------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Data Loading and Preprocessing")
    print("=" * 70)

    any_lung_df = load_and_preprocess(DATA_FILE)
    print(f"Total C3/C4 samples: {len(any_lung_df)}")
    print(f"Positive (ANY_LUNG=1): {any_lung_df['ANY_LUNG'].sum()}")

    train_df, val_df = split_data(any_lung_df)
    train_df.to_excel(os.path.join(OUTPUT_DIR, 'train_any_lung_df.xlsx'))
    val_df.to_excel(os.path.join(OUTPUT_DIR, 'val_any_lung_df.xlsx'))
    print(f"Training: {len(train_df)}, Validation: {len(val_df)}")

    # Chi-square analysis is run on the full dataset
    ranked_results_df = run_chi_square_analysis(any_lung_df)
    print("Chi-square analysis complete.")

    # -----------------------------------------------------------------
    # Step 2: Model Training (Phase 1 + Phase 2)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Combinatorial Model Training")
    print("=" * 70)

    final_weights, best_weights_df, pipeline_summary = run_training(
        ranked_results_df, train_df, OUTPUT_DIR
    )

    # Build validation-ready weights (non-zero entries only)
    validation_weights = {k: v for k, v in final_weights.items()}

    # -----------------------------------------------------------------
    # Step 3: Internal Validation
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Internal Validation")
    print("=" * 70)

    run_validation(
        val_df, validation_weights,
        base_score=BASE_SCORE,
        high_cutoff=HIGH_RISK_CUTOFF,
        low_cutoff=LOW_RISK_CUTOFF,
        output_dir=OUTPUT_DIR
    )

    # -----------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"All output files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
