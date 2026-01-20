"""
Fraud Detection - Main Orchestrator
Runs the complete ML pipeline for fraud detection with imbalanced data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import training functions
from train_model import (
    preprocess_data, apply_sampling_techniques, train_models,
    evaluate_models, save_best_model
)

# Import EDA and data handling functions
from fraud_detection_utils import (
    download_data_if_needed, load_data, sanity_check,
    analyze_imbalance, visualize_imbalance, perform_eda
)

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "creditcard.csv"
OUTPUT_DIR = Path(__file__).parent / "output"
MODELS_DIR = Path(__file__).parent / "models"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def main():
    """Main execution function."""
    print("\n" + "="*50)
    print("FRAUD DETECTION - COMPLETE ML PIPELINE")
    print("="*50)
    
    # Step 1: Download data if needed
    if not download_data_if_needed():
        print("\n✗ Failed to ensure data availability")
        return
    
    # Step 2: Load the data
    df = load_data()
    if df is None:
        print("\n✗ Failed to load data")
        return
    
    # Step 3: Sanity check
    sanity_check(df)
    
    # Step 4: Analyze imbalance
    class_distribution = analyze_imbalance(df)
    
    # Step 5: Visualize imbalance
    if class_distribution is not None:
        visualize_imbalance(class_distribution, OUTPUT_DIR)
    
    # Step 6: Exploratory Data Analysis
    df = perform_eda(df, OUTPUT_DIR)
    
    # Step 7: Preprocess data
    data_dict = preprocess_data(df)
    
    # Step 8: Apply sampling techniques
    sampling_strategies = apply_sampling_techniques(
        data_dict['X_train'], 
        data_dict['y_train']
    )
    
    # Step 9: Train models
    trained_models = train_models(
        sampling_strategies,
        data_dict['X_val'],
        data_dict['y_val']
    )
    
    # Step 10: Evaluate models and create comparisons
    results_df, best_model_key, best_model_info = evaluate_models(
        trained_models,
        data_dict['X_test'],
        data_dict['y_test'],
        output_dir=OUTPUT_DIR
    )
    
    # Step 11: Save best model
    save_best_model(best_model_info, data_dict, models_dir=MODELS_DIR)
    
    print("\n" + "="*50)
    print("✓ COMPLETE PIPELINE FINISHED!")
    print("="*50)
    print(f"\nBest Model: {best_model_info['model_name']}")
    print(f"Sampling Strategy: {best_model_info['sampling']}")
    print(f"\nGenerated Files:")
    print(f"  📊 Visualizations saved to: {OUTPUT_DIR}/")
    print(f"  🤖 Models saved to: {MODELS_DIR}/")
    print(f"\nKey outputs:")
    print(f"  - confusion_matrices_all_models.png")
    print(f"  - metrics_comparison_chart.png")
    print(f"  - model_comparison_results.csv")
    print(f"\nYou can now run 'test_model.py' to verify predictions on test data.")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
