"""
Fraud Detection - Dealing with Imbalanced Data
This script loads credit card fraud data, performs sanity checks,
analyzes class imbalance, and visualizes the results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "creditcard.csv"


def download_data_if_needed():
    """
    Download the credit card fraud dataset if it doesn't exist.
    Note: Requires Kaggle API authentication.
    """
    if DATA_FILE.exists():
        print(f"✓ Data file already exists at {DATA_FILE}")
        return True
    
    print(f"Data file not found at {DATA_FILE}")
    print("Attempting to download from Kaggle...")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=DATA_DIR,
            unzip=True
        )
        
        print(f"✓ Successfully downloaded data to {DATA_DIR}")
        return True
        
    except ImportError:
        print("✗ Kaggle API not installed. Install with: pip install kaggle")
        print("  Also ensure your Kaggle API credentials are set up.")
        return False
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return False


def load_data():
    """Load the credit card fraud dataset."""
    if not DATA_FILE.exists():
        print(f"✗ Data file not found at {DATA_FILE}")
        return None
    
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape} (rows, columns)")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def sanity_check(df):
    """
    Perform sanity checks on the data.
    Checks for missing values and basic statistics.
    """
    print("\n" + "="*50)
    print("SANITY CHECK - Data Quality")
    print("="*50)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_count = missing_values.sum()
    
    if missing_count == 0:
        print("✓ No missing values detected")
    else:
        print(f"✗ Found {missing_count} missing values:")
        print(missing_values[missing_values > 0])
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        print("✓ No duplicate rows detected")
    else:
        print(f"⚠ Found {duplicate_count} duplicate rows")
    
    # Basic statistics
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Total features: {df.shape[1]}")
    print(f"  Data types:\n{df.dtypes}")


def analyze_imbalance(df):
    """
    Analyze the class imbalance in the dataset.
    Assumes the target variable is named 'Class'.
    """
    print("\n" + "="*50)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*50)
    
    if 'Class' not in df.columns:
        print("✗ 'Class' column not found in dataset")
        return None
    
    # Get class distribution
    class_distribution = df['Class'].value_counts().sort_index()
    class_percentages = (df['Class'].value_counts(normalize=True).sort_index() * 100)
    
    print("\nClass Distribution:")
    for class_label in class_distribution.index:
        count = class_distribution[class_label]
        percentage = class_percentages[class_label]
        label = "Legitimate" if class_label == 0 else "Fraudulent"
        print(f"  Class {class_label} ({label}): {count:,} samples ({percentage:.2f}%)")
    
    # Calculate imbalance ratio
    fraud_count = class_distribution.get(1, 0)
    legit_count = class_distribution.get(0, 0)
    
    if fraud_count > 0 and legit_count > 0:
        imbalance_ratio = legit_count / fraud_count
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1 (Legitimate:Fraudulent)")
    
    return class_distribution


def visualize_imbalance(class_distribution):
    """Create a bar plot showing the class imbalance."""
    if class_distribution is None:
        print("✗ Cannot create visualization: no class distribution data")
        return
    
    print("\n" + "="*50)
    print("Creating Imbalance Visualization")
    print("="*50)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Count plot
    labels = ['Legitimate (0)', 'Fraudulent (1)']
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].bar(labels, class_distribution.values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution - Count', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(class_distribution.values):
        axes[0].text(i, v + 1000, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Percentage plot
    percentages = (class_distribution.values / class_distribution.sum()) * 100
    axes[1].bar(labels, percentages, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Class Distribution - Percentage', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, v in enumerate(percentages):
        axes[1].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(__file__).parent / "imbalance_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    
    # Show the plot
    plt.show()


def main():
    """Main execution function."""
    print("\n" + "="*50)
    print("FRAUD DETECTION - IMBALANCED DATA ANALYSIS")
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
        visualize_imbalance(class_distribution)
    
    print("\n" + "="*50)
    print("✓ Analysis Complete!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
