"""
Fraud Detection Utilities
Data loading, EDA, and visualization functions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def download_data_if_needed():
    """Download the credit card fraud dataset if it doesn't exist."""
    DATA_DIR = Path(__file__).parent / "data"
    DATA_FILE = DATA_DIR / "creditcard.csv"
    
    if DATA_FILE.exists():
        print(f"✓ Data file already exists at {DATA_FILE}")
        return True
    
    print(f"Data file not found at {DATA_FILE}")
    print("Attempting to download from Kaggle...")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
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
    DATA_DIR = Path(__file__).parent / "data"
    DATA_FILE = DATA_DIR / "creditcard.csv"
    
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
    """Perform sanity checks on the data."""
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
    """Analyze the class imbalance in the dataset."""
    print("\n" + "="*50)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*50)
    
    if 'Class' not in df.columns:
        print("✗ 'Class' column not found in dataset")
        return None
    
    class_distribution = df['Class'].value_counts().sort_index()
    class_percentages = (df['Class'].value_counts(normalize=True).sort_index() * 100)
    
    print("\nClass Distribution:")
    for class_label in class_distribution.index:
        count = class_distribution[class_label]
        percentage = class_percentages[class_label]
        label = "Legitimate" if class_label == 0 else "Fraudulent"
        print(f"  Class {class_label} ({label}): {count:,} samples ({percentage:.2f}%)")
    
    fraud_count = class_distribution.get(1, 0)
    legit_count = class_distribution.get(0, 0)
    
    if fraud_count > 0 and legit_count > 0:
        imbalance_ratio = legit_count / fraud_count
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1 (Legitimate:Fraudulent)")
    
    return class_distribution


def visualize_imbalance(class_distribution, output_dir='output'):
    """Create a bar plot showing the class imbalance."""
    if class_distribution is None:
        print("✗ Cannot create visualization: no class distribution data")
        return
    
    print("\n" + "="*50)
    print("Creating Imbalance Visualization")
    print("="*50)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    labels = ['Legitimate (0)', 'Fraudulent (1)']
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].bar(labels, class_distribution.values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution - Count', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(class_distribution.values):
        axes[0].text(i, v + 1000, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    percentages = (class_distribution.values / class_distribution.sum()) * 100
    axes[1].bar(labels, percentages, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Class Distribution - Percentage', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(percentages):
        axes[1].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "imbalance_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()


def perform_eda(df, output_dir='output'):
    """Perform comprehensive Exploratory Data Analysis."""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print("\n1. Statistical Summary:")
    print(df.describe())
    
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]
    
    print("\n2. Amount and Time Analysis:")
    print(f"\nAmount Statistics:")
    print(f"  Fraud - Mean: ${fraud['Amount'].mean():.2f}, Median: ${fraud['Amount'].median():.2f}")
    print(f"  Legit - Mean: ${legit['Amount'].mean():.2f}, Median: ${legit['Amount'].median():.2f}")
    
    print(f"\nTime Statistics:")
    print(f"  Fraud - Mean: {fraud['Time'].mean():.2f}s, Median: {fraud['Time'].median():.2f}s")
    print(f"  Legit - Mean: {legit['Time'].mean():.2f}s, Median: {legit['Time'].median():.2f}s")
    
    create_eda_visualizations(df, fraud, legit, output_dir)
    analyze_correlations(df)
    
    return df


def create_eda_visualizations(df, fraud, legit, output_dir='output'):
    """Create comprehensive EDA visualizations."""
    print("\n3. Creating EDA Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Amount distribution
    axes[0, 0].hist(legit['Amount'], bins=50, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
    axes[0, 0].hist(fraud['Amount'], bins=50, alpha=0.6, label='Fraudulent', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Amount ($)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Transaction Amount Distribution by Class', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 500)
    
    # Time distribution
    axes[0, 1].hist(legit['Time'], bins=50, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
    axes[0, 1].hist(fraud['Time'], bins=50, alpha=0.6, label='Fraudulent', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Time (seconds)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Transaction Time Distribution by Class', fontweight='bold')
    axes[0, 1].legend()
    
    # Box plots
    data_to_plot = [legit['Amount'], fraud['Amount']]
    axes[1, 0].boxplot(data_to_plot, labels=['Legitimate', 'Fraudulent'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_ylabel('Amount ($)', fontweight='bold')
    axes[1, 0].set_title('Amount Distribution - Box Plot', fontweight='bold')
    axes[1, 0].set_ylim(0, 300)
    
    # Fraud rate over time
    df_time = df.copy()
    df_time['Time_hours'] = df_time['Time'] / 3600
    time_bins = pd.cut(df_time['Time_hours'], bins=48)
    fraud_rate = df_time.groupby(time_bins)['Class'].mean() * 100
    
    axes[1, 1].plot(range(len(fraud_rate)), fraud_rate.values, linewidth=2, color='red')
    axes[1, 1].set_xlabel('Time Period', fontweight='bold')
    axes[1, 1].set_ylabel('Fraud Rate (%)', fontweight='bold')
    axes[1, 1].set_title('Fraud Rate Over Time', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "eda_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Distribution plots saved to {output_path}")
    plt.close()
    
    create_feature_importance_plot(df, output_dir)


def create_feature_importance_plot(df, output_dir='output'):
    """Create visualization of feature correlations with target."""
    correlations = df.corr()['Class'].drop('Class').sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ['red' if x > 0 else 'blue' for x in correlations]
    correlations.plot(kind='barh', ax=ax, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Correlation with Fraud Class', fontweight='bold')
    ax.set_title('Feature Correlations with Fraud Detection', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "feature_correlations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Feature correlation plot saved to {output_path}")
    plt.close()


def analyze_correlations(df):
    """Analyze and report feature correlations."""
    print("\n4. Correlation Analysis:")
    
    correlations = df.corr()['Class'].drop('Class').sort_values(ascending=False)
    
    print("\n  Top 5 Positive Correlations with Fraud:")
    for i, (feature, corr) in enumerate(correlations.head(5).items(), 1):
        print(f"    {i}. {feature}: {corr:.4f}")
    
    print("\n  Top 5 Negative Correlations with Fraud:")
    for i, (feature, corr) in enumerate(correlations.tail(5).items(), 1):
        print(f"    {i}. {feature}: {corr:.4f}")
