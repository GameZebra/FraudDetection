"""
Fraud Detection - Dealing with Imbalanced Data
This script implements a complete fraud detection pipeline including:
- Data loading and quality checks
- Exploratory Data Analysis (EDA)
- Imbalanced data handling (SMOTE, undersampling)
- Multiple ML model training and evaluation
- Model persistence and prediction
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
from scipy import stats

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score, 
    matthews_corrcoef, average_precision_score
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Imbalanced data handling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Model persistence
import joblib

warnings.filterwarnings('ignore')

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
    plt.close()


def perform_eda(df):
    """
    Perform comprehensive Exploratory Data Analysis.
    Analyzes feature distributions, correlations, and relationships with target.
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 1. Statistical Summary
    print("\n1. Statistical Summary:")
    print(df.describe())
    
    # 2. Amount and Time analysis
    print("\n2. Amount and Time Analysis:")
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]
    
    print(f"\nAmount Statistics:")
    print(f"  Fraud - Mean: ${fraud['Amount'].mean():.2f}, Median: ${fraud['Amount'].median():.2f}")
    print(f"  Legit - Mean: ${legit['Amount'].mean():.2f}, Median: ${legit['Amount'].median():.2f}")
    
    print(f"\nTime Statistics:")
    print(f"  Fraud - Mean: {fraud['Time'].mean():.2f}s, Median: {fraud['Time'].median():.2f}s")
    print(f"  Legit - Mean: {legit['Time'].mean():.2f}s, Median: {legit['Time'].median():.2f}s")
    
    # 3. Create visualizations
    create_eda_visualizations(df, fraud, legit)
    
    # 4. Correlation analysis
    analyze_correlations(df)
    
    return df


def create_eda_visualizations(df, fraud, legit):
    """Create comprehensive EDA visualizations."""
    print("\n3. Creating EDA Visualizations...")
    
    # Figure 1: Amount and Time distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Amount distribution by class
    axes[0, 0].hist(legit['Amount'], bins=50, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
    axes[0, 0].hist(fraud['Amount'], bins=50, alpha=0.6, label='Fraudulent', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Amount ($)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Transaction Amount Distribution by Class', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 500)  # Focus on lower amounts
    
    # Time distribution by class
    axes[0, 1].hist(legit['Time'], bins=50, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
    axes[0, 1].hist(fraud['Time'], bins=50, alpha=0.6, label='Fraudulent', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Time (seconds)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Transaction Time Distribution by Class', fontweight='bold')
    axes[0, 1].legend()
    
    # Box plots for Amount
    data_to_plot = [legit['Amount'], fraud['Amount']]
    axes[1, 0].boxplot(data_to_plot, labels=['Legitimate', 'Fraudulent'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_ylabel('Amount ($)', fontweight='bold')
    axes[1, 0].set_title('Amount Distribution - Box Plot', fontweight='bold')
    axes[1, 0].set_ylim(0, 300)
    
    # Fraud rate over time
    df_time = df.copy()
    df_time['Time_hours'] = df_time['Time'] / 3600
    time_bins = pd.cut(df_time['Time_hours'], bins=48)  # 48 bins for ~2 days
    fraud_rate = df_time.groupby(time_bins)['Class'].mean() * 100
    
    axes[1, 1].plot(range(len(fraud_rate)), fraud_rate.values, linewidth=2, color='red')
    axes[1, 1].set_xlabel('Time Period', fontweight='bold')
    axes[1, 1].set_ylabel('Fraud Rate (%)', fontweight='bold')
    axes[1, 1].set_title('Fraud Rate Over Time', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "eda_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Distribution plots saved to {output_path}")
    plt.close()
    
    # Figure 2: Feature correlation with target
    create_feature_importance_plot(df)


def create_feature_importance_plot(df):
    """Create visualization of feature correlations with target."""
    # Calculate correlation with target
    correlations = df.corr()['Class'].drop('Class').sort_values(ascending=False)
    
    # Plot top correlations
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ['red' if x > 0 else 'blue' for x in correlations]
    correlations.plot(kind='barh', ax=ax, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Correlation with Fraud Class', fontweight='bold')
    ax.set_title('Feature Correlations with Fraud Detection', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "feature_correlations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Feature correlation plot saved to {output_path}")
    plt.close()


def analyze_correlations(df):
    """Analyze and report feature correlations."""
    print("\n4. Correlation Analysis:")
    
    # Top positive correlations with fraud
    correlations = df.corr()['Class'].drop('Class').sort_values(ascending=False)
    
    print("\n  Top 5 Positive Correlations with Fraud:")
    for i, (feature, corr) in enumerate(correlations.head(5).items(), 1):
        print(f"    {i}. {feature}: {corr:.4f}")
    
    print("\n  Top 5 Negative Correlations with Fraud:")
    for i, (feature, corr) in enumerate(correlations.tail(5).items(), 1):
        print(f"    {i}. {feature}: {corr:.4f}")


def preprocess_data(df, test_size=0.3, val_size=0.5, random_state=42):
    """
    Preprocess data: scale features and create train/validation/test splits.
    
    Args:
        df: DataFrame with features and 'Class' column
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing scaled train/val/test splits and scaler
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"\n1. Data Split:")
    print(f"  Train: {X_train.shape[0]} samples ({y_train.sum()} fraudulent)")
    print(f"  Val:   {X_val.shape[0]} samples ({y_val.sum()} fraudulent)")
    print(f"  Test:  {X_test.shape[0]} samples ({y_test.sum()} fraudulent)")
    
    # Scale features using RobustScaler (better for outliers)
    print(f"\n2. Feature Scaling (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("  ✓ Features scaled")
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }


def apply_sampling_techniques(X_train, y_train):
    """
    Apply various sampling techniques to handle imbalanced data.
    
    Returns:
        Dictionary of sampled datasets with different techniques
    """
    print("\n" + "="*50)
    print("IMBALANCED DATA HANDLING")
    print("="*50)
    
    sampling_strategies = {}
    
    # 1. No sampling (baseline)
    print("\n1. Baseline (No Sampling):")
    print(f"  Class distribution: {np.bincount(y_train)}")
    sampling_strategies['baseline'] = (X_train, y_train)
    
    # 2. SMOTE (Synthetic Minority Over-sampling)
    print("\n2. SMOTE (Synthetic Minority Over-sampling):")
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    print(f"  Class distribution after SMOTE: {np.bincount(y_smote)}")
    sampling_strategies['smote'] = (X_smote, y_smote)
    
    # 3. Random Undersampling
    print("\n3. Random Undersampling:")
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    print(f"  Class distribution after undersampling: {np.bincount(y_rus)}")
    sampling_strategies['undersampling'] = (X_rus, y_rus)
    
    return sampling_strategies


def train_models(sampling_strategies, X_val, y_val):
    """
    Train multiple models with different sampling strategies.
    OPTIMIZED: Fast models only (GB is too slow on balanced data)
    
    Returns:
        Dictionary of trained models with their sampling strategy
    """
    print("\n" + "="*50)
    print("MODEL TRAINING (FAST OPTIMIZATION)")
    print("="*50)
    
    # Define models - skip Gradient Boosting on balanced SMOTE data
    models_to_train = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(
            n_estimators=50, max_depth=15, random_state=42, n_jobs=-1
        ),
    }
    
    trained_models = {}
    
    for sampling_name, (X_train_sampled, y_train_sampled) in sampling_strategies.items():
        print(f"\n{'='*50}")
        print(f"Training with: {sampling_name.upper()} ({X_train_sampled.shape[0]} samples)")
        print(f"{'='*50}")
        
        for model_name, model in models_to_train.items():
            print(f"\n  Training {model_name}...")
            
            # Train model
            model.fit(X_train_sampled, y_train_sampled)
            
            # Validate
            y_val_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred)
            recall = recall_score(y_val, y_val_pred)
            
            print(f"    Validation F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            
            # Store model
            key = f"{sampling_name}_{model_name.replace(' ', '_')}"
            trained_models[key] = {
                'model': model,
                'sampling': sampling_name,
                'model_name': model_name,
                'val_f1': f1,
                'val_precision': precision,
                'val_recall': recall
            }
    
    return trained_models


def evaluate_models(trained_models, X_test, y_test):
    """
    Comprehensive evaluation of all trained models on test set.
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION ON TEST SET")
    print("="*50)
    
    results = []
    
    for key, model_info in trained_models.items():
        model = model_info['model']
        sampling = model_info['sampling']
        model_name = model_info['model_name']
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        results.append({
            'sampling': sampling,
            'model': model_name,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'pr_auc': avg_precision,
            'roc_auc': roc_auc,
            'key': key
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    
    print("\n" + "="*50)
    print("TEST SET RESULTS (sorted by F1 score)")
    print("="*50)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_model_key = results_df.iloc[0]['key']
    best_model_info = trained_models[best_model_key]
    
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model_info['model_name']} with {best_model_info['sampling']}")
    print(f"{'='*50}")
    
    # Detailed report for best model
    y_pred_best = best_model_info['model'].predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Legitimate', 'Fraudulent']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Legit  Fraud")
    print(f"Actual Legit  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Fraud  {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Create visualizations
    create_evaluation_plots(trained_models, X_test, y_test, results_df)
    
    return results_df, best_model_key, best_model_info


def create_evaluation_plots(trained_models, X_test, y_test, results_df):
    """Create comprehensive evaluation visualizations."""
    print("\nCreating evaluation visualizations...")
    
    # Figure 1: Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1 Score comparison
    top_10 = results_df.head(10)
    axes[0, 0].barh(range(len(top_10)), top_10['f1'], color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_yticks(range(len(top_10)))
    axes[0, 0].set_yticklabels([f"{row['model']}\n({row['sampling']})" for _, row in top_10.iterrows()], fontsize=8)
    axes[0, 0].set_xlabel('F1 Score', fontweight='bold')
    axes[0, 0].set_title('Top 10 Models by F1 Score', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Precision vs Recall scatter
    axes[0, 1].scatter(results_df['recall'], results_df['precision'], 
                       c=results_df['f1'], cmap='viridis', s=100, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Recall', fontweight='bold')
    axes[0, 1].set_ylabel('Precision', fontweight='bold')
    axes[0, 1].set_title('Precision vs Recall (colored by F1)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('F1 Score', fontweight='bold')
    
    # ROC Curves for top 5 models
    top_5_keys = results_df.head(5)['key'].values
    for key in top_5_keys:
        model_info = trained_models[key]
        model = model_info['model']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        label = f"{model_info['model_name']} ({model_info['sampling']}) - AUC: {roc_auc:.3f}"
        axes[1, 0].plot(fpr, tpr, linewidth=2, label=label, alpha=0.7)
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1, 0].set_xlabel('False Positive Rate', fontweight='bold')
    axes[1, 0].set_ylabel('True Positive Rate', fontweight='bold')
    axes[1, 0].set_title('ROC Curves (Top 5 Models)', fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Curves for top 5 models
    for key in top_5_keys:
        model_info = trained_models[key]
        model = model_info['model']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        label = f"{model_info['model_name']} ({model_info['sampling']}) - AUC: {pr_auc:.3f}"
        axes[1, 1].plot(recall_curve, precision_curve, linewidth=2, label=label, alpha=0.7)
    
    axes[1, 1].set_xlabel('Recall', fontweight='bold')
    axes[1, 1].set_ylabel('Precision', fontweight='bold')
    axes[1, 1].set_title('Precision-Recall Curves (Top 5 Models)', fontweight='bold')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "model_evaluation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Evaluation plots saved to {output_path}")
    plt.close()


def create_individual_confusion_matrices(trained_models, X_test, y_test):
    """
    Create individual confusion matrix for each model for easy comparison.
    Each model gets its own subplot in a grid.
    """
    print("\n" + "="*50)
    print("CREATING INDIVIDUAL CONFUSION MATRICES")
    print("="*50)
    
    # Calculate grid dimensions (3 columns per row)
    n_models = len(trained_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    print(f"Creating {n_models} confusion matrices in {n_rows}x{n_cols} grid...")
    
    for idx, (model_key, model_info) in enumerate(trained_models.items()):
        model = model_info['model']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Create heatmap with proper annotations
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            ax=axes[idx],
            cbar=False,
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'],
            annot_kws={'size': 12, 'weight': 'bold'},
            linewidths=2,
            linecolor='black'
        )
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Set labels
        axes[idx].set_xlabel('Predicted Class', fontsize=11, weight='bold')
        axes[idx].set_ylabel('True Class', fontsize=11, weight='bold')
        
        # Set title with model name and key metrics
        title = f"{model_key}\n"
        title += f"F1: {f1:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}"
        axes[idx].set_title(title, fontsize=12, weight='bold', pad=15)
        
        # Add detailed metrics below the matrix
        metrics_text = (
            f"TP:{tp} | FP:{fp} | FN:{fn} | TN:{tn}\n"
            f"Accuracy: {accuracy:.3f} | Specificity: {specificity:.3f}"
        )
        axes[idx].text(
            0.5, -0.35, 
            metrics_text,
            ha='center', 
            transform=axes[idx].transAxes,
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        print(f"  [{idx+1}/{n_models}] {model_key}")
        print(f"      TP:{tp} FP:{fp} FN:{fn} TN:{tn} | F1:{f1:.4f} | Recall:{recall:.4f}")
    
    # Remove extra subplots
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Confusion Matrices - All Models Comparison', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    output_path = Path(__file__).parent / "confusion_matrices_all_models.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    print(f"✓ Generated {n_models} confusion matrices for comparison")
    plt.close()


def create_metrics_comparison_chart(trained_models, X_test, y_test):
    """
    Create bar charts comparing all models across key metrics.
    """
    print("\n" + "="*50)
    print("CREATING METRICS COMPARISON CHARTS")
    print("="*50)
    
    metrics_data = {
        'Model': [],
        'F1-Score': [],
        'Precision': [],
        'Recall': [],
        'Specificity': [],
        'Accuracy': []
    }
    
    for model_key, model_info in trained_models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics_data['Model'].append(model_key.replace('_', '\n'))
        metrics_data['F1-Score'].append(f1)
        metrics_data['Precision'].append(precision)
        metrics_data['Recall'].append(recall)
        metrics_data['Specificity'].append(specificity)
        metrics_data['Accuracy'].append(accuracy)
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 7))
    
    x = np.arange(len(df_metrics))
    width = 0.15
    
    metrics = ['F1-Score', 'Precision', 'Recall', 'Specificity', 'Accuracy']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df_metrics[metric], width, label=metric, color=colors[i], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Models', fontsize=12, weight='bold')
    ax.set_ylabel('Score', fontsize=12, weight='bold')
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, weight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_metrics['Model'], fontsize=9)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "metrics_comparison_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_model_comparison_table(trained_models, X_test, y_test):
    """
    Create detailed comparison table for all models and save as CSV.
    """
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("="*50)
    
    results = []
    
    for model_key, model_info in trained_models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Calculate additional metrics
        mcc = matthews_corrcoef(y_test, y_pred)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_pr = average_precision_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_roc = auc(fpr, tpr)
        else:
            auc_pr = np.nan
            auc_roc = np.nan
        
        results.append({
            'Model': model_key,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Accuracy': f"{accuracy:.4f}",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'Specificity': f"{specificity:.4f}",
            'F1-Score': f"{f1:.4f}",
            'MCC': f"{mcc:.4f}",
            'PR-AUC': f"{auc_pr:.4f}",
            'ROC-AUC': f"{auc_roc:.4f}"
        })
    
    results_df = pd.DataFrame(results)
    
    # Print in console
    print("\n" + results_df.to_string(index=False))
    
    # Save to CSV
    output_path = Path(__file__).parent / "model_comparison_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return results_df


def save_best_model(best_model_info, data_dict):
    """Save the best model and preprocessing pipeline."""
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "best_fraud_detector.pkl"
    joblib.dump(best_model_info['model'], model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save scaler
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(data_dict['scaler'], scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Save model metadata
    metadata = {
        'model_name': best_model_info['model_name'],
        'sampling_strategy': best_model_info['sampling'],
        'val_f1': best_model_info['val_f1'],
        'val_precision': best_model_info['val_precision'],
        'val_recall': best_model_info['val_recall'],
        'feature_names': data_dict['feature_names']
    }
    
    metadata_path = models_dir / "model_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved to {metadata_path}")
    
    return model_path, scaler_path


def predict_fraud(transaction_data, model_path=None, scaler_path=None):
    """
    Predict fraud for new transactions.
    
    Args:
        transaction_data: DataFrame with same features as training data
        model_path: Path to saved model (optional, uses default if None)
        scaler_path: Path to saved scaler (optional, uses default if None)
    
    Returns:
        DataFrame with predictions and probabilities
    """
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "best_fraud_detector.pkl"
    if scaler_path is None:
        scaler_path = Path(__file__).parent / "models" / "scaler.pkl"
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Preprocess
    X_scaled = scaler.transform(transaction_data)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Create results DataFrame
    results = transaction_data.copy()
    results['Predicted_Class'] = predictions
    results['Fraud_Probability'] = probabilities
    results['Is_Fraud'] = results['Predicted_Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    return results


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
        visualize_imbalance(class_distribution)
    
    # Step 6: Exploratory Data Analysis
    df = perform_eda(df)
    
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
    
    # Step 10: Evaluate models
    results_df, best_model_key, best_model_info = evaluate_models(
        trained_models,
        data_dict['X_test'],
        data_dict['y_test']
    )
    
    # Step 11: Create detailed model comparisons
    print("\n" + "="*50)
    print("COMPARING ALL MODELS")
    print("="*50)
    create_individual_confusion_matrices(trained_models, data_dict['X_test'], data_dict['y_test'])
    create_metrics_comparison_chart(trained_models, data_dict['X_test'], data_dict['y_test'])
    create_model_comparison_table(trained_models, data_dict['X_test'], data_dict['y_test'])
    
    # Step 12: Save best model
    model_path, scaler_path = save_best_model(best_model_info, data_dict)
    
    print("\n" + "="*50)
    print("✓ COMPLETE PIPELINE FINISHED!")
    print("="*50)
    print(f"\nBest Model: {best_model_info['model_name']}")
    print(f"Sampling Strategy: {best_model_info['sampling']}")
    print(f"Model saved to: {model_path}")
    print("\nGenerated Comparison Files:")
    print("  - confusion_matrices_all_models.png")
    print("  - metrics_comparison_chart.png")
    print("  - model_comparison_results.csv")
    print("\nYou can now use predict_fraud() to make predictions on new data.")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
