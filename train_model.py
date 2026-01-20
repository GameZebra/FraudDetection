"""
Model Training Module
Handles all model training and comparison logic
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    matthews_corrcoef, average_precision_score, roc_curve, auc,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df, test_size=0.3, val_size=0.5, random_state=42):
    """Preprocess data: scale features and create train/validation/test splits."""
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
    
    # Scale features
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
    """Apply sampling techniques to handle imbalanced data."""
    print("\n" + "="*50)
    print("IMBALANCED DATA HANDLING")
    print("="*50)
    
    sampling_strategies = {}
    
    # 1. Baseline (no sampling)
    print("\n1. Baseline (No Sampling):")
    print(f"  Class distribution: {np.bincount(y_train)}")
    sampling_strategies['baseline'] = (X_train, y_train)
    
    # 2. SMOTE
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
    """Train multiple models with different sampling strategies."""
    print("\n" + "="*50)
    print("MODEL TRAINING (FAST OPTIMIZATION)")
    print("="*50)
    
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
            
            model.fit(X_train_sampled, y_train_sampled)
            
            y_val_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred)
            recall = recall_score(y_val, y_val_pred)
            
            print(f"    Validation F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            
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


def evaluate_models(trained_models, X_test, y_test, output_dir='output'):
    """Comprehensive evaluation of trained models."""
    print("\n" + "="*50)
    print("MODEL EVALUATION ON TEST SET")
    print("="*50)
    
    results = []
    
    for key, model_info in trained_models.items():
        model = model_info['model']
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        results.append({
            'sampling': model_info['sampling'],
            'model': model_info['model_name'],
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'pr_auc': avg_precision,
            'roc_auc': roc_auc,
            'key': key
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    
    print("\n" + "="*50)
    print("TEST SET RESULTS (sorted by F1 score)")
    print("="*50)
    print(results_df.to_string(index=False))
    
    best_model_key = results_df.iloc[0]['key']
    best_model_info = trained_models[best_model_key]
    
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model_info['model_name']} with {best_model_info['sampling']}")
    print(f"{'='*50}")
    
    create_confusion_matrices(trained_models, X_test, y_test, output_dir)
    create_metrics_comparison_chart(trained_models, X_test, y_test, output_dir)
    create_model_comparison_table(trained_models, X_test, y_test, output_dir)
    
    return results_df, best_model_key, best_model_info


def create_confusion_matrices(trained_models, X_test, y_test, output_dir='output'):
    """Create individual confusion matrix for each model."""
    print("\n" + "="*50)
    print("CREATING INDIVIDUAL CONFUSION MATRICES")
    print("="*50)
    
    n_models = len(trained_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (model_key, model_info) in enumerate(trained_models.items()):
        model = model_info['model']
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
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
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        axes[idx].set_xlabel('Predicted Class', fontsize=11, weight='bold')
        axes[idx].set_ylabel('True Class', fontsize=11, weight='bold')
        
        title = f"{model_key}\n"
        title += f"F1: {f1:.3f} | Recall: {recall:.3f} | Precision: {precision:.3f}"
        axes[idx].set_title(title, fontsize=12, weight='bold', pad=15)
        
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
        print(f"      TP:{tp} FP:{fp} FN:{fn} TN:{tn} | F1:{f1:.4f}")
    
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Confusion Matrices - All Models Comparison', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    output_path = Path(output_dir) / "confusion_matrices_all_models.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def create_metrics_comparison_chart(trained_models, X_test, y_test, output_dir='output'):
    """Create bar charts comparing all models."""
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
    output_path = Path(output_dir) / "metrics_comparison_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_model_comparison_table(trained_models, X_test, y_test, output_dir='output'):
    """Create detailed comparison table."""
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
    
    print("\n" + results_df.to_string(index=False))
    
    output_path = Path(output_dir) / "model_comparison_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return results_df


def save_best_model(best_model_info, data_dict, models_dir='models'):
    """Save the best model and preprocessing pipeline."""
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "best_fraud_detector.pkl"
    joblib.dump(best_model_info['model'], model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save scaler
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(data_dict['scaler'], scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Save metadata
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
