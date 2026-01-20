"""
Test script to verify the trained fraud detection model
on test data with visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from train_model import preprocess_data
from fraud_detection_utils import load_data

# Configuration
DATA_DIR = Path(__file__).parent / "data"
TEST_DATA_FILE = DATA_DIR / "test_data.csv"
MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_test_data():
    """Load test data from test_data.csv or create it"""
    if TEST_DATA_FILE.exists():
        try:
            df = pd.read_csv(TEST_DATA_FILE)
            print(f"✓ Test data loaded: {len(df)} transactions")
            return df
        except Exception as e:
            print(f"✗ Error loading test data: {e}")
    
    print(f"Creating test data from creditcard.csv...")
    df = load_data()
    if df is not None:
        test_df = df.sample(n=10000, random_state=42)
        test_df.to_csv(TEST_DATA_FILE, index=False)
        print(f"✓ Test data created and saved: {len(test_df)} transactions")
        return test_df
    return None


def predict_fraud(transaction_data, model_path=None, scaler_path=None):
    """Predict fraud for new transactions."""
    if model_path is None:
        model_path = MODELS_DIR / "best_fraud_detector.pkl"
    if scaler_path is None:
        scaler_path = MODELS_DIR / "scaler.pkl"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    X_scaled = scaler.transform(transaction_data)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    results = transaction_data.copy()
    results['Predicted_Class'] = predictions
    results['Fraud_Probability'] = probabilities
    results['Is_Fraud'] = results['Predicted_Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    return results


def main():
    """Main execution function."""
    print("="*70)
    print("FRAUD DETECTION MODEL VERIFICATION")
    print("="*70)
    
    print("\n1. Loading test data...")
    df = load_test_data()
    if df is None:
        print("✗ Failed to load test data")
        return
    
    # Load metadata
    print("\n2. Best Model Information:")
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    metadata = joblib.load(metadata_path)
    
    print(f"   Model: {metadata['model_name']}")
    print(f"   Sampling Strategy: {metadata['sampling_strategy']}")
    print(f"   Validation F1 Score: {metadata['val_f1']:.4f}")
    print(f"   Validation Precision: {metadata['val_precision']:.4f}")
    print(f"   Validation Recall: {metadata['val_recall']:.4f}")
    
    # Prepare data
    features = metadata['feature_names']
    X = df[features].copy()
    
    fraud_transactions = df[df['Class'] == 1].copy()
    legit_transactions = df[df['Class'] == 0].copy()
    
    print(f"\n3. Test Data Summary:")
    print(f"   Total transactions: {len(df)}")
    print(f"   Fraudulent: {len(fraud_transactions)}")
    print(f"   Legitimate: {len(legit_transactions)}")
    
    # Test 1: Fraudulent transactions
    print(f"\n" + "="*70)
    print("TEST 1: PREDICTING ON FRAUDULENT TRANSACTIONS")
    print("="*70)
    
    if len(fraud_transactions) > 0:
        fraud_sample = fraud_transactions[features].iloc[:10].copy()
        predictions = predict_fraud(fraud_sample)
        
        print(f"\nPredictions on {len(predictions)} known fraudulent transactions:")
        print(predictions[['Amount', 'Predicted_Class', 'Fraud_Probability', 'Is_Fraud']].to_string())
        
        detected = (predictions['Predicted_Class'] == 1).sum()
        detection_rate = (detected / len(predictions)) * 100
        print(f"\n✓ Detection Rate: {detected}/{len(predictions)} ({detection_rate:.1f}%)")
    
    # Test 2: Legitimate transactions
    print(f"\n" + "="*70)
    print("TEST 2: PREDICTING ON LEGITIMATE TRANSACTIONS")
    print("="*70)
    
    if len(legit_transactions) > 0:
        legit_sample = legit_transactions[features].iloc[:10].copy()
        predictions = predict_fraud(legit_sample)
        
        print(f"\nPredictions on {len(predictions)} known legitimate transactions:")
        print(predictions[['Amount', 'Predicted_Class', 'Fraud_Probability', 'Is_Fraud']].to_string())
        
        false_positives = (predictions['Predicted_Class'] == 1).sum()
        false_positive_rate = (false_positives / len(predictions)) * 100
        print(f"\n✓ False Positive Rate: {false_positives}/{len(predictions)} ({false_positive_rate:.1f}%)")
    
    # Test 3: Probability distribution
    print(f"\n" + "="*70)
    print("TEST 3: ANALYZING FRAUD PROBABILITY DISTRIBUTION")
    print("="*70)
    
    print("\nPredicting on all test transactions...")
    all_predictions = predict_fraud(X)
    
    actual_fraud = all_predictions[df['Class'] == 1]
    actual_legit = all_predictions[df['Class'] == 0]
    
    print(f"\nFraudulent Transactions Fraud Probability Stats:")
    print(f"  Mean: {actual_fraud['Fraud_Probability'].mean():.4f}")
    print(f"  Median: {actual_fraud['Fraud_Probability'].median():.4f}")
    print(f"  Min: {actual_fraud['Fraud_Probability'].min():.4f}")
    print(f"  Max: {actual_fraud['Fraud_Probability'].max():.4f}")
    
    print(f"\nLegitimate Transactions Fraud Probability Stats:")
    print(f"  Mean: {actual_legit['Fraud_Probability'].mean():.4f}")
    print(f"  Median: {actual_legit['Fraud_Probability'].median():.4f}")
    print(f"  Min: {actual_legit['Fraud_Probability'].min():.4f}")
    print(f"  Max: {actual_legit['Fraud_Probability'].max():.4f}")
    
    # Test 4: Confusion matrix and metrics
    print(f"\n" + "="*70)
    print("TEST 4: FULL MODEL PERFORMANCE METRICS & VISUALIZATIONS")
    print("="*70)
    
    cm = confusion_matrix(df['Class'], all_predictions['Predicted_Class'])
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Legit  Fraud")
    print(f"Actual Legit  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Fraud  {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  True Positives (Fraud Detected): {tp}")
    print(f"  False Positives (False Alarms): {fp}")
    print(f"  True Negatives (Legit Accepted): {tn}")
    print(f"  False Negatives (Fraud Missed): {fn}")
    print(f"\n  Sensitivity (Recall): {sensitivity:.4f} - {tp}/{tp+fn} frauds caught")
    print(f"  Specificity: {specificity:.4f} - {tn}/{tn+fp} legit correctly identified")
    print(f"  Precision: {precision:.4f} - {tp}/{tp+fp} predictions are correct")
    print(f"  Negative Predictive Value: {npv:.4f} - {tn}/{tn+fn} negatives correct")
    
    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    def create_confusion_matrix_plot(y_true, y_pred, title="Confusion Matrix"):
        """Create confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legitimate', 'Fraudulent'],
                    yticklabels=['Legitimate', 'Fraudulent'],
                    annot_kws={'size': 16, 'weight': 'bold'},
                    ax=ax, linewidths=2, linecolor='black')
        
        ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        tn, fp, fn, tp = cm.ravel()
        
        fig.text(0.15, 0.85, f'True Negatives (TN)\n{tn}', 
                 fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        fig.text(0.85, 0.85, f'False Positives (FP)\n{fp}', 
                 fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        fig.text(0.15, 0.15, f'False Negatives (FN)\n{fn}', 
                 fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        fig.text(0.85, 0.15, f'True Positives (TP)\n{tp}', 
                 fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        return fig, cm
    
    def create_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
        """Create ROC curve visualization."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_pr_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
        """Create Precision-Recall curve visualization."""
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall_curve, precision_curve, color='green', lw=3, label=f'PR curve (AUC = {pr_auc:.4f})')
        ax.axhline(y=y_true.sum()/len(y_true), color='red', linestyle='--', lw=2, label='Baseline')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_probability_distribution(y_true, y_pred_proba, title="Fraud Probability Distribution"):
        """Create histogram of predicted probabilities."""
        fraud_proba = y_pred_proba[y_true == 1]
        legit_proba = y_pred_proba[y_true == 0]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(legit_proba, bins=50, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
        ax.hist(fraud_proba, bins=50, alpha=0.6, label='Fraudulent', color='red', edgecolor='black')
        
        ax.set_xlabel('Predicted Fraud Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    # Generate all visualizations
    print("\n  Creating confusion matrix visualization...")
    fig1, _ = create_confusion_matrix_plot(df['Class'], all_predictions['Predicted_Class'], 
                                            "Confusion Matrix - Model Performance")
    fig1.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("     ✓ Saved to test_confusion_matrix.png")
    
    print("  Creating ROC curve...")
    fig2 = create_roc_curve(df['Class'], all_predictions['Fraud_Probability'], 
                            "ROC Curve - Model Discrimination Ability")
    fig2.savefig(OUTPUT_DIR / "test_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("     ✓ Saved to test_roc_curve.png")
    
    print("  Creating Precision-Recall curve...")
    fig3 = create_pr_curve(df['Class'], all_predictions['Fraud_Probability'], 
                           "Precision-Recall Curve - Trade-off Analysis")
    fig3.savefig(OUTPUT_DIR / "test_pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("     ✓ Saved to test_pr_curve.png")
    
    print("  Creating probability distribution plot...")
    fig4 = create_probability_distribution(df['Class'], all_predictions['Fraud_Probability'],
                                           "Distribution of Predicted Fraud Probabilities")
    fig4.savefig(OUTPUT_DIR / "test_probability_distribution.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("     ✓ Saved to test_probability_distribution.png")
    
    print(f"\n" + "="*70)
    print("✓ MODEL VERIFICATION COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations in output/ folder:")
    print("  - test_confusion_matrix.png")
    print("  - test_roc_curve.png")
    print("  - test_pr_curve.png")
    print("  - test_probability_distribution.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
