# Fraud Detection with Imbalanced Data

A complete machine learning pipeline for detecting credit card fraud using various sampling techniques and classification models.

## 📁 Project Structure

```
FraudDetection/
├── data/                           # Data directory
│   ├── creditcard.csv             # Full dataset (284,807 transactions)
│   └── test_data.csv              # Smaller test dataset (10,000 transactions)
│
├── models/                         # Saved trained models
│   ├── best_fraud_detector.pkl    # Best trained model
│   ├── scaler.pkl                 # Feature scaler
│   └── model_metadata.pkl         # Model metadata
│
├── output/                         # Generated visualizations and results
│   ├── imbalance_visualization.png
│   ├── eda_distributions.png
│   ├── feature_correlations.png
│   ├── confusion_matrices_all_models.png
│   ├── metrics_comparison_chart.png
│   ├── model_comparison_results.csv
│   └── [test visualizations]
│
├── fraud_detection_main.py         # Main orchestrator (RUN THIS)
├── train_model.py                  # Model training module
├── fraud_detection_utils.py        # EDA and data utilities
├── test_model.py                   # Model verification script
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Train Models
```bash
python fraud_detection_main.py
```
This will:
- Load and analyze the data
- Perform EDA
- Apply sampling techniques (Baseline, SMOTE, Undersampling)
- Train 6 models (3 sampling × 2 models)
- Compare and save the best model
- Generate visualizations in `output/` folder

**Time: ~8-12 minutes**

### 2. Test Models
```bash
python test_model.py
```
This will:
- Load the trained best model
- Make predictions on test data
- Generate ROC curves, Confusion matrix, and other visualizations
- Display performance metrics

**Time: ~1-2 minutes**

## 📊 Models Trained

| Sampling Strategy | Models |
|------------------|--------|
| **Baseline** | Logistic Regression, Random Forest |
| **SMOTE** | Logistic Regression, Random Forest |
| **Undersampling** | Logistic Regression, Random Forest |

### Sampling Techniques

1. **Baseline** - No resampling (reference point)
2. **SMOTE** - Synthetic Minority Over-sampling (generates synthetic fraud cases)
3. **Random Undersampling** - Removes majority class samples

## 📈 Output Files

### Visualizations (in `output/` folder)

| File | Description |
|------|-------------|
| `imbalance_visualization.png` | Class distribution (count and percentage) |
| `eda_distributions.png` | Feature distributions by class |
| `feature_correlations.png` | Feature correlations with fraud |
| `confusion_matrices_all_models.png` | All 6 models' confusion matrices |
| `metrics_comparison_chart.png` | F1, Precision, Recall comparison |
| `test_confusion_matrix.png` | Best model's confusion matrix on test data |
| `test_roc_curve.png` | ROC curve with AUC score |
| `test_pr_curve.png` | Precision-Recall curve |
| `test_probability_distribution.png` | Predicted probability distributions |

### Results Files (in `output/` folder)

| File | Description |
|------|-------------|
| `model_comparison_results.csv` | Detailed metrics for all 6 models |

## 🔍 Key Metrics Explained

### Confusion Matrix Components
- **TP (True Positives)**: Correctly identified fraud
- **FP (False Positives)**: Legitimate flagged as fraud (false alarm)
- **TN (True Negatives)**: Correctly identified legitimate
- **FN (False Negatives)**: Fraud missed (dangerous!)

### Performance Metrics
- **F1-Score**: Balance between precision and recall
- **Precision**: Of predicted fraud, how many are actually fraud
- **Recall (Sensitivity)**: Of actual fraud, how many are detected
- **Specificity**: Of legitimate, how many are correctly identified
- **ROC-AUC**: Overall discrimination ability
- **PR-AUC**: Precision-Recall trade-off

## 📝 Data Information

### Full Dataset (creditcard.csv)
- **Transactions**: 284,807
- **Fraud Rate**: 0.17% (492 fraudulent)
- **Features**: 30 (V1-V28, Time, Amount)
- **Class Imbalance Ratio**: 577:1

### Test Dataset (test_data.csv)
- **Transactions**: 10,000 (random sample)
- **Fraud Rate**: ~0.16% (balanced sample)
- **Used for**: Quick verification and testing

## 🛠 Module Functions

### `fraud_detection_main.py`
- **Main Entry Point**: Orchestrates entire pipeline

### `train_model.py`
Key functions:
- `preprocess_data()` - Scaling and train/val/test split
- `apply_sampling_techniques()` - SMOTE, undersampling
- `train_models()` - Train multiple models
- `evaluate_models()` - Test and compare models
- `save_best_model()` - Persist best model

### `fraud_detection_utils.py`
Key functions:
- `load_data()` - Load creditcard.csv
- `sanity_check()` - Data quality checks
- `analyze_imbalance()` - Class distribution analysis
- `perform_eda()` - Exploratory data analysis
- `visualize_imbalance()` - Create visualizations

### `test_model.py`
Key functions:
- `load_test_data()` - Load test_data.csv
- `predict_fraud()` - Make predictions on new data
- `main()` - Run verification tests

## 🎯 Best Model Performance

After training, the best model is automatically selected based on **F1-Score** and saved to:
- **Model**: `models/best_fraud_detector.pkl`
- **Scaler**: `models/scaler.pkl`
- **Metadata**: `models/model_metadata.pkl`

## 💡 Tips

1. **For quick experiments**: Use `test_data.csv` (10K rows, ~1 min)
2. **For full training**: Use `creditcard.csv` (285K rows, ~12 min)
3. **For model comparison**: Check `output/metrics_comparison_chart.png`
4. **For detailed results**: Open `output/model_comparison_results.csv`

## ⚠️ Important Notes

1. **Class Imbalance**: With 577:1 ratio, accuracy is NOT a good metric
2. **Metrics Used**: F1-Score, Precision, Recall, and ROC-AUC
3. **Trade-off**: Higher recall (catch fraud) means more false alarms
4. **Real-world**: Adjust threshold based on business cost of fraud vs false alarms

## 🔧 Customization

To modify:
- **Number of models**: Edit `train_model.py` `train_models()`
- **Sampling strategies**: Edit `train_model.py` `apply_sampling_techniques()`
- **Model parameters**: Edit `train_model.py` (n_estimators, max_depth, etc.)
- **Output folder**: Modify `fraud_detection_main.py` `OUTPUT_DIR`

## 📚 References

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Imbalanced Learning: [imbalanced-learn documentation](https://imbalanced-learn.org/)
- Scikit-learn: [ML library documentation](https://scikit-learn.org/)

---

**Created**: January 20, 2026  
**Last Updated**: January 20, 2026
