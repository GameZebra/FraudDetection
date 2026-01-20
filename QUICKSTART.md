# 🎯 QUICK START GUIDE

## New Project Structure

Your fraud detection project is now organized into clean, modular components:

```
FraudDetection/
├── 📊 fraud_detection_main.py      ← RUN THIS TO TRAIN
├── 🔬 train_model.py               (Training logic)
├── 🛠  fraud_detection_utils.py    (Data & EDA tools)
├── ✅ test_model.py                (Run this to verify)
│
├── 📁 data/
│   ├── creditcard.csv              (Full: 285K transactions)
│   └── test_data.csv               (Sample: 10K transactions)
│
├── 🤖 models/
│   ├── best_fraud_detector.pkl
│   ├── scaler.pkl
│   └── model_metadata.pkl
│
└── 📈 output/
    ├── confusion_matrices_all_models.png
    ├── metrics_comparison_chart.png
    ├── model_comparison_results.csv
    └── [All visualizations]
```

## ⚡ Quick Commands

### 1️⃣ Train Models (First Time - 12 minutes)
```bash
python fraud_detection_main.py
```
**What it does:**
- Loads creditcard.csv
- Analyzes data imbalance
- Trains 6 different models
- Compares them all
- Saves best model
- Generates comparison charts

**Output:** `output/confusion_matrices_all_models.png` + `output/metrics_comparison_chart.png`

---

### 2️⃣ Test Model on New Data (1 minute)
```bash
python test_model.py
```
**What it does:**
- Loads trained model
- Uses test_data.csv (10K transactions)
- Makes predictions
- Shows performance metrics
- Generates ROC/PR curves

**Output:** `output/test_*.png` files

---

### 3️⃣ Train + Test (13 minutes total)
```bash
python fraud_detection_main.py && python test_model.py
```

---

## 📊 Understanding the Output

### confusion_matrices_all_models.png
Shows **6 confusion matrices** (one per model):
- **Top-Left**: True Negatives (correct legitimate)
- **Top-Right**: False Positives (false alarms)
- **Bottom-Left**: False Negatives (missed fraud)
- **Bottom-Right**: True Positives (caught fraud)

**Best model**: Highest TP, lowest FN

### metrics_comparison_chart.png
Bar chart comparing all 6 models:
- **F1-Score** (overall performance)
- **Precision** (accuracy of fraud predictions)
- **Recall** (how many frauds caught)
- **Specificity** (how many legit accepted)
- **Accuracy** (overall correctness)

**Best model**: Highest F1 score

### model_comparison_results.csv
Detailed table with all metrics for each model. Open in Excel to analyze.

---

## 🔍 Module Guide

### fraud_detection_main.py
**Main orchestrator**
- Calls all training functions in order
- Generates comparison visualizations
- Saves best model

### train_model.py
**Training functions**
- `preprocess_data()` - Normalize and split data
- `apply_sampling_techniques()` - Create balanced datasets
- `train_models()` - Train 6 models
- `evaluate_models()` - Test and compare
- `save_best_model()` - Save winner

### fraud_detection_utils.py
**Data utilities**
- `load_data()` - Load creditcard.csv
- `perform_eda()` - Analyze data
- `visualize_imbalance()` - Show class distribution
- `analyze_correlations()` - Feature importance

### test_model.py
**Testing & verification**
- `predict_fraud()` - Make predictions
- Tests on 10K transaction sample
- Generates ROC/PR curves
- Shows performance metrics

---

## 💡 Key Concepts

### Why 6 Models?
- **3 Sampling Techniques**: Baseline, SMOTE, Undersampling
- **2 Models Each**: Logistic Regression, Random Forest
- **Comparison**: Find best combination

### Sampling Techniques
| Technique | What It Does | Pros | Cons |
|-----------|------------|------|------|
| **Baseline** | No changes | Simple reference | Biased to majority |
| **SMOTE** | Creates fake fraud | Balanced data | Synthetic data |
| **Undersampling** | Removes legit | Simple | Loses info |

### Which Metric Matters?
- **Imbalanced data**: F1-Score (not accuracy!)
- **Business priority**: Recall (catch fraud) vs Precision (avoid false alarms)
- **Overall**: ROC-AUC (discrimination ability)

---

## 🚀 Next Steps

1. **First run**: `python fraud_detection_main.py`
   - Takes ~12 minutes
   - Trains & compares models
   - Creates charts

2. **Review results**: Open `output/metrics_comparison_chart.png`
   - See which model is best
   - Check F1 scores

3. **Verify predictions**: `python test_model.py`
   - Test on new data
   - Check confusion matrix
   - View ROC curve

4. **Customize** (if needed):
   - Edit `train_model.py` to change models
   - Edit `fraud_detection_utils.py` to add EDA
   - Change output folder path

---

## ❓ FAQ

**Q: Can I use just the small test data?**
A: Yes! Edit `fraud_detection_main.py` line with `load_data()` to load `test_data.csv` instead. 2-minute training!

**Q: How do I make predictions on my own data?**
A: Use `test_model.py` as template. Update `load_test_data()` to load your CSV.

**Q: Which model should I use?**
A: Use the best one automatically saved in `models/best_fraud_detector.pkl`

**Q: Why does test show different results than training?**
A: Different data! Training uses full 284K, testing uses 10K sample.

**Q: Can I retrain the model?**
A: Yes, just run `python fraud_detection_main.py` again. It overwrites the old model.

---

**Enjoy your fraud detection system! 🎉**
