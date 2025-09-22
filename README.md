# Fraud Detection on Imbalanced Transactions

**Goal:** Reproducible baseline→best pipeline on the Kaggle Credit Card Fraud dataset with clean **metrics**, **plots**, and **ablations**.

## Dataset
- Kaggle: *Credit Card Fraud Detection* (284,807 rows; 492 fraud)
- Put the CSV at: `data/creditcard.csv` (original column names)

## Environment
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Quickstart
```bash
python src/train.py --data_path data/creditcard.csv --seed 42 --val_size 0.15 --test_size 0.15
python src/calibrate_and_eval.py --model_path outputs/best_model.joblib --val_cache outputs/val_cache.joblib --test_cache outputs/test_cache.joblib
python src/make_resume_snippet.py --report_path outputs/report.json
```

## Metrics (fill after run)
| Model | PR-AUC | Recall@1% FPR | ECE (pre→post isotonic) |
|------:|:------:|:-------------:|:-----------------------:|
| Logistic (class-weight) |  |  |  |
| XGBoost (scale_pos_weight) |  |  |  |
| XGBoost (tuned + isotonic) |  |  |  |

## Plots
- `outputs/pr_curve.png`, `outputs/confusion_matrix.png`
- `outputs/calibration_curve_before.png`, `outputs/calibration_curve_after.png`

## Reproducibility
- Stratified **train/val/test** split (70/15/15) with fixed seed
- Cached splits for exact reproducibility
- Report saved to `outputs/report.json`
