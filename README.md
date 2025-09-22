
- # Fraud Detection on Imbalanced Transactions

**Goal:** A reproducible **baseline → best** pipeline on the Kaggle **Credit Card Fraud Detection** dataset, with clean **metrics**, **plots**, and **ablations**.
Focus areas: **class imbalance**, **PR-AUC**, **Recall\@1% FPR**, and **probability calibration** (ECE, isotonic).

> **Note:** Results below are **provisional (expected/typical)** for this setup. Replace with your exact numbers after running the code.

---

## Dataset

* Source: Kaggle — *Credit Card Fraud Detection* (284,807 rows; 492 fraud).
* Place the CSV at: `data/creditcard.csv` (keep original column names including `Class` as the label).

> ⚠️ Do **not** commit the CSV to Git—keep `data/` in `.gitignore`.

---

## Environment & Setup

```bash
python -m venv .venv && source .venv/bin/activate         # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
printf "data/\noutputs/\n.venv/\n__pycache__/\n" > .gitignore
```

Key deps: `scikit-learn`, `xgboost`, `matplotlib`, `joblib`.

---

## Reproducibility

* Stratified **train/val/test** split = **70/15/15** with fixed `--seed`.
* Splits cached to `outputs/val_cache.joblib` and `outputs/test_cache.joblib`.
* Full test metrics saved to `outputs/report.json`.

---

## Quickstart

```bash
# 1) Train baselines and XGBoost; generates plots + report.json
python src/train.py --data_path data/creditcard.csv --seed 42 --val_size 0.15 --test_size 0.15

# 2) Calibrate best model (isotonic) using validation; re-evaluate on test
python src/calibrate_and_eval.py \
  --model_path outputs/best_model.joblib \
  --val_cache outputs/val_cache.joblib \
  --test_cache outputs/test_cache.joblib

# 3) Generate two crisp resume bullets from report.json
python src/make_resume_snippet.py --report_path outputs/report.json
```

---

## Results (Provisional/typical for this pipeline)

|                          Model |           PR-AUC          |     Recall@**1% FPR**    |         **ECE** (pre → post isotonic)         |
| -----------------------------: | :-----------------------: | :----------------------: | :-------------------------------------------: |
|      Logistic (class-weighted) | **0.88–0.90** → **0.890** | **0.68–0.74** → **0.71** | **0.07–0.09 → 0.04–0.05** → **0.078 → 0.045** |
|   XGBoost (scale\_pos\_weight) | **0.91–0.93** → **0.918** | **0.76–0.82** → **0.79** | **0.06–0.07 → 0.03–0.04** → **0.066 → 0.032** |
| **XGBoost (tuned + isotonic)** | **0.92–0.94** → **0.926** | **0.80–0.84** → **0.82** | **0.06–0.07 → 0.02–0.03** → **0.063 → 0.028** |

**Thresholding:** Confusion matrices are produced at the **test-set threshold corresponding to 1% FPR** (picked via ROC sweep on test scores).

**Files generated (after step 1 & 2):**

* Precision-Recall: `outputs/pr_curve.png`, `outputs/pr_curve_calibrated.png`
* Confusion matrices: `outputs/confusion_matrix.png`, `outputs/confusion_matrix_calibrated.png`
* Calibration curves: `outputs/calibration_curve_before.png`, `outputs/calibration_curve_after.png`
* Report JSON: `outputs/report.json`

---

## Ablations (Provisional examples)

| Ablation                     | Setting A → B                              |      Δ PR-AUC      |  Δ Recall\@1% FPR  | Notes                                        |
| ---------------------------- | ------------------------------------------ | :----------------: | :----------------: | -------------------------------------------- |
| **Class imbalance strategy** | LogReg `balanced` → XGB `scale_pos_weight` |  **≈ +0.02–0.03**  |  **≈ +0.05–0.09**  | Weighted trees handle rare positives better. |
| **Tree depth**               | max\_depth 3 → **4**                       | **≈ +0.004–0.008** |  **≈ +0.01–0.02**  | Mild gain; monitor overfit.                  |
| **Calibration**              | none → **isotonic**                        |   **0 to +0.005**  |       \~0.00       | ECE typically halves or better.              |
| **Early stopping**           | off → **on**                               | **≈ +0.002–0.004** | **≈ +0.005–0.015** | More stable generalization.                  |

---

## How to read `outputs/report.json`

Example structure you’ll get after running:

```json
{
  "best_model": "xgboost",
  "test": {
    "pr_auc": 0.9261,
    "recall_at_1pct_fpr": 0.8235,
    "ece_before_isotonic": 0.0632,
    "pr_auc_after_isotonic": 0.9304,
    "ece_after_isotonic": 0.0281,
    "threshold_at_1pct_fpr": 0.9137,
    "counts": {"TP": 320, "FP": 412, "TN": 40556, "FN": 68}
  }
}
```

---

## Resume bullets (auto-generated)

After you run the pipeline, execute:

```bash
python src/make_resume_snippet.py --report_path outputs/report.json
```

It prints two bullets like:

```
- Built fraud detector on imbalanced data; **XGBoost** improved **PR-AUC 0.926** (post-iso: 0.930); at **1% FPR** recall was **0.824**.
- **Isotonic calibration** reduced **ECE 0.063 → 0.028**; reproducible stratified splits; PR/confusion/calibration plots in repo.
```

Paste them into your README and resume (keep the numbers **bold**).

---

## Project structure

```
fraud-detection-imbalanced/
├─ README.md
├─ requirements.txt
├─ data/                      # put Kaggle CSV here (git-ignored)
├─ outputs/                   # metrics + plots (git-ignored)
└─ src/
   ├─ train.py
   ├─ calibrate_and_eval.py
   └─ make_resume_snippet.py
```

---

## License

MIT — educational/interview use.

---

# IMDb Sentiment — Baseline → DistilBERT with Ablations

**Goal:** Demonstrate scientific experimentation on the IMDb sentiment dataset: **baselines → fine-tuning**, clear **metrics**, and compact **ablations** suitable for a resume.

> **Note:** Results below are **provisional (expected/typical)** for this setup. Replace with your exact numbers after running.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

```bash
# Baseline TF-IDF + Logistic
python src/baseline_logreg.py --seed 42

# DistilBERT fine-tune (CPU or GPU)
python src/finetune_distilbert.py --seed 42 --epochs 2 --lr 5e-5 --max_len 256

# Generate README/resume snippet from report
python src/make_resume_snippet.py --report_path outputs/report.json
```

---

## Results (Provisional)

|                 Model |         F1 (test)         |       PR-AUC (test)       |
| --------------------: | :-----------------------: | :-----------------------: |
|     TF‑IDF + Logistic | **0.89–0.91** → **0.900** | **0.94–0.96** → **0.950** |
| **DistilBERT (best)** | **0.93–0.95** → **0.944** | **0.96–0.98** → **0.972** |

**Files generated:**

* Baseline: `outputs/pr_curve_baseline.png`, `outputs/confusion_baseline.png`
* DistilBERT: `outputs/pr_curve_distilbert.png`, `outputs/confusion_distilbert.png`
* Report JSON: `outputs/report.json`

---

## Ablations (Provisional examples)

| Ablation          | Setting A → B   |        Δ F1        |      Δ PR-AUC     | Notes                                                     |
| ----------------- | --------------- | :----------------: | :---------------: | --------------------------------------------------------- |
| **Epochs**        | 1 → **2**       |  **≈ +0.01–0.02**  | **≈ +0.005–0.01** | Diminishing returns beyond 3 without more regularization. |
| **Learning rate** | 3e-5 → **5e-5** |  **≈ +0.005–0.01** |       \~flat      | 5e-5 often best; watch for instability on CPU.            |
| **max\_seq\_len** | 128 → **256**   | **≈ +0.005–0.015** | **≈ +0.005–0.01** | Longer context helps sarcasm/negation.                    |

---

## Resume bullets (auto-generated)

After a run:

```bash
python src/make_resume_snippet.py --report_path outputs/report.json
```

Example output (replace with yours):

```
- Reproduced IMDb baselines: **TF-IDF + Logistic** then **DistilBERT**; achieved **F1 0.944**, **PR-AUC 0.972** on test.
- Ablated epochs/lr/max_len; tuned token length improved **F1 by +0.044** over baseline; plots + confusion in repo.
```

---

## Project structure

```
imdb-sentiment-distilbert/
├─ README.md
├─ requirements.txt
├─ outputs/                   # metrics + plots
└─ src/
   ├─ baseline_logreg.py
   ├─ finetune_distilbert.py
   └─ make_resume_snippet.py
```

---

## License

MIT — educational/interview use.

