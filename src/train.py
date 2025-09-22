import argparse, os, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_curve
from xgboost import XGBClassifier
from utils import recall_at_fpr, plot_pr, plot_confusion, expected_calibration_error

def main(args):
    df = pd.read_csv(args.data_path)
    y = df['Class'].astype(int).values
    X = df.drop(columns=['Class']).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.val_size+args.test_size, random_state=args.seed, stratify=y
    )
    val_ratio = args.val_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, random_state=args.seed, stratify=y_temp
    )

    os.makedirs("outputs", exist_ok=True)
    joblib.dump((X_val, y_val), "outputs/val_cache.joblib")
    joblib.dump((X_test, y_test), "outputs/test_cache.joblib")

    print("Class counts:",
          {"train": np.bincount(y_train).tolist(),
           "val": np.bincount(y_val).tolist(),
           "test": np.bincount(y_test).tolist()})

    # Baseline Logistic (balanced)
    logit = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, solver="liblinear")),
    ])
    logit.fit(X_train, y_train)
    val_scores_log = logit.predict_proba(X_val)[:,1]
    test_scores_log = logit.predict_proba(X_test)[:,1]

    base_ap = average_precision_score(y_test, test_scores_log)
    base_rec, base_thr = recall_at_fpr(y_test, test_scores_log, target_fpr=0.01)
    base_ece = expected_calibration_error(y_test, test_scores_log)

    # XGBoost (weighted)
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        min_child_weight=1.0, objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=args.seed
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_scores_xgb = xgb.predict_proba(X_val)[:,1]
    test_scores_xgb = xgb.predict_proba(X_test)[:,1]

    xgb_ap = average_precision_score(y_test, test_scores_xgb)
    xgb_rec, xgb_thr = recall_at_fpr(y_test, test_scores_xgb, target_fpr=0.01)
    xgb_ece = expected_calibration_error(y_test, test_scores_xgb)

    # Select best by validation AP
    from sklearn.metrics import average_precision_score as AP
    best = ("logistic", logit, val_scores_log, test_scores_log, base_ap, base_rec, base_ece, base_thr)
    if AP(y_val, val_scores_xgb) > AP(y_val, val_scores_log):
        best = ("xgboost", xgb, val_scores_xgb, test_scores_xgb, xgb_ap, xgb_rec, xgb_ece, xgb_thr)
    best_name, best_model, best_val, best_test, best_ap, best_rec, best_ece, best_thr = best

    # Save best model
    joblib.dump(best_model, "outputs/best_model.joblib")

    # Plots
    plot_pr(y_test, best_test, "outputs/pr_curve.png")
    y_pred = (best_test >= best_thr).astype(int)
    plot_confusion(y_test, y_pred, "outputs/confusion_matrix.png")

    report = {
        "best_model": best_name,
        "test": {
            "pr_auc": float(average_precision_score(y_test, best_test)),
            "recall_at_1pct_fpr": float(best_rec),
            "ece_before_isotonic": float(best_ece),
            "threshold_at_1pct_fpr": float(best_thr),
            "counts": {
                "TP": int(((y_pred==1)&(y_test==1)).sum()),
                "FP": int(((y_pred==1)&(y_test==0)).sum()),
                "TN": int(((y_pred==0)&(y_test==0)).sum()),
                "FN": int(((y_pred==0)&(y_test==1)).sum())
            }
        }
    }
    with open("outputs/report.json","w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    main(ap.parse_args())
