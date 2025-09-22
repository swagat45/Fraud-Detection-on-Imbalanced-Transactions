import argparse, json, joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, roc_curve
import matplotlib.pyplot as plt
from utils import plot_pr, plot_confusion, expected_calibration_error

def main(args):
    model = joblib.load(args.model_path)
    X_val, y_val = joblib.load(args.val_cache)
    X_test, y_test = joblib.load(args.test_cache)

    calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calib.fit(X_val, y_val)

    test_scores_before = model.predict_proba(X_test)[:,1]
    test_scores_after = calib.predict_proba(X_test)[:,1]

    ece_before = expected_calibration_error(y_test, test_scores_before)
    ece_after = expected_calibration_error(y_test, test_scores_after)
    ap_before = average_precision_score(y_test, test_scores_before)
    ap_after = average_precision_score(y_test, test_scores_after)

    # PR & Confusion
    plot_pr(y_test, test_scores_after, "outputs/pr_curve_calibrated.png")
    fpr, tpr, thr = roc_curve(y_test, test_scores_after)
    idx = max(0, (np.searchsorted(fpr, 0.01, side="right") - 1))
    y_pred = (test_scores_after >= thr[idx]).astype(int)
    plot_confusion(y_test, y_pred, "outputs/confusion_matrix_calibrated.png")

    # Calibration curves
    for scores, name in [(test_scores_before, "before"), (test_scores_after, "after")]:
        prob_true, prob_pred = calibration_curve(y_test, scores, n_bins=10, strategy="uniform")
        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o", label="Empirical")
        plt.plot([0,1],[0,1],"--", label="Perfect")
        plt.xlabel("Predicted probability"); plt.ylabel("Empirical probability")
        plt.title(f"Calibration Curve ({name})"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(f"outputs/calibration_curve_{name}.png", bbox_inches="tight"); plt.close()

    with open("outputs/report.json") as f:
        report = json.load(f)
    report["test"].update({
        "ece_after_isotonic": float(ece_after),
        "pr_auc_after_isotonic": float(ap_after)
    })
    with open("outputs/report.json","w") as f:
        json.dump(report, f, indent=2)

    joblib.dump(calib, "outputs/best_model_calibrated.joblib")
    print({"ap_before": ap_before, "ap_after": ap_after, "ece_before": ece_before, "ece_after": ece_after})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--val_cache", required=True)
    ap.add_argument("--test_cache", required=True)
    main(ap.parse_args())
