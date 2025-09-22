import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def recall_at_fpr(y_true, y_scores, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    mask = fpr <= target_fpr
    if not np.any(mask):
        idx = np.argmin(np.abs(fpr - target_fpr))
    else:
        idx = np.where(mask)[0][-1]
    return tpr[idx], thr[idx]

def plot_pr(y_true, y_scores, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AP={ap:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    import numpy as np
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.title("Confusion Matrix"); plt.colorbar(im)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def expected_calibration_error(y_true, y_prob, n_bins=10):
    import numpy as np
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.mean(mask)
    return float(ece)
