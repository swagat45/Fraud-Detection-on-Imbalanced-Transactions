import argparse, json

TEMPLATE = (
    "- Built fraud detector on imbalanced data; **XGBoost** improved **PR-AUC {ap:.3f}** "
    "(post-iso: {ap_cal:.3f}); at **1% FPR** recall was **{rec:.3f}**.\n"
    "- **Isotonic calibration** reduced **ECE {ece_b:.3f} → {ece_a:.3f}**; reproducible stratified splits; "
    "PR/confusion/calibration plots in repo."
)

def main(args):
    with open(args.report_path) as f:
        rep_all = json.load(f)
    rep = rep_all.get("test", rep_all)
    ap = rep["pr_auc"]
    rec = rep["recall_at_1pct_fpr"]
    ece_b = rep["ece_before_isotonic"]
    ap_cal = rep.get("pr_auc_after_isotonic", ap)
    ece_a = rep.get("ece_after_isotonic", ece_b)
    print(TEMPLATE.format(ap=ap, rec=rec, ece_b=ece_b, ap_cal=ap_cal, ece_a=ece_a))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_path", required=True)
    main(ap.parse_args())
