import sys
from pathlib import Path
import pandas as pd

CSV_MODEL_1 = Path("results/image_detector_1_results.csv")
CSV_MODEL_2 = Path("results/image_detector_2_results.csv")


def evaluate_model_at_threshold(df, score_col, threshold):
    preds = df[score_col].apply(lambda score: "AI" if score >= threshold else "HUMAN")
    actuals = df["actual_label"]
    
    tp = len(df[(actuals == "AI") & (preds == "AI")])
    tn = len(df[(actuals == "HUMAN") & (preds == "HUMAN")])
    fp = len(df[(actuals == "HUMAN") & (preds == "AI")])
    fn = len(df[(actuals == "AI") & (preds == "HUMAN")])

    total = len(df)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1
    }


def evaluate_ensemble_at_thresholds(merged_df, t1, t2):
    p1 = merged_df["ai_score_m1"].apply(lambda s: "AI" if s >= t1 else "HUMAN")
    p2 = merged_df["ai_score_m2"].apply(lambda s: "AI" if s >= t2 else "HUMAN")
    actuals = merged_df["actual_label"]
    
    agreed_preds = []
    strict_preds = []
    
    for pred1, pred2 in zip(p1, p2):
        if pred1 == pred2:
            agreed_preds.append(pred1)
        else:
            agreed_preds.append("UNCERTAIN")
        
        # Strict mode: Fallback to Model 2 on disagreement
        strict_preds.append(pred1 if pred1 == pred2 else pred2)

    total = len(merged_df)
    agreed_mask = [p != "UNCERTAIN" for p in agreed_preds]
    agreed_count = sum(agreed_mask)
    uncertain_count = total - agreed_count
    
    # Conservative Mode Metrics
    tp_cons = len(merged_df[(actuals == "AI") & (pd.Series(agreed_preds) == "AI")])
    tn_cons = len(merged_df[(actuals == "HUMAN") & (pd.Series(agreed_preds) == "HUMAN")])
    fp_cons = len(merged_df[(actuals == "HUMAN") & (pd.Series(agreed_preds) == "AI")])
    fn_cons = len(merged_df[(actuals == "AI") & (pd.Series(agreed_preds) == "HUMAN")])
    
    prec_agreed = tp_cons / (tp_cons + fp_cons) if (tp_cons + fp_cons) > 0 else 0.0
    acc_agreed = (tp_cons + tn_cons) / agreed_count if agreed_count > 0 else 0.0
    acc_overall_cons = (tp_cons + tn_cons) / total if total > 0 else 0.0
    coverage = (agreed_count / total) * 100 if total > 0 else 0.0
    uncertainty_rate = (uncertain_count / total) * 100 if total > 0 else 0.0

    # Strict Mode Metrics
    tp_strict = len(merged_df[(actuals == "AI") & (pd.Series(strict_preds) == "AI")])
    tn_strict = len(merged_df[(actuals == "HUMAN") & (pd.Series(strict_preds) == "HUMAN")])
    fp_strict = len(merged_df[(actuals == "HUMAN") & (pd.Series(strict_preds) == "AI")])
    fn_strict = len(merged_df[(actuals == "AI") & (pd.Series(strict_preds) == "HUMAN")])
    
    acc_strict = (tp_strict + tn_strict) / total if total > 0 else 0.0
    f1_strict = 2 * (tp_strict / (tp_strict + fp_strict)) * (tp_strict / (tp_strict + fn_strict)) / ((tp_strict / (tp_strict + fp_strict)) + (tp_strict / (tp_strict + fn_strict))) if (tp_strict + fp_strict > 0 and tp_strict + fn_strict > 0) else 0.0

    return {
        "t1": t1, "t2": t2,
        "tp_cons": tp_cons, "tn_cons": tn_cons, "fp_cons": fp_cons, "fn_cons": fn_cons,
        "uncertain_count": uncertain_count, "coverage": coverage, "uncertainty_rate": uncertainty_rate,
        "prec_agreed": prec_agreed, "acc_agreed": acc_agreed, "acc_overall_cons": acc_overall_cons,
        "tp_strict": tp_strict, "tn_strict": tn_strict, "fp_strict": fp_strict, "fn_strict": fn_strict,
        "acc_strict": acc_strict
    }


def run_threshold_grid_search():
    print("=" * 80)
    print("COMPREHENSIVE THRESHOLD BENCHMARK (MODEL 1, MODEL 2 & ENSEMBLE)")
    print("=" * 80)

    if not CSV_MODEL_1.exists() or not CSV_MODEL_2.exists():
        print("Error: Missing base evaluation CSV files.")
        sys.exit(1)

    df1 = pd.read_csv(CSV_MODEL_1)
    df2 = pd.read_csv(CSV_MODEL_2)

    merged = pd.merge(
        df1[["image_path", "actual_label", "ai_score"]],
        df2[["image_path", "ai_score"]],
        on="image_path",
        suffixes=("_m1", "_m2")
    )

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    print("\n--- INDIVIDUAL MODEL THRESHOLD BENCHMARK ---")
    print(f"{'Thresh':<8} | {'M1 Acc':<8} | {'M1 Prec':<8} | {'M1 Rec':<8} | {'M1 F1':<8} | {'M2 Acc':<8} | {'M2 Prec':<8} | {'M2 Rec':<8} | {'M2 F1':<8}")
    print("-" * 80)

    for t in thresholds:
        m1 = evaluate_model_at_threshold(df1, "ai_score", t)
        m2 = evaluate_model_at_threshold(df2, "ai_score", t)
        print(f"{t:<8.2f} | {m1['acc']*100:<7.2f}% | {m1['prec']*100:<7.2f}% | {m1['rec']*100:<7.2f}% | {m1['f1']*100:<7.2f}% | {m2['acc']*100:<7.2f}% | {m2['prec']*100:<7.2f}% | {m2['rec']*100:<7.2f}% | {m2['f1']*100:<7.2f}%")

    print("\n--- TWO-MODEL ENSEMBLE GRID SEARCH (CONSERVATIVE vs STRICT) ---")
    print(f"{'M1 Th':<6} | {'M2 Th':<6} | {'Cons Prec':<10} | {'Agreed Acc':<11} | {'Coverage':<9} | {'Uncert Rate':<12} | {'Strict Acc':<10}")
    print("-" * 80)

    for t1 in [0.40, 0.50, 0.60, 0.70]:
        for t2 in [0.40, 0.50, 0.60]:
            res = evaluate_ensemble_at_thresholds(merged, t1, t2)
            print(f"{res['t1']:<6.2f} | {res['t2']:<6.2f} | {res['prec_agreed']*100:<9.2f}% | {res['acc_agreed']*100:<10.2f}% | {res['coverage']:<8.2f}% | {res['uncertainty_rate']:<11.2f}% | {res['acc_strict']*100:<9.2f}%")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_threshold_grid_search()