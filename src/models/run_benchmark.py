"""Consolidated benchmark runner: evaluate every trained model in one command.

This is a read-and-evaluate consumer of work done in earlier phases. It does not
train anything by default. It:

  1. Loads the already-trained base models (random_forest.pkl, xgboost.pkl,
     lightgbm.pkl) and the stacking meta-classifier (stacking_meta.pkl) from
     src/models/artifacts/, if they exist.
  2. Evaluates all four models (RF, XGBoost, LightGBM, Stacked Ensemble) on the
     TEST set at the default 0.5 decision threshold, using the same metric
     definitions and the same consolidated comparison table format that
     train_stacking_ensemble.py writes into final_comparison.json.
  3. Reprints the threshold tuning summary from the existing
     src/models/artifacts/threshold_tuning.json (read only, never regenerated
     here).
  4. Prints everything to the console AND saves a fresh timestamped copy of the
     whole report to src/models/artifacts/benchmark_report_<date>.json.

With --retrain, it first re-runs the existing training scripts
(train_random_forest.py, train_xgboost.py, train_lightgbm.py,
train_stacking_ensemble.py) in order, then evaluates. Those scripts regenerate
their own .pkl models and companion metric JSON files (including
final_comparison.json); that is a deliberate, opt-in side effect of --retrain.
Without --retrain, this script only ever writes benchmark_report_<date>.json and
touches no existing artifact.

Positive class = phishing. A false positive is a legitimate email flagged as
phishing, so FPR = FP / (FP + TN) over the legitimate rows.

Outputs:
  src/models/artifacts/benchmark_report_<date>.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

# run_benchmark.py lives alongside the training scripts; make sure they import
# whether this file is run as a path or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_stacking_ensemble import (  # noqa: E402
    BASE_MODEL_NAMES,
    evaluate_by_source,
    fmt,
    load_split,
    print_report,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "src" / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

BASE_MODEL_PATHS = {
    "random_forest": ARTIFACTS_DIR / "random_forest.pkl",
    "xgboost": ARTIFACTS_DIR / "xgboost.pkl",
    "lightgbm": ARTIFACTS_DIR / "lightgbm.pkl",
}
META_PATH = ARTIFACTS_DIR / "stacking_meta.pkl"
THRESHOLD_TUNING_PATH = ARTIFACTS_DIR / "threshold_tuning.json"

# Re-run in this order: base models first, then the stacking meta-classifier
# which depends on them.
RETRAIN_SCRIPTS = [
    "train_random_forest.py",
    "train_xgboost.py",
    "train_lightgbm.py",
    "train_stacking_ensemble.py",
]

DEFAULT_THRESHOLD = 0.5

DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "stacked_ensemble": "Stacked Ensemble",
}

SYNTHETIC_SLICE = "synthetic_healthcare"


def retrain_all() -> None:
    """Re-run the existing training scripts in dependency order.

    Each script is invoked exactly as it would be by hand
    (`python src/models/train_*.py`), so its behaviour and its outputs are
    unchanged. This regenerates the .pkl models and their companion metric JSON
    files; that is the point of --retrain.
    """
    print("=" * 78)
    print("--retrain: re-running training scripts (this regenerates .pkl models")
    print("and their companion metric JSON files, including final_comparison.json)")
    print("=" * 78)
    for script in RETRAIN_SCRIPTS:
        script_path = MODELS_DIR / script
        print(f"\n>>> {sys.executable} {script_path}")
        subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT), check=True)
    print("\n" + "=" * 78)
    print("--retrain: all training scripts finished")
    print("=" * 78)


def load_models() -> tuple[dict, object | None]:
    """Load whatever trained artifacts are on disk.

    Missing base models are a hard error (there is nothing to benchmark). A
    missing meta-classifier only drops the stacked ensemble from the run.
    """
    missing_base = [name for name, p in BASE_MODEL_PATHS.items() if not p.exists()]
    if missing_base:
        raise SystemExit(
            "missing base model artifact(s): "
            + ", ".join(f"{name} ({BASE_MODEL_PATHS[name].name})" for name in missing_base)
            + ".\nTrain them first, or re-run this script with --retrain."
        )

    base_models = {name: joblib.load(p) for name, p in BASE_MODEL_PATHS.items()}

    meta_clf = None
    if META_PATH.exists():
        meta_clf = joblib.load(META_PATH)
    else:
        print(
            f"WARNING: {META_PATH.name} not found; the stacked ensemble will be "
            "skipped. Re-run with --retrain to build it."
        )
    return base_models, meta_clf


def evaluate_on_test(base_models: dict, meta_clf) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate every available model on the test set at the default threshold.

    Returns (results_by_model, y_test, test_source). results_by_model mirrors the
    `results_by_model` structure in final_comparison.json: each entry has
    `test_overall` and `test_by_source_dataset`.
    """
    X_test, y_test, test_source = load_split("test")
    print(
        f"\ntest: {X_test.shape}  phishing={int(y_test.sum())}  "
        f"legit={int((y_test == 0).sum())}"
    )

    results: dict = {}
    test_probs: dict = {}
    for name in BASE_MODEL_NAMES:
        score = base_models[name].predict_proba(X_test)[:, 1]
        test_probs[name] = score
        pred = (score >= DEFAULT_THRESHOLD).astype(int)
        overall, by_source = evaluate_by_source(y_test, pred, score, test_source)
        print_report(
            f"TEST SET - {DISPLAY_NAMES[name].upper()} (individual baseline)",
            overall,
            by_source,
        )
        results[name] = {"test_overall": overall, "test_by_source_dataset": by_source}

    if meta_clf is not None:
        X_meta_test = np.column_stack([test_probs[name] for name in BASE_MODEL_NAMES])
        ens_score = meta_clf.predict_proba(X_meta_test)[:, 1]
        ens_pred = meta_clf.predict(X_meta_test)
        overall, by_source = evaluate_by_source(y_test, ens_pred, ens_score, test_source)
        print_report("TEST SET - STACKED ENSEMBLE", overall, by_source)
        results["stacked_ensemble"] = {
            "test_overall": overall,
            "test_by_source_dataset": by_source,
        }

    return results, y_test, test_source


def model_order(results: dict) -> list[str]:
    order = [name for name in BASE_MODEL_NAMES if name in results]
    if "stacked_ensemble" in results:
        order.append("stacked_ensemble")
    return order


def print_headline_table(results: dict) -> dict:
    """Consolidated comparison table, same columns as
    `headline_comparison_test_overall` in final_comparison.json.
    """
    print("\n" + "-" * 78)
    print("CONSOLIDATED COMPARISON - ALL MODELS ON TEST (overall, positive = phishing)")
    print("-" * 78)
    print(
        f"{'model':<20} {'prec':>8} {'recall':>8} {'f1':>8} {'fpr':>8} {'auc':>8}"
        "   (fpr target < 0.005)"
    )
    headline: dict = {}
    for name in model_order(results):
        o = results[name]["test_overall"]
        headline[name] = {
            "precision": o["precision"],
            "recall": o["recall"],
            "f1": o["f1"],
            "false_positive_rate": o["false_positive_rate"],
            "auc_roc": o["auc_roc"],
        }
        print(
            f"{DISPLAY_NAMES[name]:<20} {fmt(o['precision']):>8} {fmt(o['recall']):>8} "
            f"{fmt(o['f1']):>8} {fmt(o['false_positive_rate']):>8} {fmt(o['auc_roc']):>8}"
        )
    return headline


def print_synthetic_table(results: dict) -> dict:
    """FPR / recall on the synthetic_healthcare slice for every model, matching
    `synthetic_healthcare_summary` in final_comparison.json.
    """
    print("\n" + "-" * 78)
    print(f"{SYNTHETIC_SLICE.upper()} SLICE - ALL MODELS ON TEST")
    print("-" * 78)
    print(f"{'model':<20} {'fpr':>8} {'recall':>8}")
    summary: dict = {}
    for name in model_order(results):
        slice_metrics = results[name]["test_by_source_dataset"].get(SYNTHETIC_SLICE)
        if slice_metrics is None:
            continue
        summary[name] = {
            "fpr": slice_metrics["false_positive_rate"],
            "recall": slice_metrics["recall"],
        }
        print(
            f"{DISPLAY_NAMES[name]:<20} {fmt(slice_metrics['false_positive_rate']):>8} "
            f"{fmt(slice_metrics['recall']):>8}"
        )
    return summary


def compute_rq1_answer(results: dict) -> dict | None:
    """Does the stacked ensemble beat the best individual baseline on test?

    Same shape as `rq1_answer` in final_comparison.json. Returns None when the
    ensemble is not part of this run.
    """
    base = [name for name in BASE_MODEL_NAMES if name in results]
    if "stacked_ensemble" not in results or not base:
        return None
    ens = results["stacked_ensemble"]["test_overall"]

    def best(metric: str, mode: str) -> tuple[str | None, float | None]:
        vals = {
            name: results[name]["test_overall"][metric]
            for name in base
            if results[name]["test_overall"][metric] is not None
        }
        if not vals:
            return None, None
        picker = max if mode == "max" else min
        name = picker(vals, key=vals.get)
        return name, vals[name]

    f1_name, f1_val = best("f1", "max")
    fpr_name, fpr_val = best("false_positive_rate", "min")
    auc_name, auc_val = best("auc_roc", "max")

    return {
        "best_individual_baseline_by_f1": f1_name,
        "best_individual_baseline_f1": f1_val,
        "ensemble_f1": ens["f1"],
        "f1_delta_ensemble_minus_best_baseline": (
            ens["f1"] - f1_val if f1_val is not None else None
        ),
        "ensemble_outperforms_best_baseline_on_f1": bool(
            f1_val is not None and ens["f1"] > f1_val
        ),
        "best_individual_baseline_by_fpr": fpr_name,
        "best_individual_baseline_fpr": fpr_val,
        "ensemble_fpr": ens["false_positive_rate"],
        "fpr_delta_ensemble_minus_best_baseline": (
            ens["false_positive_rate"] - fpr_val if fpr_val is not None else None
        ),
        "ensemble_improves_fpr_vs_best_baseline": bool(
            fpr_val is not None and ens["false_positive_rate"] < fpr_val
        ),
        "best_individual_baseline_by_auc": auc_name,
        "best_individual_baseline_auc": auc_val,
        "ensemble_auc": ens["auc_roc"],
        "auc_delta_ensemble_minus_best_baseline": (
            ens["auc_roc"] - auc_val if auc_val is not None else None
        ),
        "ensemble_improves_auc_vs_best_baseline": bool(
            auc_val is not None and ens["auc_roc"] > auc_val
        ),
    }


def print_rq1_answer(rq1: dict | None) -> None:
    print("\n" + "-" * 78)
    print("RQ1 ANSWER - stacked ensemble vs best individual baseline (test)")
    print("-" * 78)
    if rq1 is None:
        print("stacked ensemble not in this run; RQ1 comparison skipped.")
        return
    print(
        f"best baseline by F1  : {rq1['best_individual_baseline_by_f1']} "
        f"(F1={fmt(rq1['best_individual_baseline_f1'])})"
    )
    print(
        f"ensemble F1          : {fmt(rq1['ensemble_f1'])}  "
        f"(delta {rq1['f1_delta_ensemble_minus_best_baseline']:+.4f})"
    )
    print(
        f"best baseline by FPR : {rq1['best_individual_baseline_by_fpr']} "
        f"(FPR={fmt(rq1['best_individual_baseline_fpr'])})"
    )
    print(
        f"ensemble FPR         : {fmt(rq1['ensemble_fpr'])}  "
        f"(delta {rq1['fpr_delta_ensemble_minus_best_baseline']:+.4f})"
    )
    print(
        f"best baseline by AUC : {rq1['best_individual_baseline_by_auc']} "
        f"(AUC={fmt(rq1['best_individual_baseline_auc'])})"
    )
    print(
        f"ensemble AUC         : {fmt(rq1['ensemble_auc'])}  "
        f"(delta {rq1['auc_delta_ensemble_minus_best_baseline']:+.4f})"
    )


def load_threshold_tuning() -> dict | None:
    if not THRESHOLD_TUNING_PATH.exists():
        print(
            f"\nWARNING: {THRESHOLD_TUNING_PATH.name} not found; threshold tuning "
            "summary skipped."
        )
        return None
    with open(THRESHOLD_TUNING_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def summarise_threshold_tuning(tuning: dict | None) -> dict | None:
    """Pull the summary rows out of the existing threshold_tuning.json. Read
    only: this script never recomputes or rewrites that file.
    """
    if tuning is None:
        return None
    validation = tuning.get("validation", {})
    test = tuning.get("test", {})
    return {
        "read_from": str(THRESHOLD_TUNING_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_created_utc": tuning.get("created_utc"),
        "selection_set": tuning.get("selection_set"),
        "validation_default_threshold_0_5": validation.get("default_threshold_0_5"),
        "validation_fpr_target_results": validation.get("fpr_target_results"),
        "selected_threshold": validation.get("selected_threshold"),
        "selected_threshold_target": validation.get("selected_threshold_target"),
        "test_at_default_threshold_0_5": test.get("at_default_threshold_0_5"),
        "test_at_selected_threshold": test.get("at_selected_threshold"),
        "test_auc_roc": test.get("auc_roc"),
        "test_recall_drop_selected_vs_default": test.get("recall_drop_selected_vs_default"),
    }


def _threshold_row(label: str, m: dict | None) -> str:
    if not m:
        return f"{label:<26} (not available)"
    return (
        f"{label:<26} thr={m.get('threshold', float('nan')):.4f}  "
        f"prec={fmt(m.get('precision'))}  recall={fmt(m.get('recall'))}  "
        f"f1={fmt(m.get('f1'))}  fpr={fmt(m.get('false_positive_rate'))}"
    )


def print_threshold_summary(summary: dict | None) -> None:
    print("\n" + "-" * 78)
    print("THRESHOLD TUNING SUMMARY (stacked ensemble; read from threshold_tuning.json)")
    print("-" * 78)
    if summary is None:
        print("threshold_tuning.json not available.")
        return
    print(
        f"selection set: {summary.get('selection_set')}   "
        f"source created: {summary.get('source_created_utc')}"
    )

    print("\nvalidation - reference point and FPR targets (max recall s.t. FPR < target):")
    print(_threshold_row("default (0.5)", summary.get("validation_default_threshold_0_5")))
    fpr_targets = summary.get("validation_fpr_target_results") or {}
    for target in sorted(fpr_targets, key=float):
        print(_threshold_row(f"FPR < {target}", fpr_targets[target]))

    print(
        f"\nselected threshold: {summary.get('selected_threshold')}   "
        f"target: {summary.get('selected_threshold_target')}"
    )

    print("\ntest - selected threshold vs default 0.5:")
    print(_threshold_row("default (0.5)", summary.get("test_at_default_threshold_0_5")))
    print(
        _threshold_row(
            f"selected ({summary.get('selected_threshold')})",
            summary.get("test_at_selected_threshold"),
        )
    )
    print(f"test auc-roc (threshold-independent): {fmt(summary.get('test_auc_roc'))}")
    recall_drop = summary.get("test_recall_drop_selected_vs_default")
    if recall_drop is not None:
        print(
            f"recall cost of the stricter threshold: {recall_drop * 100:.2f} "
            "percentage points"
        )


def build_report(
    retrain: bool,
    results: dict,
    y_test: np.ndarray,
    headline: dict,
    synthetic_summary: dict,
    rq1: dict | None,
    threshold_summary: dict | None,
) -> dict:
    return {
        "description": (
            "Consolidated benchmark: all available trained models evaluated on "
            "the test set in a single run, at the default 0.5 threshold, plus "
            "the threshold tuning summary read from threshold_tuning.json."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "src/models/run_benchmark.py",
        "retrained": retrain,
        "positive_class": "phishing",
        "decision_threshold": DEFAULT_THRESHOLD,
        "test_set_touched": True,
        "test_set_touched_note": (
            "run_benchmark.py evaluates already-trained models on the test set; "
            "it never trains or tunes on test."
        ),
        "models_evaluated": model_order(results),
        "data": {"test_rows": int(len(y_test))},
        "results_by_model": results,
        "headline_comparison_test_overall": headline,
        "synthetic_healthcare_summary": synthetic_summary,
        "rq1_answer": rq1,
        "threshold_tuning_summary": threshold_summary,
        "source_artifacts": {
            "base_models": {
                name: str(path.relative_to(ROOT)).replace("\\", "/")
                for name, path in BASE_MODEL_PATHS.items()
            },
            "meta_classifier": (
                str(META_PATH.relative_to(ROOT)).replace("\\", "/")
                if META_PATH.exists()
                else None
            ),
            "threshold_tuning": (
                str(THRESHOLD_TUNING_PATH.relative_to(ROOT)).replace("\\", "/")
                if THRESHOLD_TUNING_PATH.exists()
                else None
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidated benchmark runner. Loads the already-trained models and "
            "evaluates all four on the test set. Use --retrain to re-run the "
            "training scripts first."
        )
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help=(
            "Re-run train_random_forest.py, train_xgboost.py, train_lightgbm.py "
            "and train_stacking_ensemble.py before evaluating. This regenerates "
            "their .pkl models and companion metric JSON files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("CONSOLIDATED MODEL BENCHMARK")
    print("=" * 78)
    print(
        "Evaluates all trained models on the TEST set. No training on test; "
        "no existing artifact is modified unless --retrain is passed."
    )

    if args.retrain:
        retrain_all()
    else:
        print("\n(load-and-evaluate only; pass --retrain to rebuild the models first)")

    base_models, meta_clf = load_models()
    results, y_test, _ = evaluate_on_test(base_models, meta_clf)

    headline = print_headline_table(results)
    synthetic_summary = print_synthetic_table(results)
    rq1 = compute_rq1_answer(results)
    print_rq1_answer(rq1)

    tuning = load_threshold_tuning()
    threshold_summary = summarise_threshold_tuning(tuning)
    print_threshold_summary(threshold_summary)

    report = build_report(
        retrain=args.retrain,
        results=results,
        y_test=y_test,
        headline=headline,
        synthetic_summary=synthetic_summary,
        rq1=rq1,
        threshold_summary=threshold_summary,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    out_path = ARTIFACTS_DIR / f"benchmark_report_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"saved benchmark report -> {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
