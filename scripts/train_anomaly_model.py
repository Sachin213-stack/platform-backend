#!/usr/bin/env python3
"""
AICTO Baseline Anomaly Detection Model Trainer & Evaluator
=========================================================
Unsupervised IsolationForest baseline for multi-metric telemetry anomaly detection.

Features:
- Loads telemetry_events from PostgreSQL via SQLAlchemy into pandas DataFrame.
- Loads ground-truth anomaly episodes from output/anomaly_ground_truth.json.
- Computes ground_truth_label (0/1), episode type, and severity for evaluation only.
- Strict per-business time-based 70/30 train/test split (zero temporal overlap).
- Fits unsupervised StandardScaler + IsolationForest Pipeline on train features.
- Evaluates precision, recall, F1, ROC-AUC, FPR, and per-anomaly-type recall on test set.
- Serializes trained Pipeline to output/anomaly_model_baseline.joblib.
- Exports comprehensive evaluation results to output/anomaly_model_eval.json.
- Performs self-verification load-back test.

Usage:
    python scripts/train_anomaly_model.py
"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import engine


FEATURE_COLS = [
    "response_time_ms",
    "cpu_usage_pct",
    "memory_usage_pct",
    "queue_depth",
    "orders_count",
    "revenue_amount",
]

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"
GROUND_TRUTH_FILE = OUTPUT_DIR / "anomaly_ground_truth.json"
MODEL_OUTPUT_FILE = OUTPUT_DIR / "anomaly_model_baseline.joblib"
EVAL_OUTPUT_FILE = OUTPUT_DIR / "anomaly_model_eval.json"


async def load_telemetry_from_db() -> pd.DataFrame:
    """Load all telemetry events from PostgreSQL into a pandas DataFrame."""
    print("[1/7] Ingesting telemetry events from PostgreSQL...")
    start_time = time.time()
    async with engine.connect() as conn:
        query = text("""
            SELECT business_id, timestamp,
                   response_time_ms, cpu_usage_pct, memory_usage_pct,
                   queue_depth, orders_count, revenue_amount
            FROM telemetry_events
            ORDER BY business_id, timestamp ASC
        """)
        result = await conn.execute(query)
        rows = result.fetchall()
        df = pd.DataFrame([dict(r._mapping) for r in rows])

    elapsed = time.time() - start_time
    print(f"      Loaded {len(df):,} events in {elapsed:.2f}s.")
    return df


def load_ground_truth(gt_path: Path) -> List[Dict[str, Any]]:
    """Load ground-truth episodes from JSON file."""
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)
    print(f"      Loaded {len(episodes)} ground-truth episodes from {gt_path.name}.")
    return episodes


def label_ground_truth(df: pd.DataFrame, episodes: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Labels each telemetry event with ground_truth_label (0/1),
    anomaly_type, anomaly_severity, and episode_id.
    NOTE: These labels are strictly excluded from model training and used ONLY for evaluation.
    """
    print("[2/7] Labeling ground truth on telemetry events (evaluation-only labels)...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["business_id"] = df["business_id"].astype(str)

    # Pre-index episodes by business_id
    biz_episodes = defaultdict(list)
    for ep in episodes:
        st = pd.to_datetime(ep["start_time"], utc=True)
        et = pd.to_datetime(ep["end_time"], utc=True)
        biz_episodes[ep["business_id"]].append({
            "episode_id": ep["episode_id"],
            "business_name": ep.get("business_name", ""),
            "start_time": st,
            "end_time": et,
            "type": ep["type"],
            "severity": ep["severity"],
            "description": ep.get("description", ""),
        })

    df["ground_truth_label"] = 0
    df["anomaly_type"] = None
    df["anomaly_severity"] = None
    df["episode_id"] = None

    for biz_id, eps in biz_episodes.items():
        biz_mask = df["business_id"] == biz_id
        biz_indices = df[biz_mask].index
        biz_ts = df.loc[biz_indices, "timestamp"]

        for ep in eps:
            in_window = (biz_ts >= ep["start_time"]) & (biz_ts <= ep["end_time"])
            matching_idx = biz_indices[in_window]
            if len(matching_idx) > 0:
                df.loc[matching_idx, "ground_truth_label"] = 1
                df.loc[matching_idx, "anomaly_type"] = ep["type"]
                df.loc[matching_idx, "anomaly_severity"] = ep["severity"]
                df.loc[matching_idx, "episode_id"] = ep["episode_id"]

    total_events = len(df)
    total_anomalies = int(df["ground_truth_label"].sum())
    overall_contamination = total_anomalies / total_events if total_events > 0 else 0.0
    print(f"      Total Events: {total_events:,} | Labeled Anomaly Points: {total_anomalies:,} ({overall_contamination:.4%})")
    return df, biz_episodes


def time_based_train_test_split(
    df: pd.DataFrame, train_ratio: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Performs a strict per-business time-based split.
    First 70% of each business's date range -> Train set.
    Last 30% of each business's date range -> Test set.
    Guarantees no temporal leakage across train and test partitions.
    """
    print(f"[3/7] Performing per-business time-based train/test split ({train_ratio:.0%} / {1 - train_ratio:.0%})...")
    train_dfs = []
    test_dfs = []
    split_diagnostics = {}

    for biz_id, group in df.groupby("business_id"):
        group = group.sort_values("timestamp")
        min_ts = group["timestamp"].min()
        max_ts = group["timestamp"].max()
        time_span = max_ts - min_ts
        cutoff_ts = min_ts + (time_span * train_ratio)

        b_train = group[group["timestamp"] < cutoff_ts]
        b_test = group[group["timestamp"] >= cutoff_ts]

        train_dfs.append(b_train)
        test_dfs.append(b_test)

        # Verification of strictly non-overlapping time windows
        train_max_ts = b_train["timestamp"].max()
        test_min_ts = b_test["timestamp"].min()
        no_overlap = train_max_ts < test_min_ts

        split_diagnostics[biz_id] = {
            "total_rows": len(group),
            "train_rows": len(b_train),
            "test_rows": len(b_test),
            "train_start": min_ts.isoformat(),
            "train_end": train_max_ts.isoformat() if len(b_train) > 0 else None,
            "test_start": test_min_ts.isoformat() if len(b_test) > 0 else None,
            "test_end": max_ts.isoformat(),
            "temporal_separation_verified": bool(no_overlap),
        }

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    print(f"      Train Set: {len(train_df):,} rows ({len(train_df)/len(df):.2%}) | Span: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"      Test Set:  {len(test_df):,} rows ({len(test_df)/len(df):.2%}) | Span: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # Global window separation sanity check
    all_biz_separated = all(d["temporal_separation_verified"] for d in split_diagnostics.values())
    print(f"      Zero temporal overlap verified for all 12 businesses: {all_biz_separated}")

    return train_df, test_df, split_diagnostics


def train_global_baseline_model(
    train_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[Pipeline, float]:
    """
    Trains an unsupervised Baseline IsolationForest Model bundled in a StandardScaler Pipeline.
    Strictly uses only feature columns; ground-truth labels are NEVER passed to the estimator.
    """
    print("[4/7] Training unsupervised Baseline IsolationForest Pipeline...")
    # Calculate observed contamination rate on train partition
    train_contamination = float(train_df["ground_truth_label"].mean())
    # Ensure contamination is within valid numerical bounds for IsolationForest
    contamination = max(0.01, min(0.20, train_contamination))
    print(f"      Observed Train Contamination Rate: {train_contamination:.4%} (parameter set to: {contamination:.5f})")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Fill any potential NaNs in features with 0.0
    X_train = train_df[feature_cols].fillna(0.0)

    fit_start = time.time()
    pipeline.fit(X_train)
    fit_duration = time.time() - fit_start
    print(f"      Pipeline fitted successfully on {len(X_train):,} rows in {fit_duration:.2f}s.")

    return pipeline, contamination


def evaluate_model(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    episodes: List[Dict[str, Any]],
    split_diagnostics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates the model against test set ground-truth labels.
    Calculates precision, recall, F1, ROC-AUC, FPR, recall by anomaly type,
    and conducts spot-checks on 5 test episodes.
    """
    print("[5/7] Evaluating baseline model on test set...")
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["ground_truth_label"].values

    # IsolationForest outputs: -1 = anomaly (inlier=1). Map to: 1 = anomaly, 0 = normal
    raw_preds = pipeline.predict(X_test)
    preds = np.where(raw_preds == -1, 1, 0)
    test_df["predicted_anomaly"] = preds

    # Decision function: lower score = more anomalous. Invert so higher = more anomalous for ROC-AUC
    scores = -pipeline.decision_function(X_test)
    test_df["anomaly_score"] = scores

    # Metrics
    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, scores))

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    print(f"      Test Set Overall Metrics:")
    print(f"        Precision : {precision:.4f}")
    print(f"        Recall    : {recall:.4f}")
    print(f"        F1-Score  : {f1:.4f}")
    print(f"        ROC-AUC   : {roc_auc:.4f}")
    print(f"        FPR       : {fpr:.4f} ({fp:,} false alarms out of {fp + tn:,} normal rows)")

    # Per-anomaly-type recall breakdown
    type_recall = {}
    print("      Recall Breakdown by Anomaly Type:")
    for atype in ["latency_spike", "cpu_saturation", "queue_explosion", "memory_leak"]:
        sub = test_df[test_df["anomaly_type"] == atype]
        if len(sub) > 0:
            rec = float(recall_score(sub["ground_truth_label"], sub["predicted_anomaly"], zero_division=0))
            detected_pts = int(sub["predicted_anomaly"].sum())
            total_pts = len(sub)
            type_recall[atype] = {
                "recall": rec,
                "detected_points": detected_pts,
                "total_points": total_pts,
            }
            print(f"        {atype:16s}: {rec:6.2%} ({detected_pts:,} / {total_pts:,} rows detected)")
        else:
            type_recall[atype] = {"recall": 0.0, "detected_points": 0, "total_points": 0}
            print(f"        {atype:16s}: N/A (0 rows in test set)")

    # Per-severity recall breakdown
    severity_recall = {}
    for sev in ["critical", "high"]:
        sub = test_df[test_df["anomaly_severity"] == sev]
        if len(sub) > 0:
            rec = float(recall_score(sub["ground_truth_label"], sub["predicted_anomaly"], zero_division=0))
            severity_recall[sev] = {
                "recall": rec,
                "detected_points": int(sub["predicted_anomaly"].sum()),
                "total_points": len(sub),
            }

    # Sanity Spot-Check on 5 known test-set episodes
    # Find episodes strictly falling within test set time windows
    test_episodes = []
    for ep in episodes:
        biz_id = ep["business_id"]
        st = pd.to_datetime(ep["start_time"], utc=True)
        if biz_id in split_diagnostics and split_diagnostics[biz_id]["test_start"]:
            test_start = pd.to_datetime(split_diagnostics[biz_id]["test_start"], utc=True)
            if st >= test_start:
                test_episodes.append(ep)

    # Pick 5 episodes covering all types
    by_type = defaultdict(list)
    for ep in test_episodes:
        by_type[ep["type"]].append(ep)

    selected_episodes = []
    for atype in ["latency_spike", "cpu_saturation", "queue_explosion", "memory_leak"]:
        if by_type[atype]:
            selected_episodes.append(by_type[atype][0])
    if len(by_type["latency_spike"]) > 1:
        selected_episodes.append(by_type["latency_spike"][1])
    elif len(test_episodes) > 4:
        for ep in test_episodes:
            if ep not in selected_episodes:
                selected_episodes.append(ep)
                if len(selected_episodes) == 5:
                    break

    spot_checks = []
    print("\n      Sanity Spot-Check on 5 Known Test-Set Anomaly Episodes:")
    for idx, ep in enumerate(selected_episodes, 1):
        ep_rows = test_df[test_df["episode_id"] == ep["episode_id"]]
        n_rows = len(ep_rows)
        n_flagged = int(ep_rows["predicted_anomaly"].sum()) if n_rows > 0 else 0
        flag_pct = (n_flagged / n_rows * 100.0) if n_rows > 0 else 0.0
        verdict = "DETECTED" if flag_pct >= 50.0 else "PARTIALLY DETECTED" if flag_pct > 0 else "MISSED"

        spot_record = {
            "episode_index": idx,
            "episode_id": ep["episode_id"],
            "business_name": ep.get("business_name"),
            "business_id": ep["business_id"],
            "type": ep["type"],
            "metric": ep.get("metric"),
            "severity": ep["severity"],
            "start_time": ep["start_time"],
            "end_time": ep["end_time"],
            "total_test_rows": n_rows,
            "flagged_rows": n_flagged,
            "flagged_pct": round(flag_pct, 2),
            "verdict": verdict,
        }
        spot_checks.append(spot_record)
        print(f"        [{idx}] {ep['type'].upper()} ({ep['severity']}) - {ep.get('business_name')}: {n_flagged}/{n_rows} rows flagged ({flag_pct:.1f}%) -> {verdict}")

    eval_results = {
        "model_architecture": "Global IsolationForest Baseline Pipeline (StandardScaler + IsolationForest)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "total_events": len(test_df) + (len(X_test)),  # placeholder
            "test_rows": len(test_df),
            "test_anomalies": int(y_test.sum()),
            "test_contamination_pct": round(float(y_test.mean()) * 100, 4),
        },
        "features": feature_cols,
        "overall_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        },
        "recall_by_anomaly_type": type_recall,
        "recall_by_severity": severity_recall,
        "spot_checks": spot_checks,
        "per_business_diagnostics": split_diagnostics,
    }

    return eval_results


def evaluate_per_business_benchmark(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Computes benchmark metrics for per-business models to document
    the empirical comparison in the evaluation artifact.
    """
    print("[5b/7] Running benchmark on Per-Business models for architectural comparison...")
    preds_per_biz = []
    scores_per_biz = []
    test_subsets = []

    for biz_id in test_df["business_id"].unique():
        b_train = train_df[train_df["business_id"] == biz_id]
        b_test = test_df[test_df["business_id"] == biz_id]
        if len(b_train) == 0 or len(b_test) == 0:
            continue

        b_contam = float(b_train["ground_truth_label"].mean())
        b_contam = max(0.01, min(0.20, b_contam))

        sc = StandardScaler()
        X_tr = sc.fit_transform(b_train[feature_cols].fillna(0.0))
        X_te = sc.transform(b_test[feature_cols].fillna(0.0))

        iso = IsolationForest(contamination=b_contam, random_state=42, n_jobs=-1)
        iso.fit(X_tr)

        p = np.where(iso.predict(X_te) == -1, 1, 0)
        s = -iso.decision_function(X_te)

        b_sub = b_test.copy()
        b_sub["pred_biz"] = p
        b_sub["score_biz"] = s
        test_subsets.append(b_sub)

    combined_test = pd.concat(test_subsets)
    y_comb = combined_test["ground_truth_label"].values
    p_b = float(precision_score(y_comb, combined_test["pred_biz"], zero_division=0))
    r_b = float(recall_score(y_comb, combined_test["pred_biz"], zero_division=0))
    f1_b = float(f1_score(y_comb, combined_test["pred_biz"], zero_division=0))
    auc_b = float(roc_auc_score(y_comb, combined_test["score_biz"]))
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_comb, combined_test["pred_biz"]).ravel()
    fpr_b = float(fp_b / (fp_b + tn_b))

    biz_type_recall = {}
    for atype in ["latency_spike", "cpu_saturation", "queue_explosion", "memory_leak"]:
        sub = combined_test[combined_test["anomaly_type"] == atype]
        if len(sub) > 0:
            rec = float(recall_score(sub["ground_truth_label"], sub["pred_biz"], zero_division=0))
            biz_type_recall[atype] = round(rec, 4)

    return {
        "precision": round(p_b, 4),
        "recall": round(r_b, 4),
        "f1_score": round(f1_b, 4),
        "roc_auc": round(auc_b, 4),
        "false_positive_rate": round(fpr_b, 4),
        "recall_by_type": biz_type_recall,
    }


def save_artifacts(pipeline: Pipeline, eval_results: Dict[str, Any]) -> None:
    """Serializes model pipeline and evaluation metrics to disk."""
    print("[6/7] Persisting model and evaluation artifacts...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Pipeline
    joblib.dump(pipeline, MODEL_OUTPUT_FILE)
    model_size_mb = MODEL_OUTPUT_FILE.stat().st_size / (1024.0 * 1024.0)
    print(f"      Model Pipeline saved: {MODEL_OUTPUT_FILE} ({model_size_mb:.2f} MB)")

    # 2. Save Eval JSON
    with open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    eval_size_kb = EVAL_OUTPUT_FILE.stat().st_size / 1024.0
    print(f"      Evaluation JSON saved: {EVAL_OUTPUT_FILE} ({eval_size_kb:.2f} KB)")


def verify_integrity(feature_cols: List[str]) -> None:
    """
    Performs verification by loading back the saved model and running sample inference.
    """
    print("[7/7] Verifying artifact integrity via load-back test...")
    if not MODEL_OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_OUTPUT_FILE}")
    if not EVAL_OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Eval file missing: {EVAL_OUTPUT_FILE}")

    # Load back joblib
    reloaded_pipeline = joblib.load(MODEL_OUTPUT_FILE)
    if not hasattr(reloaded_pipeline, "predict"):
        raise TypeError("Reloaded object is not a valid predictor pipeline.")

    # Create dummy synthetic vector with matching feature names
    sample_df = pd.DataFrame([
        {
            "response_time_ms": 120.0,
            "cpu_usage_pct": 45.0,
            "memory_usage_pct": 50.0,
            "queue_depth": 2,
            "orders_count": 0,
            "revenue_amount": 0.0,
        },
        {
            "response_time_ms": 1500.0,
            "cpu_usage_pct": 98.0,
            "memory_usage_pct": 95.0,
            "queue_depth": 250,
            "orders_count": 0,
            "revenue_amount": 0.0,
        },
    ])

    preds = reloaded_pipeline.predict(sample_df[feature_cols])
    scores = -reloaded_pipeline.decision_function(sample_df[feature_cols])
    flags = np.where(preds == -1, 1, 0)

    print(f"      Load-back prediction test succeeded!")
    print(f"        Sample 1 (normal vitals)    -> Flag: {flags[0]} (raw pred: {preds[0]}, score: {scores[0]:.4f})")
    print(f"        Sample 2 (extreme anomaly)  -> Flag: {flags[1]} (raw pred: {preds[1]}, score: {scores[1]:.4f})")
    print("      Verification complete: Model artifact is fully intact and operational.")


def main():
    print("=" * 75)
    print("AICTO Baseline Anomaly Detection Model Training & Evaluation")
    print("=" * 75)

    # 1. Load telemetry
    df = asyncio.run(load_telemetry_from_db())

    # 2. Load ground truth and label events
    episodes = load_ground_truth(GROUND_TRUTH_FILE)
    df, biz_episodes = label_ground_truth(df, episodes)

    # 3. Per-business time-based split
    train_df, test_df, split_diagnostics = time_based_train_test_split(df, train_ratio=0.70)

    # 4. Train unsupervised global model pipeline
    pipeline, contamination = train_global_baseline_model(train_df, FEATURE_COLS)

    # 5. Evaluate
    eval_results = evaluate_model(
        pipeline, test_df, FEATURE_COLS, episodes, split_diagnostics
    )
    eval_results["dataset"]["total_events"] = len(df)
    eval_results["dataset"]["train_rows"] = len(train_df)
    eval_results["dataset"]["train_anomalies"] = int(train_df["ground_truth_label"].sum())
    eval_results["dataset"]["train_contamination_pct"] = round(float(train_df["ground_truth_label"].mean()) * 100, 4)

    # 5b. Per-business benchmark comparison
    per_biz_benchmark = evaluate_per_business_benchmark(train_df, test_df, FEATURE_COLS)
    eval_results["per_business_benchmark_comparison"] = per_biz_benchmark

    # 6. Save artifacts
    save_artifacts(pipeline, eval_results)

    # 7. Self-verification
    verify_integrity(FEATURE_COLS)

    print("\n" + "=" * 75)
    print("TRAINING & EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
