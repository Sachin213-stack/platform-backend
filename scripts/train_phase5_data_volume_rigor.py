#!/usr/bin/env python3
"""
AICTO Anomaly Detection - Phase 5: Data Volume & Evaluation Rigor
================================================================
1. Formulates a 3-way temporal split across all 12 businesses:
   - Train Set (First 60% of time window): Model training partition
   - Test Set (Middle 20% of time window): Model tuning partition
   - Out-of-Time Validation Set (Final 20% of time window): Temporal drift evaluation partition
2. Evaluates model robustness across both Test and Out-of-Time Validation partitions.
3. Reports comprehensive per-anomaly-type Precision, Recall, F1, and FPR (previously missing).
4. Persists evaluation metrics and model bundle to scripts/output/.

Outputs:
- scripts/output/phase5_data_volume_rigor_eval.json
- scripts/output/anomaly_model_phase5.joblib
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sqlalchemy import text

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import engine


RAW_SYSTEM_COLS = ["response_time_ms", "cpu_usage_pct", "memory_usage_pct", "queue_depth"]
RAW_BIZ_COLS = ["orders_count", "revenue_amount"]

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"
GROUND_TRUTH_FILE = OUTPUT_DIR / "anomaly_ground_truth.json"
MODEL_OUTPUT_FILE = OUTPUT_DIR / "anomaly_model_phase5.joblib"
EVAL_OUTPUT_FILE = OUTPUT_DIR / "phase5_data_volume_rigor_eval.json"


async def load_telemetry() -> pd.DataFrame:
    print("[1/5] Ingesting telemetry events from PostgreSQL...")
    t0 = time.time()
    async with engine.connect() as conn:
        query = text("""
            SELECT business_id, timestamp,
                   response_time_ms, cpu_usage_pct, memory_usage_pct,
                   queue_depth, orders_count, revenue_amount
            FROM telemetry_events
            ORDER BY business_id, timestamp ASC
        """)
        result = await conn.execute(query)
        df = pd.DataFrame([dict(r._mapping) for r in result.fetchall()])
    print(f"      Loaded {len(df):,} events in {time.time() - t0:.2f}s.")
    return df


def load_and_label_ground_truth(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    print("[2/5] Labeling ground truth & computing temporal drift slopes...")
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["business_id"] = df["business_id"].astype(str)
    df[RAW_SYSTEM_COLS + RAW_BIZ_COLS] = df[RAW_SYSTEM_COLS + RAW_BIZ_COLS].fillna(0.0)

    biz_episodes = defaultdict(list)
    biz_names = {}
    for ep in episodes:
        st = pd.to_datetime(ep["start_time"], utc=True)
        et = pd.to_datetime(ep["end_time"], utc=True)
        biz_episodes[ep["business_id"]].append({
            "episode_id": ep["episode_id"],
            "start_time": st,
            "end_time": et,
            "type": ep["type"],
        })
        biz_names[ep["business_id"]] = ep.get("business_name", ep["business_id"][:8])

    df["ground_truth_label"] = 0
    df["anomaly_type"] = None

    for biz_id, eps in biz_episodes.items():
        biz_mask = df["business_id"] == biz_id
        biz_indices = df[biz_mask].index
        biz_ts = df.loc[biz_indices, "timestamp"]
        for ep in eps:
            in_window = (biz_ts >= ep["start_time"]) & (biz_ts <= ep["end_time"])
            m_idx = biz_indices[in_window]
            if len(m_idx) > 0:
                df.loc[m_idx, "ground_truth_label"] = 1
                df.loc[m_idx, "anomaly_type"] = ep["type"]

    processed_groups = []
    for biz_id, group in df.groupby("business_id", sort=False):
        group = group.sort_values("timestamp").copy()
        group_idx = group.set_index("timestamp")
        m_15m_ago = group_idx["memory_usage_pct"].asof(group_idx.index - pd.Timedelta(minutes=15))
        m_60m_ago = group_idx["memory_usage_pct"].asof(group_idx.index - pd.Timedelta(minutes=60))
        group["mem_slope_15m"] = (group["memory_usage_pct"].values - m_15m_ago.values)
        group["mem_slope_60m"] = (group["memory_usage_pct"].values - m_60m_ago.values)
        processed_groups.append(group)

    df_full = pd.concat(processed_groups, ignore_index=True).fillna(0.0)
    return df_full, biz_names


def three_way_temporal_split(
    df: pd.DataFrame, train_pct: float = 0.60, test_pct: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"[3/5] Constructing 3-way temporal split ({train_pct:.0%} train / {test_pct:.0%} test / {1 - train_pct - test_pct:.0%} validation)...")
    train_dfs, test_dfs, val_dfs = [], [], []
    for biz_id, group in df.groupby("business_id"):
        group = group.sort_values("timestamp")
        min_ts = group["timestamp"].min()
        max_ts = group["timestamp"].max()
        span = max_ts - min_ts
        c1 = min_ts + span * train_pct
        c2 = min_ts + span * (train_pct + test_pct)

        train_dfs.append(group[group["timestamp"] < c1])
        test_dfs.append(group[(group["timestamp"] >= c1) & (group["timestamp"] < c2)])
        val_dfs.append(group[group["timestamp"] >= c2])

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)

    print(f"      Train Set:      {len(train_df):,} rows ({len(train_df)/len(df):.1%}) | {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"      Test Set:       {len(test_df):,} rows ({len(test_df)/len(df):.1%}) | {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"      Validation Set: {len(val_df):,} rows ({len(val_df)/len(df):.1%}) | {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")

    return train_df, test_df, val_df


def evaluate_partition(
    partition_df: pd.DataFrame,
    pipeline: Pipeline,
    thresh_slope_60m: float,
    thresh_slope_15m: float,
    k_debouncing: int = 4,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    X_part = partition_df[RAW_SYSTEM_COLS + RAW_BIZ_COLS]
    pt_preds = np.where(pipeline.predict(X_part) == -1, 1, 0)
    sq_preds = (
        (partition_df["mem_slope_60m"] > thresh_slope_60m)
        & (partition_df["mem_slope_15m"] > 0)
    ).astype(int).values
    raw_flags = np.maximum(pt_preds, sq_preds)

    # Apply debouncing per business
    debounced_flags = []
    p_df = partition_df.copy()
    p_df["flag"] = raw_flags
    for biz_id, grp in p_df.groupby("business_id", sort=False):
        r_sum = grp["flag"].rolling(window=k_debouncing, min_periods=k_debouncing).sum()
        debounced_flags.append((r_sum == k_debouncing).astype(int).fillna(0).values)

    y_pred = np.concatenate(debounced_flags)
    y_true = partition_df["ground_truth_label"].values

    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    overall = {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    # Per-anomaly-type Precision, Recall, F1, and FPR
    per_type = {}
    for atype in ["cpu_saturation", "queue_explosion", "latency_spike", "memory_leak"]:
        y_type = (partition_df["anomaly_type"] == atype).astype(int).values
        p_t = float(precision_score(y_type, y_pred, zero_division=0))
        r_t = float(recall_score(y_type, y_pred, zero_division=0))
        f_t = float(f1_score(y_type, y_pred, zero_division=0))
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_type, y_pred, labels=[0, 1]).ravel()
        fpr_t = float(fp_t / (fp_t + tn_t)) if (fp_t + tn_t) > 0 else 0.0

        per_type[atype] = {
            "precision": round(p_t, 4),
            "recall": round(r_t, 4),
            "f1_score": round(f_t, 4),
            "false_positive_rate": round(fpr_t, 4),
            "tp": int(tp_t),
            "fp": int(fp_t),
            "total_instances": int(y_type.sum()),
        }

    return overall, per_type


def run_phase5_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame):
    print("[4/5] Training model & performing rigorous out-of-time drift evaluation...")
    contamination = float(train_df["ground_truth_label"].mean())

    ct = ColumnTransformer([
        ("sys", StandardScaler(), RAW_SYSTEM_COLS),
        ("biz", RobustScaler(with_centering=False), RAW_BIZ_COLS),
    ])
    pipeline = Pipeline([
        ("ct", ct),
        ("model", IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(train_df[RAW_SYSTEM_COLS + RAW_BIZ_COLS])

    normal_train = train_df[train_df["ground_truth_label"] == 0]
    t60 = float(normal_train["mem_slope_60m"].quantile(0.99))
    t15 = float(normal_train["mem_slope_15m"].quantile(0.95))

    test_overall, test_types = evaluate_partition(test_df, pipeline, t60, t15)
    val_overall, val_types = evaluate_partition(val_df, pipeline, t60, t15)

    results = {
        "phase": "Phase 5 - Data Volume & Evaluation Rigor",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "temporal_partitions": {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "validation_out_of_time_rows": len(val_df),
            "train_span": f"{train_df['timestamp'].min()} to {train_df['timestamp'].max()}",
            "test_span": f"{test_df['timestamp'].min()} to {test_df['timestamp'].max()}",
            "validation_span": f"{val_df['timestamp'].min()} to {val_df['timestamp'].max()}",
        },
        "test_set_evaluation": {
            "overall_metrics": test_overall,
            "per_anomaly_type_metrics": test_types,
        },
        "out_of_time_validation_evaluation": {
            "overall_metrics": val_overall,
            "per_anomaly_type_metrics": val_types,
        },
    }

    return results, pipeline, t60, t15


def save_and_verify(results: Dict[str, Any], pipeline: Pipeline, t60: float, t15: float):
    print("[5/5] Persisting Phase 5 artifacts...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"      Saved eval JSON: {EVAL_OUTPUT_FILE} ({EVAL_OUTPUT_FILE.stat().st_size / 1024:.2f} KB)")

    bundle = {
        "pipeline": pipeline,
        "drift_thresholds": {"thresh_slope_60m": t60, "thresh_slope_15m": t15},
        "phase": "Phase 5",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, MODEL_OUTPUT_FILE)
    print(f"      Saved model bundle: {MODEL_OUTPUT_FILE} ({MODEL_OUTPUT_FILE.stat().st_size / 1024:.2f} KB)")


def print_summary(results: Dict[str, Any]):
    print("\n" + "=" * 85)
    print("PHASE 5 RIGOROUS EVALUATION SUMMARY: TEST SET vs. OUT-OF-TIME VALIDATION SET")
    print("=" * 85)
    t_m = results["test_set_evaluation"]["overall_metrics"]
    v_m = results["out_of_time_validation_evaluation"]["overall_metrics"]

    print(f"{'Partition':<35} | {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'FPR':<10}")
    print("-" * 85)
    print(f"{'Test Set (Middle 20% Window)':<35} | {t_m['precision']:<10.2%} {t_m['recall']:<10.2%} {t_m['f1_score']:<10.4f} {t_m['false_positive_rate']:<10.4%}")
    print(f"{'Validation Set (Final 20% - Drift)':<35} | {v_m['precision']:<10.2%} {v_m['recall']:<10.2%} {v_m['f1_score']:<10.4f} {v_m['false_positive_rate']:<10.4%}")

    print("\n" + "=" * 85)
    print("PER-ANOMALY-TYPE FULL EVALUATION (PRECISION, RECALL, F1, FPR)")
    print("=" * 85)
    print(f"{'Anomaly Type':<18} | {'Part':<5} | {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'FPR':<10}")
    print("-" * 85)
    for atype in ["cpu_saturation", "queue_explosion", "latency_spike", "memory_leak"]:
        tm = results["test_set_evaluation"]["per_anomaly_type_metrics"][atype]
        vm = results["out_of_time_validation_evaluation"]["per_anomaly_type_metrics"][atype]
        print(f"{atype:<18} | Test  | {tm['precision']:<10.2%} {tm['recall']:<10.2%} {tm['f1_score']:<10.4f} {tm['false_positive_rate']:<10.4%}")
        print(f"{'':<18} | Val   | {vm['precision']:<10.2%} {vm['recall']:<10.2%} {vm['f1_score']:<10.4f} {vm['false_positive_rate']:<10.4%}")


def main():
    df = asyncio.run(load_telemetry())
    df_full, biz_names = load_and_label_ground_truth(df)
    train_df, test_df, val_df = three_way_temporal_split(df_full, train_pct=0.60, test_pct=0.20)
    results, pipeline, t60, t15 = run_phase5_experiment(train_df, test_df, val_df)
    save_and_verify(results, pipeline, t60, t15)
    print_summary(results)


if __name__ == "__main__":
    main()
