#!/usr/bin/env python3
"""
AICTO Anomaly Detection - Phase 6: Hybrid Hierarchical Model Architecture
========================================================================
1. Implements a Hybrid Hierarchical Anomaly Detector:
   - Global Model acts as the cold-start default for onboarding / immature tenants (< 30 days).
   - Per-Tenant Models activate automatically when a tenant has mature telemetry (> 30 days).
2. Generates an exhaustive Global vs. Per-Tenant FPR and performance comparison table across all 12 tenants.
3. Integrates calibrated alert debouncing and drift slope detection.
4. Serializes the complete production pipeline bundle to scripts/output/.

Outputs:
- scripts/output/phase6_hybrid_hierarchical_eval.json
- scripts/output/anomaly_model_phase6_production.joblib
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
MODEL_OUTPUT_FILE = OUTPUT_DIR / "anomaly_model_phase6_production.joblib"
EVAL_OUTPUT_FILE = OUTPUT_DIR / "phase6_hybrid_hierarchical_eval.json"


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


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dfs, test_dfs = [], []
    for biz_id, group in df.groupby("business_id"):
        group = group.sort_values("timestamp")
        min_ts = group["timestamp"].min()
        max_ts = group["timestamp"].max()
        cutoff = min_ts + (max_ts - min_ts) * train_ratio
        train_dfs.append(group[group["timestamp"] < cutoff])
        test_dfs.append(group[group["timestamp"] >= cutoff])

    return pd.concat(train_dfs).reset_index(drop=True), pd.concat(test_dfs).reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_phase6_hierarchical_comparison(
    train_df: pd.DataFrame, test_df: pd.DataFrame, biz_names: Dict[str, str]
):
    print("[3/5] Fitting Global Model and 12 Per-Tenant Models for hierarchical comparison...")
    global_contamination = float(train_df["ground_truth_label"].mean())

    # 1. Global Model Pipeline
    global_ct = ColumnTransformer([
        ("sys", StandardScaler(), RAW_SYSTEM_COLS),
        ("biz", RobustScaler(with_centering=False), RAW_BIZ_COLS),
    ])
    global_pipeline = Pipeline([
        ("ct", global_ct),
        ("model", IsolationForest(contamination=global_contamination, random_state=42, n_jobs=-1)),
    ])
    global_pipeline.fit(train_df[RAW_SYSTEM_COLS + RAW_BIZ_COLS])

    # Global drift thresholds
    normal_train = train_df[train_df["ground_truth_label"] == 0]
    global_t60 = float(normal_train["mem_slope_60m"].quantile(0.99))
    global_t15 = float(normal_train["mem_slope_15m"].quantile(0.95))

    # 2. Per-Tenant Specialized Models
    per_tenant_models = {}
    per_tenant_drift_thresh = {}

    for biz_id in train_df["business_id"].unique():
        b_tr = train_df[train_df["business_id"] == biz_id]
        b_contam = float(b_tr["ground_truth_label"].mean())
        b_contam = max(0.01, min(0.20, b_contam))

        b_ct = ColumnTransformer([
            ("sys", StandardScaler(), RAW_SYSTEM_COLS),
            ("biz", RobustScaler(with_centering=False), RAW_BIZ_COLS),
        ])
        b_pipe = Pipeline([
            ("ct", b_ct),
            ("model", IsolationForest(contamination=b_contam, random_state=42, n_jobs=-1)),
        ])
        b_pipe.fit(b_tr[RAW_SYSTEM_COLS + RAW_BIZ_COLS])
        per_tenant_models[biz_id] = b_pipe

        b_norm = b_tr[b_tr["ground_truth_label"] == 0]
        per_tenant_drift_thresh[biz_id] = {
            "t60": float(b_norm["mem_slope_60m"].quantile(0.99)),
            "t15": float(b_norm["mem_slope_15m"].quantile(0.95)),
        }

    # 3. Comprehensive Global vs. Per-Tenant Side-by-Side Evaluation on Test Set
    print("[4/5] Evaluating Global vs. Per-Tenant metrics on test partitions...")
    tenant_comparison = {}
    k_debouncing = 4

    for biz_id, group in test_df.groupby("business_id", sort=False):
        name = biz_names.get(biz_id, biz_id[:8])
        y_true = group["ground_truth_label"].values
        X_test_b = group[RAW_SYSTEM_COLS + RAW_BIZ_COLS]

        # --- Evaluate Global Model on this tenant ---
        g_pt = np.where(global_pipeline.predict(X_test_b) == -1, 1, 0)
        g_sq = ((group["mem_slope_60m"] > global_t60) & (group["mem_slope_15m"] > 0)).astype(int).values
        g_raw = np.maximum(g_pt, g_sq)
        r_sum_g = pd.Series(g_raw).rolling(window=k_debouncing, min_periods=k_debouncing).sum()
        g_deb = (r_sum_g == k_debouncing).astype(int).fillna(0).values
        g_metrics = compute_metrics(y_true, g_deb)

        # --- Evaluate Per-Tenant Model on this tenant ---
        t_pipe = per_tenant_models[biz_id]
        t60_th = per_tenant_drift_thresh[biz_id]["t60"]
        t_pt = np.where(t_pipe.predict(X_test_b) == -1, 1, 0)
        t_sq = ((group["mem_slope_60m"] > t60_th) & (group["mem_slope_15m"] > 0)).astype(int).values
        t_raw = np.maximum(t_pt, t_sq)
        r_sum_t = pd.Series(t_raw).rolling(window=k_debouncing, min_periods=k_debouncing).sum()
        t_deb = (r_sum_t == k_debouncing).astype(int).fillna(0).values
        t_metrics = compute_metrics(y_true, t_deb)

        tenant_comparison[biz_id] = {
            "business_name": name,
            "test_rows": len(group),
            "anomalies": int(y_true.sum()),
            "global_model": g_metrics,
            "per_tenant_model": t_metrics,
            "fpr_difference": round(t_metrics["false_positive_rate"] - g_metrics["false_positive_rate"], 4),
            "precision_difference": round(t_metrics["precision"] - g_metrics["precision"], 4),
            "recall_difference": round(t_metrics["recall"] - g_metrics["recall"], 4),
        }

    # Aggregate Global vs. Per-Tenant overall metrics
    g_all_preds = []
    t_all_preds = []
    all_y = []

    for biz_id, group in test_df.groupby("business_id", sort=False):
        y_true = group["ground_truth_label"].values
        X_test_b = group[RAW_SYSTEM_COLS + RAW_BIZ_COLS]
        all_y.extend(y_true)

        # Global
        g_pt = np.where(global_pipeline.predict(X_test_b) == -1, 1, 0)
        g_sq = ((group["mem_slope_60m"] > global_t60) & (group["mem_slope_15m"] > 0)).astype(int).values
        g_deb = (pd.Series(np.maximum(g_pt, g_sq)).rolling(window=k_debouncing, min_periods=k_debouncing).sum() == k_debouncing).astype(int).fillna(0).values
        g_all_preds.extend(g_deb)

        # Tenant
        t_pipe = per_tenant_models[biz_id]
        t60_th = per_tenant_drift_thresh[biz_id]["t60"]
        t_pt = np.where(t_pipe.predict(X_test_b) == -1, 1, 0)
        t_sq = ((group["mem_slope_60m"] > t60_th) & (group["mem_slope_15m"] > 0)).astype(int).values
        t_deb = (pd.Series(np.maximum(t_pt, t_sq)).rolling(window=k_debouncing, min_periods=k_debouncing).sum() == k_debouncing).astype(int).fillna(0).values
        t_all_preds.extend(t_deb)

    total_y = np.array(all_y)
    agg_global = compute_metrics(total_y, np.array(g_all_preds))
    agg_tenant = compute_metrics(total_y, np.array(t_all_preds))

    results = {
        "phase": "Phase 6 - Hybrid Hierarchical Model Architecture",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_summary": {
            "cold_start_tier": "Global Multi-Tenant IsolationForest Pipeline (< 30 days telemetry)",
            "mature_tier": "Per-Tenant Calibrated IsolationForest Pipelines (> 30 days telemetry)",
            "alert_debouncing": "k=4 consecutive anomalous events required before dispatching alerts",
        },
        "overall_comparison": {
            "global_architecture": agg_global,
            "per_tenant_architecture": agg_tenant,
        },
        "per_tenant_comparison_table": tenant_comparison,
    }

    # Production Hierarchical Bundle
    production_bundle = {
        "architecture": "HybridHierarchicalAnomalyDetector",
        "global_model": global_pipeline,
        "global_drift_thresholds": {"thresh_slope_60m": global_t60, "thresh_slope_15m": global_t15},
        "per_tenant_models": per_tenant_models,
        "per_tenant_drift_thresholds": per_tenant_drift_thresh,
        "operational_config": {
            "cold_start_threshold_days": 30,
            "debouncing_method": "k_consecutive",
            "k": 4,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return results, production_bundle


def save_and_verify(results: Dict[str, Any], production_bundle: Dict[str, Any]):
    print("[5/5] Persisting Phase 6 production bundle & evaluation artifacts...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"      Saved eval JSON: {EVAL_OUTPUT_FILE} ({EVAL_OUTPUT_FILE.stat().st_size / 1024:.2f} KB)")

    joblib.dump(production_bundle, MODEL_OUTPUT_FILE)
    print(f"      Saved production joblib bundle: {MODEL_OUTPUT_FILE} ({MODEL_OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB)")

    # Load-back verification
    loaded = joblib.load(MODEL_OUTPUT_FILE)
    assert "global_model" in loaded
    assert "per_tenant_models" in loaded
    assert len(loaded["per_tenant_models"]) == 12
    print(f"      Integrity verified: 1 Global + 12 Per-Tenant Models successfully loaded from production bundle.")


def print_summary(results: Dict[str, Any]):
    print("\n" + "=" * 90)
    print("PHASE 6: GLOBAL vs. PER-TENANT METRICS & FPR COMPARISON TABLE (ALL 12 TENANTS)")
    print("=" * 90)
    print(f"{'Tenant Name':<26} | {'Global Prec':<11} {'Global FPR':<11} | {'Tenant Prec':<11} {'Tenant FPR':<11} | {'FPR Delta':<9}")
    print("-" * 90)
    for b_id, d in results["per_tenant_comparison_table"].items():
        name = d["business_name"]
        gm = d["global_model"]
        tm = d["per_tenant_model"]
        delta = d["fpr_difference"]
        sign = "+" if delta > 0 else ""
        print(f"{name:<26} | {gm['precision']:<11.2%} {gm['false_positive_rate']:<11.4%} | {tm['precision']:<11.2%} {tm['false_positive_rate']:<11.4%} | {sign}{delta:<8.4%}")

    print("\n" + "=" * 90)
    print("OVERALL ARCHITECTURAL COMPARISON")
    print("=" * 90)
    g = results["overall_comparison"]["global_architecture"]
    t = results["overall_comparison"]["per_tenant_architecture"]
    print(f"{'Architecture':<35} | {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'FPR':<10}")
    print("-" * 90)
    print(f"{'Global Multi-Tenant Model':<35} | {g['precision']:<10.2%} {g['recall']:<10.2%} {g['f1_score']:<10.4f} {g['false_positive_rate']:<10.4%}")
    print(f"{'Tenant-Specialized Models':<35} | {t['precision']:<10.2%} {t['recall']:<10.2%} {t['f1_score']:<10.4f} {t['false_positive_rate']:<10.4%}")


def main():
    df = asyncio.run(load_telemetry())
    df_full, biz_names = load_and_label_ground_truth(df)
    train_df, test_df = time_based_split(df_full, train_ratio=0.70)
    results, production_bundle = run_phase6_hierarchical_comparison(train_df, test_df, biz_names)
    save_and_verify(results, production_bundle)
    print_summary(results)


if __name__ == "__main__":
    main()
