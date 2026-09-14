#!/usr/bin/env python3
"""
AICTO Anomaly Detection - Phase 1: Feature Separation (System vs. Business Metrics)
==================================================================================
Investigates the impact of isolating zero-inflated domain features (orders_count, revenue_amount)
from core infrastructure metrics (response_time_ms, cpu_usage_pct, memory_usage_pct, queue_depth).

Options evaluated:
- Baseline: StandardScaler on all 6 features globally.
- Option (a): Tenant-conditional scaling for business metrics (scaled on active retail tenants, 0.0 for SaaS).
- Option (b): ColumnTransformer with StandardScaler for system metrics and RobustScaler(with_centering=False) for business metrics.
- Option (c): Pure infrastructure metrics (orders_count and revenue_amount excluded from anomaly feature space).

Outputs:
- scripts/output/phase1_feature_separation_eval.json
- scripts/output/anomaly_model_phase1.joblib
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
    roc_auc_score,
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


SYSTEM_COLS = ["response_time_ms", "cpu_usage_pct", "memory_usage_pct", "queue_depth"]
BIZ_COLS = ["orders_count", "revenue_amount"]
ALL_FEATURES = SYSTEM_COLS + BIZ_COLS

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"
GROUND_TRUTH_FILE = OUTPUT_DIR / "anomaly_ground_truth.json"
MODEL_OUTPUT_FILE = OUTPUT_DIR / "anomaly_model_phase1.joblib"
EVAL_OUTPUT_FILE = OUTPUT_DIR / "phase1_feature_separation_eval.json"


async def load_telemetry() -> pd.DataFrame:
    print("[1/6] Ingesting telemetry events from PostgreSQL...")
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


def load_and_label_ground_truth(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict[str, Any]]]:
    print("[2/6] Loading ground truth and assigning labels...")
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["business_id"] = df["business_id"].astype(str)
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)

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
            "severity": ep["severity"],
        })
        biz_names[ep["business_id"]] = ep.get("business_name", ep["business_id"][:8])

    df["ground_truth_label"] = 0
    df["anomaly_type"] = None
    df["anomaly_severity"] = None

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
                df.loc[m_idx, "anomaly_severity"] = ep["severity"]

    print(f"      Labeled {df['ground_truth_label'].sum():,} anomaly rows ({df['ground_truth_label'].mean():.4%}).")
    return df, biz_names, episodes


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[3/6] Performing per-business time-based split ({train_ratio:.0%} train / {1 - train_ratio:.0%} test)...")
    train_dfs = []
    test_dfs = []
    for biz_id, group in df.groupby("business_id"):
        group = group.sort_values("timestamp")
        min_ts = group["timestamp"].min()
        max_ts = group["timestamp"].max()
        cutoff = min_ts + (max_ts - min_ts) * train_ratio
        train_dfs.append(group[group["timestamp"] < cutoff])
        test_dfs.append(group[group["timestamp"] >= cutoff])

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    print(f"      Train: {len(train_df):,} rows | Test: {len(test_df):,} rows.")
    return train_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def compute_per_tenant_breakdown(
    test_df: pd.DataFrame, pred_col: str, biz_names: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    tenant_results = {}
    for biz_id, group in test_df.groupby("business_id"):
        name = biz_names.get(biz_id, biz_id[:8])
        y = group["ground_truth_label"].values
        p = group[pred_col].values
        p_val = float(precision_score(y, p, zero_division=0))
        r_val = float(recall_score(y, p, zero_division=0))
        f1_val = float(f1_score(y, p, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tenant_results[biz_id] = {
            "business_name": name,
            "test_rows": len(group),
            "anomalies": int(y.sum()),
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1_score": round(f1_val, 4),
            "false_positive_rate": round(fpr_val, 4),
            "false_positives": int(fp),
            "true_positives": int(tp),
        }
    return tenant_results


def compute_type_recall(test_df: pd.DataFrame, pred_col: str) -> Dict[str, Dict[str, Any]]:
    type_metrics = {}
    for atype in ["latency_spike", "cpu_saturation", "queue_explosion", "memory_leak"]:
        sub = test_df[test_df["anomaly_type"] == atype]
        if len(sub) > 0:
            rec = float(recall_score(sub["ground_truth_label"], sub[pred_col], zero_division=0))
            type_metrics[atype] = {
                "recall": round(rec, 4),
                "detected": int(sub[pred_col].sum()),
                "total": len(sub),
            }
        else:
            type_metrics[atype] = {"recall": 0.0, "detected": 0, "total": 0}
    return type_metrics


def run_phase1_experiments(train_df: pd.DataFrame, test_df: pd.DataFrame, biz_names: Dict[str, str]):
    print("[4/6] Training & evaluating Phase 1 feature separation candidates...")
    contamination = float(train_df["ground_truth_label"].mean())
    y_test = test_df["ground_truth_label"].values

    # -----------------------------------------------------------------------
    # Candidate 0: Baseline (StandardScaler on all 6 features)
    # -----------------------------------------------------------------------
    print("      Fitting Candidate 0: Baseline...")
    pipe_base = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)),
    ])
    pipe_base.fit(train_df[ALL_FEATURES])
    preds_base = np.where(pipe_base.predict(test_df[ALL_FEATURES]) == -1, 1, 0)
    scores_base = -pipe_base.decision_function(test_df[ALL_FEATURES])
    test_df["pred_base"] = preds_base
    m_base = compute_metrics(y_test, preds_base, scores_base)
    t_base = compute_per_tenant_breakdown(test_df, "pred_base", biz_names)
    rec_base = compute_type_recall(test_df, "pred_base")

    # -----------------------------------------------------------------------
    # Candidate A: Option (a) - Tenant-Conditional Scaling
    # -----------------------------------------------------------------------
    print("      Fitting Candidate A: Tenant-Conditional Scaling...")
    train_retail_mask = train_df["orders_count"] > 0
    sc_sys_a = StandardScaler()
    sc_biz_a = StandardScaler()

    X_tr_sys_a = sc_sys_a.fit_transform(train_df[SYSTEM_COLS])
    X_te_sys_a = sc_sys_a.transform(test_df[SYSTEM_COLS])

    sc_biz_a.fit(train_df.loc[train_retail_mask, BIZ_COLS])

    X_tr_biz_a = np.zeros((len(train_df), len(BIZ_COLS)))
    X_tr_biz_a[train_retail_mask] = sc_biz_a.transform(train_df.loc[train_retail_mask, BIZ_COLS])

    retail_biz_ids = set(train_df.loc[train_retail_mask, "business_id"].unique())
    test_retail_mask = test_df["business_id"].isin(retail_biz_ids)

    X_te_biz_a = np.zeros((len(test_df), len(BIZ_COLS)))
    if test_retail_mask.any():
        X_te_biz_a[test_retail_mask] = sc_biz_a.transform(test_df.loc[test_retail_mask, BIZ_COLS])

    X_train_a = np.hstack([X_tr_sys_a, X_tr_biz_a])
    X_test_a = np.hstack([X_te_sys_a, X_te_biz_a])

    iso_a = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso_a.fit(X_train_a)
    preds_a = np.where(iso_a.predict(X_test_a) == -1, 1, 0)
    scores_a = -iso_a.decision_function(X_test_a)
    test_df["pred_a"] = preds_a
    m_a = compute_metrics(y_test, preds_a, scores_a)
    t_a = compute_per_tenant_breakdown(test_df, "pred_a", biz_names)
    rec_a = compute_type_recall(test_df, "pred_a")

    # -----------------------------------------------------------------------
    # Candidate B: Option (b) - ColumnTransformer (StandardScaler + RobustScaler)
    # -----------------------------------------------------------------------
    print("      Fitting Candidate B: ColumnTransformer (Option b)...")
    col_transformer = ColumnTransformer(
        transformers=[
            ("system_scaler", StandardScaler(), SYSTEM_COLS),
            ("business_scaler", RobustScaler(with_centering=False), BIZ_COLS),
        ]
    )
    pipe_b = Pipeline([
        ("features", col_transformer),
        ("model", IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)),
    ])
    pipe_b.fit(train_df[ALL_FEATURES])
    preds_b = np.where(pipe_b.predict(test_df[ALL_FEATURES]) == -1, 1, 0)
    scores_b = -pipe_b.decision_function(test_df[ALL_FEATURES])
    test_df["pred_b"] = preds_b
    m_b = compute_metrics(y_test, preds_b, scores_b)
    t_b = compute_per_tenant_breakdown(test_df, "pred_b", biz_names)
    rec_b = compute_type_recall(test_df, "pred_b")

    # -----------------------------------------------------------------------
    # Candidate C: Option (c) - System Metrics Only (Drop Biz Cols)
    # -----------------------------------------------------------------------
    print("      Fitting Candidate C: System Metrics Only (Option c)...")
    pipe_c = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)),
    ])
    pipe_c.fit(train_df[SYSTEM_COLS])
    preds_c = np.where(pipe_c.predict(test_df[SYSTEM_COLS]) == -1, 1, 0)
    scores_c = -pipe_c.decision_function(test_df[SYSTEM_COLS])
    test_df["pred_c"] = preds_c
    m_c = compute_metrics(y_test, preds_c, scores_c)
    t_c = compute_per_tenant_breakdown(test_df, "pred_c", biz_names)
    rec_c = compute_type_recall(test_df, "pred_c")

    results = {
        "phase": "Phase 1 - Feature Separation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "description": "StandardScaler on all 6 metrics (system + business)",
            "global_metrics": m_base,
            "type_recall": rec_base,
            "tenant_metrics": t_base,
        },
        "option_a_conditional_scaling": {
            "description": "Tenant-conditional scaling for business metrics (scaled on active retail tenants, 0 for SaaS)",
            "global_metrics": m_a,
            "type_recall": rec_a,
            "tenant_metrics": t_a,
        },
        "option_b_column_transformer": {
            "description": "ColumnTransformer: StandardScaler(system) + RobustScaler(business, with_centering=False)",
            "global_metrics": m_b,
            "type_recall": rec_b,
            "tenant_metrics": t_b,
        },
        "option_c_system_only": {
            "description": "System metrics only (response_time_ms, cpu_usage_pct, memory_usage_pct, queue_depth)",
            "global_metrics": m_c,
            "type_recall": rec_c,
            "tenant_metrics": t_c,
        },
    }

    # Best production model choice: Option B (robust pipeline) or Option C (pure infrastructure)
    # We serialize Option B as the main production candidate supporting all schema columns cleanly
    selected_pipeline = pipe_b

    return results, selected_pipeline


def save_and_verify(results: Dict[str, Any], pipeline: Pipeline):
    print("[5/6] Persisting evaluation JSON and trained model artifact...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"      Saved eval JSON: {EVAL_OUTPUT_FILE} ({EVAL_OUTPUT_FILE.stat().st_size / 1024:.2f} KB)")

    joblib.dump(pipeline, MODEL_OUTPUT_FILE)
    print(f"      Saved model joblib: {MODEL_OUTPUT_FILE} ({MODEL_OUTPUT_FILE.stat().st_size / 1024:.2f} KB)")

    print("[6/6] Verifying artifact integrity via load-back test...")
    loaded_pipe = joblib.load(MODEL_OUTPUT_FILE)
    sample_df = pd.DataFrame([{
        "response_time_ms": 120.0,
        "cpu_usage_pct": 45.0,
        "memory_usage_pct": 50.0,
        "queue_depth": 2,
        "orders_count": 0,
        "revenue_amount": 0.0,
    }])
    pred = loaded_pipe.predict(sample_df[ALL_FEATURES])
    print(f"      Load-back verification succeeded! Sample raw pred: {pred[0]}")


def print_console_summary(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("PHASE 1 EVALUATION SUMMARY - COMPARATIVE RESULTS")
    print("=" * 80)
    print(f"{'Option':<35} | {'Prec':<7} {'Recall':<7} {'F1':<7} {'ROC-AUC':<8} {'FPR':<7}")
    print("-" * 80)
    for key, name in [
        ("baseline", "Baseline (Original 6-feat)"),
        ("option_a_conditional_scaling", "Option (a) Conditional Scaling"),
        ("option_b_column_transformer", "Option (b) ColumnTransformer"),
        ("option_c_system_only", "Option (c) System Metrics Only"),
    ]:
        m = results[key]["global_metrics"]
        print(f"{name:<35} | {m['precision']:<7.4f} {m['recall']:<7.4f} {m['f1_score']:<7.4f} {m['roc_auc']:<8.4f} {m['false_positive_rate']:<7.4f}")

    print("\n" + "=" * 80)
    print("PER-TENANT COMPARISON: BASELINE vs. OPTION B vs. OPTION A (FPR Focus)")
    print("=" * 80)
    t_base = results["baseline"]["tenant_metrics"]
    t_b = results["option_b_column_transformer"]["tenant_metrics"]
    t_a = results["option_a_conditional_scaling"]["tenant_metrics"]

    print(f"{'Tenant Name':<26} | {'Base Prec':<9} {'Base FPR':<9} | {'OptB Prec':<9} {'OptB FPR':<9} | {'OptA Prec':<9} {'OptA FPR':<9}")
    print("-" * 95)
    for b_id, d in t_base.items():
        name = d["business_name"]
        print(
            f"{name:<26} | "
            f"{d['precision']:<9.3f} {d['false_positive_rate']:<9.3f} | "
            f"{t_b[b_id]['precision']:<9.3f} {t_b[b_id]['false_positive_rate']:<9.3f} | "
            f"{t_a[b_id]['precision']:<9.3f} {t_a[b_id]['false_positive_rate']:<9.3f}"
        )


def main():
    df = asyncio.run(load_telemetry())
    df, biz_names, episodes = load_and_label_ground_truth(df)
    train_df, test_df = time_based_split(df, train_ratio=0.70)
    results, selected_pipeline = run_phase1_experiments(train_df, test_df, biz_names)
    save_and_verify(results, selected_pipeline)
    print_console_summary(results)


if __name__ == "__main__":
    main()
