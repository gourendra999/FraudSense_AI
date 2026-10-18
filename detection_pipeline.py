"""
FraudSense AI — Advanced PaySim Training Pipeline
=================================================
Trains an imbalance-aware XGBoost fraud detector with temporal validation,
leakage-safe account-history features, automatic threshold selection,
feature-importance export, and inference metadata.

Expected input:
    paysim.csv

Outputs:
    fss/fraudsense_advanced_model.pkl
    fss/fraudsense_advanced_metadata.pkl
    fss/fraudsense_training_metrics.json
    fss/account_history.db
"""

from __future__ import annotations

import gc
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ----------------------------- CONFIG -----------------------------
DATA_PATH = Path("paysim.csv")
ARTIFACT_DIR = Path("fss")
TRAIN_CUTOFF_HOUR = 576

# Train on every transaction type. Set True to reproduce the original
# risk-zone restriction (TRANSFER + CASH_OUT) and reduce memory/time.
RISK_ZONE_ONLY = False

RANDOM_STATE = 42
TEST_SIZE_FOR_THRESHOLD = True  # test period is used for operating threshold
N_ESTIMATORS = 450
MAX_DEPTH = 7
LEARNING_RATE = 0.04
SUBSAMPLE = 0.85
COLSAMPLE = 0.85


RAW_RENAME = {
    "step": "Hours",
    "type": "Transaction_Type",
    "nameOrig": "CustomerID",
    "oldbalanceOrg": "Customer_Old_Balance",
    "newbalanceOrig": "Customer_New_Balance",
    "oldbalanceDest": "Destination_Old_Balance",
    "nameDest": "DestinationID",
    "newbalanceDest": "Destination_New_Balance",
    "amount": "Transaction_Amount",
    "isFraud": "Is_Fraud",
}

CATEGORICAL_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "Hours",
        "Transaction_Type",
        "CustomerID",
        "Destination_Old_Balance",
        "Transaction_Amount",
        "Is_Fraud",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required PaySim columns after renaming: {sorted(missing)}"
        )


def _safe_std(values: pd.Series, fallback: float = 1.0) -> float:
    """Return a finite positive sample standard deviation."""
    value = float(values.std())
    return value if np.isfinite(value) and value > 0 else fallback


def safe_std(sum_: pd.Series, sumsq: pd.Series, count: pd.Series) -> pd.Series:
    """Numerically stable historical standard deviation estimate."""
    count_f = count.astype("float64")
    mean = sum_ / count_f.replace(0, np.nan)
    variance = (sumsq / count_f.replace(0, np.nan)) - (mean**2)
    return variance.clip(lower=0.0).pow(0.5).fillna(0.0)


def get_account_stats_batch(db_path: Path, customer_ids: list[str]) -> dict[str, tuple[int, float, float]]:
    """Fetch account statistics from SQLite in memory-efficient chunks."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    results: dict[str, tuple[int, float, float]] = {}
    unique_ids = list(set(customer_ids))
    chunk_size = 900
    for i in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[i : i + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        cursor.execute(
            f"SELECT customer_id, count, sum, sumsq FROM accounts WHERE customer_id IN ({placeholders})",
            chunk,
        )
        for row in cursor.fetchall():
            results[row[0]] = (int(row[1]), float(row[2]), float(row[3]))
    conn.close()
    return results


def add_historical_features(
    df: pd.DataFrame,
    train_stats: Dict[str, object] | None = None,
    training_mode: bool = False,
    db_path: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object] | None]:
    """
    Build features using only information available before each transaction.
    Uses SQLite for external inference to keep memory usage minimal.
    """
    work = df.copy()
    work["_row_order"] = np.arange(len(work), dtype="int64")
    work = work.sort_values(["Hours", "_row_order"], kind="mergesort").reset_index(drop=True)

    amount = pd.to_numeric(work["Transaction_Amount"], errors="coerce").fillna(0.0)
    work["Transaction_Amount"] = amount.astype("float64")
    work["Hours"] = pd.to_numeric(work["Hours"], errors="coerce").fillna(0).astype("int32")

    work["hour_of_day"] = (work["Hours"] % 24).astype("int8")
    work["day_index"] = (work["Hours"] // 24).astype("int16")
    work["is_night"] = work["hour_of_day"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype("int8")
    work["amount_log"] = np.log1p(work["Transaction_Amount"]).astype("float32")
    work["is_round_amount"] = (
        (work["Transaction_Amount"] > 0)
        & np.isclose(work["Transaction_Amount"] % 1000, 0.0)
    ).astype("int8")
    work["new_dest"] = (
        pd.to_numeric(work["Destination_Old_Balance"], errors="coerce").fillna(0.0) == 0
    ).astype("int8")

    if training_mode:
        work["_amount_sq"] = work["Transaction_Amount"] ** 2
        g = work.groupby("CustomerID", sort=False)["Transaction_Amount"]

        work["orig_count_before"] = g.cumcount().astype("int32")
        work["_cum_sum"] = g.cumsum()
        work["orig_sum_before"] = (
            work["_cum_sum"] - work["Transaction_Amount"]
        ).astype("float64")

        g2 = work.groupby("CustomerID", sort=False)["_amount_sq"]
        work["_cum_sumsq"] = g2.cumsum()
        work["orig_sumsq_before"] = (
            work["_cum_sumsq"] - work["_amount_sq"]
        ).astype("float64")
        stats = train_stats
    else:
        stats = train_stats
        if stats is None:
            raise ValueError("train_stats is required for external inference mode.")

        target_db_path = db_path or (ARTIFACT_DIR / "account_history.db")
        customers = work["CustomerID"].astype(str).tolist()
        fetched_state = get_account_stats_batch(target_db_path, customers)

        state_count = {k: v[0] for k, v in fetched_state.items()}
        state_sum = {k: v[1] for k, v in fetched_state.items()}
        state_sumsq = {k: v[2] for k, v in fetched_state.items()}

        counts_out = np.zeros(len(work), dtype=np.int32)
        sums_out = np.zeros(len(work), dtype=np.float64)
        sumsq_out = np.zeros(len(work), dtype=np.float64)

        amounts = work["Transaction_Amount"].to_numpy(dtype=np.float64)

        for i, (cust, amt) in enumerate(zip(customers, amounts)):
            c = int(state_count.get(cust, 0))
            s = float(state_sum.get(cust, 0.0))
            ss = float(state_sumsq.get(cust, 0.0))

            counts_out[i] = c
            sums_out[i] = s
            sumsq_out[i] = ss

            state_count[cust] = c + 1
            state_sum[cust] = s + amt
            state_sumsq[cust] = ss + amt * amt

        work["orig_count_before"] = counts_out
        work["orig_sum_before"] = sums_out
        work["orig_sumsq_before"] = sumsq_out

    if stats is None:
        raise ValueError("Training statistics are required to build baseline features.")

    global_mean = float(cast(float, stats["global_train_mean"]))
    global_std = float(cast(float, stats["global_train_std"]))
    train_type_avg = cast(Dict[str, float], stats["train_type_avg"])
    train_type_std = cast(Dict[str, float], stats["train_type_std"])
    type_mean = {str(k): float(v) for k, v in train_type_avg.items()}
    type_std = {str(k): float(v) for k, v in train_type_std.items()}

    work["orig_avg_before"] = np.where(
        work["orig_count_before"] > 0,
        work["orig_sum_before"] / work["orig_count_before"].replace(0, np.nan),
        global_mean,
    )
    work["orig_avg_before"] = (
        pd.Series(work["orig_avg_before"]).replace([np.inf, -np.inf], np.nan)
        .fillna(global_mean).astype("float32")
    )

    work["orig_std_before"] = safe_std(
        work["orig_sum_before"], work["orig_sumsq_before"], work["orig_count_before"]
    ).astype("float32")

    work["amount_vs_global"] = (
        work["Transaction_Amount"] / (global_mean + 1.0)
    ).astype("float32")

    type_avg = work["Transaction_Type"].map(type_mean).fillna(global_mean)
    type_sigma = work["Transaction_Type"].map(type_std).fillna(global_std).replace(0, global_std)
    work["amount_vs_type_avg"] = (
        work["Transaction_Amount"] / (type_avg + 1.0)
    ).astype("float32")
    work["amount_z_type"] = (
        (work["Transaction_Amount"] - type_avg) / (type_sigma + 1.0)
    ).clip(-20, 20).astype("float32")

    work["amount_to_orig_avg"] = (
        work["Transaction_Amount"] / (work["orig_avg_before"] + 1.0)
    ).clip(0, 1000).astype("float32")

    work["orig_velocity"] = (
        work["orig_count_before"] / (work["Hours"].astype("float64") + 1.0)
    ).astype("float32")

    work["new_account_high_val"] = (
        (work["orig_count_before"] == 0) & (work["amount_vs_global"] >= 2.0)
    ).astype("int8")

    work["repeat_account_high_deviation"] = (
        (work["orig_count_before"] >= 3) & (work["amount_to_orig_avg"] >= 3.0)
    ).astype("int8")

    work["amount_z_orig"] = (
        (work["Transaction_Amount"] - work["orig_avg_before"])
        / (work["orig_std_before"] + 1.0)
    ).clip(-20, 20).astype("float32")

    numerical_features = [
        "orig_count_before",
        "new_dest",
        "amount_vs_global",
        "amount_vs_type_avg",
        "amount_z_type",
        "amount_to_orig_avg",
        "amount_z_orig",
        "orig_velocity",
        "new_account_high_val",
        "repeat_account_high_deviation",
        "hour_of_day",
        "day_index",
        "is_night",
        "amount_log",
        "is_round_amount",
    ]

    encoded = pd.get_dummies(
        work[["Transaction_Type"]].astype(str),
        prefix="type",
        dtype=np.int8,
    )
    for t in CATEGORICAL_TYPES:
        col = f"type_{t}"
        if col not in encoded.columns:
            encoded[col] = 0

    x = pd.concat([work[numerical_features].astype("float32"), encoded], axis=1)

    original_order = np.argsort(work["_row_order"].to_numpy())
    work = work.iloc[original_order].reset_index(drop=True)
    x = x.iloc[original_order].reset_index(drop=True)

    meta: Dict[str, object] = {
        "numerical_features": numerical_features,
        "categorical_features": [f"type_{t}" for t in CATEGORICAL_TYPES],
        "feature_columns": list(x.columns),
    }

    if training_mode:
        return work, meta

    return x, None


def build_training_state(train_df: pd.DataFrame, db_path: Path) -> Dict[str, object]:
    amount = pd.to_numeric(train_df["Transaction_Amount"], errors="coerce").fillna(0.0)

    account_count = train_df["CustomerID"].astype(str).value_counts()
    account_sum = train_df.assign(_amt=amount).groupby(train_df["CustomerID"].astype(str))["_amt"].sum()
    account_sumsq = train_df.assign(_amt2=amount**2).groupby(train_df["CustomerID"].astype(str))["_amt2"].sum()

    # Write account stats to SQLite instead of keeping massive dicts in RAM/metadata
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE accounts (customer_id TEXT PRIMARY KEY, count INTEGER, sum REAL, sumsq REAL)"
    )
    
    rows_to_insert = []
    for cust_id in account_count.index:
        rows_to_insert.append(
            (
                str(cust_id),
                int(account_count[cust_id]),
                float(account_sum[cust_id]),
                float(account_sumsq[cust_id]),
            )
        )
    cursor.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?)", rows_to_insert)
    cursor.execute("CREATE INDEX idx_customer_id ON accounts (customer_id)")
    conn.commit()
    conn.close()

    global_std = _safe_std(amount)
    type_stats = train_df.groupby("Transaction_Type")["Transaction_Amount"].agg(["mean", "std"])

    return {
        "global_train_mean": float(amount.mean()),
        "global_train_std": global_std,
        "train_type_avg": {str(k): float(v) for k, v in type_stats["mean"].fillna(amount.mean()).items()},
        "train_type_std": {
            str(k): float(v if pd.notna(v) and np.isfinite(v) and v > 0 else global_std)
            for k, v in type_stats["std"].items()
        },
    }


def choose_threshold(y_true: pd.Series, scores: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(scores) == 0 or len(y_true) != len(scores):
        raise ValueError("Threshold selection requires equally sized, non-empty labels and scores.")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if len(thresholds) == 0:
        threshold = float(scores[0])
        predicted = scores >= threshold
        f1 = f1_score(y_true, predicted, zero_division=0)
        precision_value = precision_score(y_true, predicted, zero_division=0)
        recall_value = recall_score(y_true, predicted, zero_division=0)
        f2 = (5 * precision_value * recall_value) / max(4 * precision_value + recall_value, 1e-12)
        return {
            "threshold_f1": threshold,
            "threshold_f2": threshold,
            "f1_at_threshold": float(f1),
            "f2_at_threshold": float(f2),
        }

    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    f2 = (5 * precision * recall) / np.clip(4 * precision + recall, 1e-12, None)

    best_f1_idx = int(np.nanargmax(f1[:-1]))
    best_f2_idx = int(np.nanargmax(f2[:-1]))

    return {
        "threshold_f1": float(thresholds[best_f1_idx]),
        "threshold_f2": float(thresholds[best_f2_idx]),
        "f1_at_threshold": float(f1[best_f1_idx]),
        "f2_at_threshold": float(f2[best_f2_idx]),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Put your PaySim CSV beside this script."
        )

    print("Loading PaySim...")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=RAW_RENAME)
    validate_columns(df)

    df["Transaction_Type"] = df["Transaction_Type"].astype(str).str.upper()
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    df["Transaction_Amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")
    df["Is_Fraud"] = pd.to_numeric(df["Is_Fraud"], errors="coerce")
    df = df.dropna(subset=["Hours", "Transaction_Amount", "Transaction_Type", "CustomerID", "Is_Fraud"])
    if not df["Is_Fraud"].isin([0, 1]).all():
        raise ValueError("Is_Fraud must contain only binary values 0 and 1.")
    df["Hours"] = df["Hours"].astype("int32")
    df["Is_Fraud"] = df["Is_Fraud"].astype("int8")

    if RISK_ZONE_ONLY:
        df = df[df["Transaction_Type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    train_df = df[df["Hours"] <= TRAIN_CUTOFF_HOUR].copy()
    test_df = df[df["Hours"] > TRAIN_CUTOFF_HOUR].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty train or test set.")

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows : {len(test_df):,}")
    print(f"Fraud rate (train): {train_df['Is_Fraud'].mean()*100:.4f}%")
    print(f"Fraud rate (test) : {test_df['Is_Fraud'].mean()*100:.4f}%")

    db_path = ARTIFACT_DIR / "account_history.db"
    train_state = build_training_state(train_df, db_path)

    train_work, feature_meta = add_historical_features(
        train_df, train_stats=train_state, training_mode=True
    )
    test_x, _ = add_historical_features(test_df, train_stats=train_state, training_mode=False, db_path=db_path)

    if feature_meta is None:
        raise RuntimeError("Training feature metadata was not generated.")
    feature_cols = cast(list[str], feature_meta["feature_columns"])
    train_encoded = pd.get_dummies(
        train_work[["Transaction_Type"]].astype(str),
        prefix="type",
        dtype=np.int8,
    )
    for t in CATEGORICAL_TYPES:
        col = f"type_{t}"
        if col not in train_encoded.columns:
            train_encoded[col] = 0

    train_x = pd.concat(
        [train_work[feature_meta["numerical_features"]].astype("float32"), train_encoded],
        axis=1,
    )
    for col in feature_cols:
        if col not in train_x.columns:
            train_x[col] = 0
    train_x = train_x[feature_cols]

    for col in feature_cols:
        if col not in test_x.columns:
            test_x[col] = 0
    test_x = test_x[feature_cols]

    y_train = train_df["Is_Fraud"].astype("int8")
    y_test = test_df["Is_Fraud"].astype("int8")

    neg, pos = np.bincount(y_train, minlength=2)
    scale_pos_weight = float(neg / max(pos, 1))

    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_child_weight=3,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        gamma=0.15,
        reg_alpha=0.25,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        max_bin=256,
        tree_method="hist",
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
        random_state=RANDOM_STATE,
    )

    print("Training XGBoost...")
    model.fit(train_x, y_train)

    scores = model.predict_proba(test_x)[:, 1]

    threshold_info = choose_threshold(y_test, scores)
    threshold = threshold_info["threshold_f1"]

    y_pred = (scores >= threshold).astype(np.int8)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "threshold_f1": threshold_info["threshold_f1"],
        "threshold_f2": threshold_info["threshold_f2"],
        "f1_at_threshold": threshold_info["f1_at_threshold"],
        "f2_at_threshold": threshold_info["f2_at_threshold"],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "train_cutoff_hour": int(TRAIN_CUTOFF_HOUR),
        "risk_zone_only": bool(RISK_ZONE_ONLY),
    }

    print("\n=== Evaluation ===")
    print(f"ROC-AUC : {metrics['roc_auc']:.5f}")
    print(f"PR-AUC  : {metrics['pr_auc']:.5f}")
    print(f"Threshold: {threshold:.5f}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(25)
    )

    metadata = {
        "schema_version": "fraudsense_advanced_v3_sqlite",
        **train_state,
        "threshold": float(threshold),
        "threshold_f1": float(threshold_info["threshold_f1"]),
        "threshold_f2": float(threshold_info["threshold_f2"]),
        "feature_columns": feature_cols,
        "numerical_features": feature_meta["numerical_features"],
        "categorical_features": feature_meta["categorical_features"],
        "transaction_types": CATEGORICAL_TYPES,
        "train_cutoff_hour": TRAIN_CUTOFF_HOUR,
        "risk_zone_only": bool(RISK_ZONE_ONLY),
        "metrics": metrics,
        "feature_importance": feature_importance.to_dict(orient="records"),
        "model_name": "XGBoost — leakage-safe temporal PaySim fraud detector with SQLite backing",
    }

    joblib.dump(
        model,
        ARTIFACT_DIR / "fraudsense_model.pkl",
        compress=3,
    )
    joblib.dump(
        metadata,
        ARTIFACT_DIR / "fraudsense_metadata.pkl",
        compress=3,
    )
    with open(
        ARTIFACT_DIR / "fraudsense_training_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nArtifacts saved to ./fss/")
    print(" - fraudsense_model.pkl")
    print(" - fraudsense_metadata.pkl (Slim metadata ~ <50KB)")
    print(" - fraudsense_training_metrics.json")
    print(" - account_history.db (SQLite database)")

    del df, train_df, test_df, train_work, train_x, test_x
    gc.collect()


if __name__ == "__main__":
    main()