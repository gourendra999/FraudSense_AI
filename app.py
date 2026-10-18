"""
FraudSense AI — Advanced Interactive Streamlit Application
===========================================================
Run:
    streamlit run app.py

Expected model artifacts in ./fss/:
    fraudsense_model.pkl
    fraudsense_metadata.pkl
    account_history.db
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
print("DEBUG 1: FraudSense AI app.py started", flush=True)

# Import the exact feature builder used during training.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from detection_pipeline import (  # noqa: E402
    CATEGORICAL_TYPES,
    RAW_RENAME,
    add_historical_features,
)

APP_TITLE = "FraudSense AI"
MODEL_PATH = ROOT / "fss" / "fraudsense_model.pkl"
META_PATH = ROOT / "fss" / "fraudsense_metadata.pkl"
DB_PATH = ROOT / "fss" / "account_history.db"

st.set_page_config(
    page_title="FraudSense AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- STYLE -----------------------------
st.markdown(
    """
<style>
:root {
  --bg: #07111f;
  --panel: rgba(15, 28, 48, 0.72);
  --border: rgba(148, 163, 184, 0.18);
  --text: #e5eefc;
  --muted: #94a3b8;
  --blue: #60a5fa;
  --green: #34d399;
  --red: #fb7185;
  --amber: #fbbf24;
}
html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
.stApp {
  background:
    radial-gradient(circle at top right, rgba(37, 99, 235, .18), transparent 35%),
    radial-gradient(circle at bottom left, rgba(16, 185, 129, .10), transparent 30%),
    var(--bg);
  color: var(--text);
}
.block-container { padding-top: 1.3rem; max-width: 1500px; }
.hero {
  padding: 1.6rem 1.8rem;
  border: 1px solid var(--border);
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(30, 64, 175, .25), rgba(5, 150, 105, .11));
  box-shadow: 0 16px 50px rgba(0,0,0,.20);
}
.hero h1 { margin: 0; font-size: 2.8rem; letter-spacing: -.03em; }
.hero p { color: var(--muted); margin: .45rem 0 0; }
.kpi {
  padding: 1rem 1.1rem;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--panel);
  min-height: 110px;
}
.kpi .label { color: var(--muted); font-size: .82rem; }
.kpi .value { font-size: 1.65rem; font-weight: 800; margin-top: .2rem; }
.risk-card {
  padding: 1.2rem 1.35rem;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: var(--panel);
}
.risk-high { border-left: 6px solid var(--red); }
.risk-medium { border-left: 6px solid var(--amber); }
.risk-low { border-left: 6px solid var(--green); }
.reason {
  padding: .65rem .8rem;
  border-radius: 12px;
  background: rgba(255,255,255,.035);
  border: 1px solid rgba(255,255,255,.07);
  margin: .35rem 0;
}
.small { color: var(--muted); font-size: .83rem; }
div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  padding: .65rem;
  border-radius: 15px;
}
.stButton > button {
  border-radius: 12px;
  min-height: 46px;
  font-weight: 750;
}
</style>
""",
    unsafe_allow_html=True,
)
# ------------------------------ checking files -----------------------------
for name in [
    "fraudsense_model.pkl",
    "fraudsense_metadata.pkl",
    "account_history.db",
]:
    path = ROOT / "fss" / name

    if not path.exists():
        raise FileNotFoundError(f"Model file missing: {path}")

    with path.open("rb") as file:
        header = file.read(200)

    if b"git-lfs.github.com/spec/v1" in header:
        raise RuntimeError(f"Git LFS pointer detected: {name}")

    print(
        f"Verified {name}: {path.stat().st_size:,} bytes",
        flush=True,
    )

print("DEBUG 2: All imports completed", flush=True)
print("DEBUG 3: Starting model loading", flush=True)


# ----------------------------- LOAD -----------------------------
@st.cache_resource(show_spinner="Loading FraudSense AI model...")
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file missing: {META_PATH}")

    print(
        f"DEBUG 4A: Loading model ({MODEL_PATH.stat().st_size:,} bytes)",
        flush=True,
    )
    model = joblib.load(MODEL_PATH)
    print("DEBUG 4B: Model loaded successfully", flush=True)

    print(
        f"DEBUG 4C: Loading metadata ({META_PATH.stat().st_size:,} bytes)",
        flush=True,
    )
    meta = joblib.load(META_PATH)
    print("DEBUG 4D: Metadata loaded successfully", flush=True)
    print(
        f"DEBUG 4E: Metadata type={type(meta).__name__}; "
        f"keys={list(meta.keys())[:30] if isinstance(meta, dict) else 'not a dict'}",
        flush=True,
    )
    return model, meta

try:
    model, meta = load_assets()
except Exception as exc:
    print(f"MODEL/ METADATA LOAD ERROR: {type(exc).__name__}: {exc}", flush=True)
    st.error("FraudSense AI could not load its model artifacts. Check the deployment logs for the exact error.")
    st.exception(exc)
    st.stop()

print("DEBUG 5: Model and metadata loaded; continuing to render UI", flush=True)

if "history" not in st.session_state:
    st.session_state.history = []


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def risk_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "HIGH"
    if score >= threshold * 0.50:
        return "MEDIUM"
    return "LOW"


def risk_class(label: str) -> str:
    return {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[label]


def get_single_account_stats(customer_id: str) -> Tuple[int, float, float]:
    """Query single customer history from SQLite database."""
    if not DB_PATH.exists():
        return 0, 0.0, 0.0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT count, sum, sumsq FROM accounts WHERE customer_id = ?", (customer_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return int(row[0]), float(row[1]), float(row[2])
    return 0, 0.0, 0.0


def score_single(
    amount: float,
    step: int,
    tx_type: str,
    customer_id: str,
    new_dest: bool,
) -> Tuple[float, int, pd.DataFrame, Dict[str, float]]:
    """Build exactly the feature schema expected by the advanced model using SQLite."""
    if model is None or meta is None:
        raise RuntimeError("FraudSense model or metadata is unavailable; load the model artifacts first.")

    customer_id = str(customer_id).strip()

    global_mean = float(meta["global_train_mean"])
    global_std = float(meta["global_train_std"])
    type_mean = {str(k): float(v) for k, v in meta["train_type_avg"].items()}
    type_std = {str(k): float(v) for k, v in meta["train_type_std"].items()}

    count, total, total_sq = get_single_account_stats(customer_id)

    account_mean = total / count if count else global_mean
    variance = (total_sq / count) - (account_mean ** 2) if count else 0.0
    account_std = math.sqrt(max(variance, 0.0))

    hour_of_day = int(step % 24)
    day_index = int(step // 24)
    is_night = int(hour_of_day in {0, 1, 2, 3, 4, 5, 22, 23})

    tx_avg = type_mean.get(tx_type, global_mean)
    tx_std = type_std.get(tx_type, global_std)
    if not np.isfinite(tx_std) or tx_std <= 0:
        tx_std = global_std

    amount_vs_global = amount / (global_mean + 1.0)
    amount_vs_type = amount / (tx_avg + 1.0)
    amount_z_type = float(np.clip((amount - tx_avg) / (tx_std + 1.0), -20, 20))
    amount_to_orig = float(np.clip(amount / (account_mean + 1.0), 0, 1000))
    amount_z_orig = float(np.clip((amount - account_mean) / (account_std + 1.0), -20, 20))
    orig_velocity = count / (step + 1.0)

    new_account_high = int(count == 0 and amount_vs_global >= 2.0)
    repeat_high = int(count >= 3 and amount_to_orig >= 3.0)

    numerical = {
        "orig_count_before": count,
        "new_dest": int(bool(new_dest)),
        "amount_vs_global": float(amount_vs_global),
        "amount_vs_type_avg": float(amount_vs_type),
        "amount_z_type": amount_z_type,
        "amount_to_orig_avg": amount_to_orig,
        "amount_z_orig": amount_z_orig,
        "orig_velocity": float(orig_velocity),
        "new_account_high_val": new_account_high,
        "repeat_account_high_deviation": repeat_high,
        "hour_of_day": hour_of_day,
        "day_index": day_index,
        "is_night": is_night,
        "amount_log": float(np.log1p(amount)),
        "is_round_amount": int(amount > 0 and np.isclose(amount % 1000, 0.0)),
    }

    row = {feature: numerical[feature] for feature in meta["numerical_features"]}
    for col in meta["categorical_features"]:
        row[col] = 1 if col == f"type_{tx_type}" else 0

    x = pd.DataFrame([row])
    x = x.reindex(columns=meta["feature_columns"], fill_value=0)

    score = float(model.predict_proba(x)[:, 1][0])
    prediction = int(score >= float(meta["threshold"]))

    diagnostics = {
        "account_count": float(count),
        "amount_vs_global": float(amount_vs_global),
        "amount_vs_type": float(amount_vs_type),
        "amount_to_orig": float(amount_to_orig),
        "amount_z_orig": float(amount_z_orig),
        "amount_z_type": float(amount_z_type),
        "hour": float(hour_of_day),
        "is_night": float(is_night),
        "new_dest": float(bool(new_dest)),
    }
    return score, prediction, x, diagnostics


def build_explanations(d: Dict[str, float], score: float) -> List[str]:
    reasons: List[str] = []

    if d["new_dest"]:
        reasons.append("Destination account is marked as new / zero-balance.")
    if d["account_count"] == 0:
        reasons.append("Origin account is unseen in the training history.")
    elif d["account_count"] < 3:
        reasons.append(f"Origin account has limited history ({int(d['account_count'])} prior transactions).")
    if d["amount_vs_global"] >= 2:
        reasons.append(f"Amount is {d['amount_vs_global']:.1f}× the global training average.")
    if d["amount_vs_type"] >= 2:
        reasons.append(f"Amount is {d['amount_vs_type']:.1f}× the typical amount for this transaction type.")
    if d["amount_to_orig"] >= 3 and d["account_count"] >= 3:
        reasons.append(f"Amount is {d['amount_to_orig']:.1f}× this account's historical average.")
    if abs(d["amount_z_type"]) >= 3:
        reasons.append("Transaction amount is a strong statistical outlier for its transaction type.")
    if abs(d["amount_z_orig"]) >= 3 and d["account_count"] >= 2:
        reasons.append("Transaction amount is a strong outlier relative to the origin account's history.")
    if d["is_night"]:
        reasons.append(f"Transaction occurs at hour {int(d['hour'])}, inside the configured night-time window.")
    if not reasons:
        reasons.append("No single high-severity behavioral rule dominated; the model score is driven by the combined feature pattern.")

    return reasons[:6]


def risk_gauge(score: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#fb7185" if score >= threshold else "#34d399"},
                "steps": [
                    {"range": [0, max(threshold * 100 * 0.5, 1)], "color": "rgba(52,211,153,.16)"},
                    {"range": [max(threshold * 100 * 0.5, 1), threshold * 100], "color": "rgba(251,191,36,.16)"},
                    {"range": [threshold * 100, 100], "color": "rgba(251,113,133,.18)"},
                ],
                "threshold": {
                    "line": {"color": "#fbbf24", "width": 4},
                    "thickness": .8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=25, r=25, t=25, b=10))
    return fig


def shap_chart(x: pd.DataFrame) -> go.Figure | None:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(x)
        vals = np.asarray(vals)
        if vals.ndim == 2:
            vals = vals[0]

        pairs = (
            pd.DataFrame({"feature": x.columns, "impact": vals})
            .assign(abs_impact=lambda z: z["impact"].abs())
            .sort_values("abs_impact", ascending=False)
            .head(10)
            .sort_values("impact")
        )

        fig = px.bar(
            pairs,
            x="impact",
            y="feature",
            orientation="h",
            title="Top model contribution signals",
            labels={"impact": "SHAP contribution", "feature": "Feature"},
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
        return fig
    except Exception:
        return None


def load_analytics_file(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        local = ROOT / "paysim.csv"
        if not local.exists():
            return None
        return pd.read_csv(local)

    return pd.read_csv(uploaded_file)


# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown("## 🛡️ FraudSense AI")
    st.caption("Advanced PaySim fraud intelligence")

    if model is None or meta is None:
        st.error("Advanced model artifacts are missing.")
        st.info(
            "Run `python detection_pipeline.py` first. "
            "The script creates the model and metadata under `fss/`."
        )
    else:
        st.success("Advanced XGBoost model loaded")
        st.caption(f"Schema: {meta.get('schema_version', 'unknown')}")
        st.caption(f"Operating threshold: {meta['threshold']:.4f}")

    st.divider()
    st.markdown("### Navigation")
    st.caption("Use the tabs across the main area for detection, batch scoring, analytics, and model validation.")

# ----------------------------- HERO -----------------------------
st.markdown(
    """
<div class="hero">
  <h1>🛡️ FraudSense AI</h1>
  <p>Advanced transaction-behavior intelligence for PaySim-style fraud detection.</p>
</div>
""",
    unsafe_allow_html=True,
)

if model is None or meta is None:
    st.stop()

tabs = st.tabs(
    [
        "📊 Command Center",
        "🔍 Live Detection",
        "📦 Batch Screening",
        "📈 Dataset Analytics",
        "🧠 Model Insights",
        "🕘 Detection History",
    ]
)

# ----------------------------- COMMAND CENTER -----------------------------
with tabs[0]:
    m = meta["metrics"]
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("PR-AUC", f"{m['pr_auc']:.4f}")
    with k2:
        st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
    with k3:
        st.metric("Precision", fmt_pct(m["precision"]))
    with k4:
        st.metric("Recall", fmt_pct(m["recall"]))
    with k5:
        st.metric("F1", f"{m['f1']:.4f}")

    st.markdown("### Model operating profile")
    c1, c2 = st.columns([1.1, 1.9])
    with c1:
        st.markdown(
            f"""
            <div class="risk-card risk-medium">
              <div class="small">Optimized decision threshold</div>
              <div style="font-size:2rem;font-weight:800">{meta['threshold']:.5f}</div>
              <div class="small">
                F1 threshold: {meta['threshold_f1']:.5f}<br>
                F2 threshold: {meta['threshold_f2']:.5f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        cm = np.array(m["confusion_matrix"])
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            title="Temporal hold-out confusion matrix",
            aspect="auto",
        )
        fig.update_layout(height=330, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SQLite-Backed Metadata Optimization")
    st.markdown(
        """
        **Low-Memory Architecture:** Customer history records are stored in an indexed SQLite database (`account_history.db`) 
        rather than pickled dictionaries. This keeps the metadata file slim and eliminates Out-Of-Memory (OOM) errors on Streamlit Cloud.
        """
    )

# ----------------------------- LIVE DETECTION -----------------------------
with tabs[1]:
    st.markdown("## 🔍 Real-time transaction analysis")
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.01,
            max_value=10_000_000.0,
            value=1500.0,
            step=100.0,
            help="Amount used by the model in the same units as PaySim.",
        )
        tx_type = st.selectbox("Transaction Type", CATEGORICAL_TYPES, index=4)
        customer_id = st.text_input(
            "Origin Account ID",
            value="C123456789",
            help="Use a PaySim-style origin account identifier. Known accounts use their stored training history.",
        ).strip()
        step = st.slider("Step / Hour of simulation", 0, 743, 200)
        new_dest = st.toggle(
            "New destination account / zero previous destination balance",
            value=False,
        )

        a1, a2 = st.columns(2)
        with a1:
            demo = st.button("🎲 Load suspicious demo", use_container_width=True)
        with a2:
            analyze = st.button("🚨 Analyze transaction", type="primary", use_container_width=True)

        if demo:
            amount = 750_000.0
            tx_type = "TRANSFER"
            customer_id = "C_NEW_0001"
            step = 713
            new_dest = True
            analyze = True

        if analyze:
            if not customer_id:
                st.error("Origin Account ID is required.")
            else:
                score, pred, x, diag = score_single(
                    amount=float(amount),
                    step=int(step),
                    tx_type=tx_type,
                    customer_id=customer_id,
                    new_dest=bool(new_dest),
                )

                threshold = float(meta["threshold"])
                label = risk_label(score, threshold)
                reasons = build_explanations(diag, score)

                st.session_state.history.append(
                    {
                        "amount": float(amount),
                        "type": tx_type,
                        "customer_id": customer_id,
                        "step": int(step),
                        "new_dest": bool(new_dest),
                        "score": score,
                        "prediction": pred,
                        "risk": label,
                    }
                )
                st.session_state.history = st.session_state.history[-30:]

                with right:
                    st.plotly_chart(
                        risk_gauge(score, threshold),
                        use_container_width=True,
                    )

                rclass = risk_class(label)
                title = {
                    "HIGH": "🚨 High-risk transaction",
                    "MEDIUM": "⚠️ Medium-risk transaction",
                    "LOW": "✅ Low-risk transaction",
                }[label]

                st.markdown(
                    f"""
                    <div class="risk-card {rclass}">
                      <div style="font-size:1.4rem;font-weight:800">{title}</div>
                      <div style="font-size:1.05rem;margin-top:.3rem">
                        Model fraud score: <b>{score*100:.2f}%</b>
                      </div>
                      <div class="small">
                        Operating threshold: {threshold*100:.2f}% score-equivalent
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Account history", f"{int(diag['account_count'])} txns")
                c2.metric("vs global avg", f"{diag['amount_vs_global']:.2f}×")
                c3.metric("vs account avg", f"{diag['amount_to_orig']:.2f}×")
                c4.metric("Hour", f"{int(diag['hour'])}:00")

                st.markdown("### Behavioral reasons")
                for reason in reasons:
                    st.markdown(f'<div class="reason">• {reason}</div>', unsafe_allow_html=True)

                st.markdown("### Model explanation")
                shap_fig = shap_chart(x)
                if shap_fig is not None:
                    st.plotly_chart(shap_fig, use_container_width=True)
                else:
                    st.info("Optional SHAP visualization is unavailable; the behavioral explanations above are still active.")

                with st.expander("View model-ready feature vector"):
                    st.dataframe(x.T.rename(columns={0: "value"}), use_container_width=True)

    if not analyze and not demo:
        st.info("Enter a transaction and select **Analyze transaction** to generate a real-time risk score and explanation.")

# ----------------------------- BATCH SCREENING -----------------------------
with tabs[2]:
    st.markdown("## 📦 Batch fraud screening")
    st.caption(
        "Upload a raw PaySim-style CSV. Batches are screened efficiently using SQLite-backed history."
    )
    uploaded_batch = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key="batch_upload",
    )
    max_rows = st.slider("Maximum rows to score in this session", 100, 100_000, 25_000, step=100)

    if uploaded_batch is not None:
        try:
            batch = pd.read_csv(uploaded_batch)
            batch = batch.rename(columns=RAW_RENAME)
            required = {"Hours", "Transaction_Type", "CustomerID", "Destination_Old_Balance", "Transaction_Amount"}
            missing = sorted(required - set(batch.columns))

            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                if len(batch) > max_rows:
                    st.warning(
                        f"File has {len(batch):,} rows. Scoring only the first {max_rows:,} rows in the interactive session."
                    )
                    batch = batch.head(max_rows).copy()

                batch["CustomerID"] = batch["CustomerID"].astype(str)
                batch["Transaction_Type"] = batch["Transaction_Type"].astype(str).str.upper()

                with st.spinner("Engineering behavior features from SQLite and scoring batch..."):
                    X_batch, _ = add_historical_features(
                        batch,
                        train_stats={
                            "global_train_mean": meta["global_train_mean"],
                            "global_train_std": meta["global_train_std"],
                            "train_type_avg": meta["train_type_avg"],
                            "train_type_std": meta["train_type_std"],
                        },
                        db_path=DB_PATH,
                    )
                    X_batch = X_batch.reindex(columns=meta["feature_columns"], fill_value=0)
                    batch_scores = model.predict_proba(X_batch)[:, 1]
                    batch_out = batch.reset_index(drop=True).copy()
                    batch_out["fraud_score"] = batch_scores
                    batch_out["prediction"] = (batch_scores >= meta["threshold"]).astype(int)
                    batch_out["risk_level"] = [
                        risk_label(float(s), float(meta["threshold"])) for s in batch_scores
                    ]

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Rows screened", f"{len(batch_out):,}")
                b2.metric("High-risk", f"{int((batch_out['risk_level']=='HIGH').sum()):,}")
                b3.metric("Medium-risk", f"{int((batch_out['risk_level']=='MEDIUM').sum()):,}")
                b4.metric("Max score", fmt_pct(float(batch_out["fraud_score"].max())))

                fig = px.histogram(
                    batch_out,
                    x="fraud_score",
                    nbins=50,
                    title="Batch fraud-score distribution",
                )
                fig.add_vline(x=float(meta["threshold"]), line_dash="dash", annotation_text="Operating threshold")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    batch_out[
                        [
                            c for c in [
                                "Hours",
                                "Transaction_Type",
                                "CustomerID",
                                "Transaction_Amount",
                                "fraud_score",
                                "prediction",
                                "risk_level",
                            ] if c in batch_out.columns
                        ]
                    ].sort_values("fraud_score", ascending=False),
                    use_container_width=True,
                    height=430,
                )

                csv_bytes = batch_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download scored transactions",
                    data=csv_bytes,
                    file_name="fraudsense_scored_transactions.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")

# ----------------------------- DATASET ANALYTICS -----------------------------
with tabs[3]:
    st.markdown("## 📈 Interactive PaySim analytics")
    analytics_file = st.file_uploader(
        "Upload a dataset for visualization (optional)",
        type=["csv"],
        key="analytics_upload",
    )

    try:
        analytics = load_analytics_file(analytics_file)
        if analytics is None:
            st.info("Upload `paysim.csv` to populate the analytics dashboard, or place it beside the app.")
        else:
            analytics = analytics.rename(columns=RAW_RENAME)
            required = {"Hours", "Transaction_Type", "Transaction_Amount"}
            missing = sorted(required - set(analytics.columns))
            if missing:
                st.error(f"Analytics file is missing: {missing}")
            else:
                viz = analytics.copy()
                if len(viz) > 250_000:
                    viz = viz.sample(250_000, random_state=42)
                    st.caption("Visualizations are sampled to 250k rows for responsiveness.")

                has_target = "Is_Fraud" in viz.columns
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Transactions", f"{len(analytics):,}")
                if has_target:
                    k2.metric("Fraud rate", fmt_pct(float(analytics["Is_Fraud"].mean())))
                    k3.metric(
                        "Fraud amount",
                        f"{analytics.loc[analytics['Is_Fraud']==1, 'Transaction_Amount'].sum():,.0f}",
                    )
                else:
                    k2.metric("Fraud rate", "N/A")
                    k3.metric("Fraud amount", "N/A")
                k4.metric(
                    "Median amount",
                    f"{analytics['Transaction_Amount'].median():,.2f}",
                )

                c1, c2 = st.columns(2)
                with c1:
                    by_type = viz.groupby("Transaction_Type", as_index=False)
                    if has_target:
                        type_fraud = by_type["Is_Fraud"].mean()
                        fig1 = px.bar(
                            type_fraud,
                            x="Transaction_Type",
                            y="Is_Fraud",
                            title="Fraud rate by transaction type",
                            labels={"Is_Fraud": "Fraud rate"},
                        )
                        fig1.update_yaxes(tickformat=".2%")
                    else:
                        type_count = by_type.size().rename(columns={"size": "count"})
                        fig1 = px.bar(type_count, x="Transaction_Type", y="count", title="Transaction volume by type")
                    st.plotly_chart(fig1, use_container_width=True)

                with c2:
                    hourly = viz.copy()
                    hourly["hour_of_day"] = hourly["Hours"] % 24
                    if has_target:
                        hour_fraud = hourly.groupby("hour_of_day", as_index=False)["Is_Fraud"].mean()
                        fig2 = px.line(
                            hour_fraud,
                            x="hour_of_day",
                            y="Is_Fraud",
                            markers=True,
                            title="Fraud rate by hour of day",
                            labels={"Is_Fraud": "Fraud rate", "hour_of_day": "Hour"},
                        )
                        fig2.update_yaxes(tickformat=".2%")
                    else:
                        hour_count = hourly.groupby("hour_of_day", as_index=False).size()
                        fig2 = px.line(hour_count, x="hour_of_day", y="size", markers=True, title="Transaction volume by hour")
                    st.plotly_chart(fig2, use_container_width=True)

    except Exception as exc:
        st.error(f"Analytics dashboard failed: {exc}")

# ----------------------------- MODEL INSIGHTS -----------------------------
with tabs[4]:
    st.markdown("## 🧠 Model explainability & validation")

    fi = pd.DataFrame(meta.get("feature_importance", []))
    if not fi.empty:
        fi = fi.sort_values("importance", ascending=True)
        fig = px.bar(
            fi.tail(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Top global model features",
        )
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)

    m = meta["metrics"]
    cm = np.array(m["confusion_matrix"])
    left, right = st.columns(2)
    with left:
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            title="Confusion matrix",
            aspect="auto",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with right:
        report = pd.DataFrame(
            {
                "Metric": ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Operating threshold"],
                "Value": [
                    m["pr_auc"],
                    m["roc_auc"],
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    meta["threshold"],
                ],
            }
        )
        st.dataframe(report, hide_index=True, use_container_width=True)

# ----------------------------- HISTORY -----------------------------
with tabs[5]:
    st.markdown("## 🕘 Recent detection history")
    history = pd.DataFrame(st.session_state.history)

    if history.empty:
        st.info("No transactions have been analyzed in this session yet.")
    else:
        st.dataframe(
            history.sort_values("score", ascending=False),
            use_container_width=True,
        )

        fig = px.line(
            history.reset_index(),
            x="index",
            y="score",
            markers=True,
            hover_data=["type", "customer_id", "amount", "risk"],
            title="Session fraud-score trend",
            labels={"index": "Detection number", "score": "Fraud score"},
        )
        fig.add_hline(
            y=float(meta["threshold"]),
            line_dash="dash",
            annotation_text="Operating threshold",
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🗑️ Clear detection history"):
            st.session_state.history = []
            st.rerun()