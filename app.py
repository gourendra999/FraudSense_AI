import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="FraudSense AI", layout="wide", page_icon="🛡️")

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #172554,
        #0b3d2e
    );
    color: white;
}

/* Header */
.main-title{
    text-align:center;
    color:#60A5FA;
    font-size:3rem;
    font-weight:800;
}

.sub-title{
    text-align:center;
    color:#34D399;
    font-size:1.1rem;
}
            
/* selectbox */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    color: white !important;
}

/* Number Input Container */
div[data-testid="stNumberInput"] {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
}

/* Cards */
.metric-card{
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);

    color:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.4);
    border-left:5px solid #60A5FA;
}
.fraud-card{
    background: rgba(63, 21, 24, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);

    color:white;
    padding:25px;
    border-radius:15px;
    border-left:8px solid #EF4444;
}

.safe-card{
    background: rgba(18, 53, 36, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);

    color:white;
    padding:25px;
    border-radius:15px;
    border-left:8px solid #22C55E;
}

.stButton > button {
    width: 100%;
    height: 55px;

    background: linear-gradient(
        90deg,
        #2563EB,
        #10B981
    );

    color: white;
    font-size: 18px;
    font-weight: 700;

    border: none;
    border-radius: 12px;

    box-shadow: 0px 4px 15px rgba(37,99,235,0.4);
}

.stButton > button:hover {
    transform: translateY(-2px);
    transition: 0.3s;
}

</style>
""", unsafe_allow_html=True)

# # HEADER
# # -----------------------------

st.markdown("""
<div class="main-title">
🛡️ FraudSense AI
</div>

<div class="sub-title">
Real-Time Transactional Behaviour Detection System
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
# 💳 Transaction Details
Enter transaction information for risk analysis.
""")


@st.cache_resource # Keeps the model in memory so the app stays fast
def load_assets():
    model = joblib.load('fss/fraud_sense_model1.pkl')
    meta = joblib.load('fss/fraudsense_metadata1.pkl')
    return model, meta

model, meta = load_assets()

# ---INPUTS ---
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=0.0)
tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], index=None, placeholder="Select transaction type")
name_Orig = st.text_input("Origin Account ID", placeholder="e.g. C123456789")
new_dest = st.checkbox("New Destination Account (Zero Balance)", value=False)
new_dest_value = 1 if new_dest else 0,
step = st.slider("Step (Hour of Month)", 0, 744, 0)

# --- REAL-TIME FEATURE ENGINEERING ---
if st.button("🔍 Analyze Transaction"):
    if amount == 0.0 :
        st.warning("⚠️ Please fill in all required input values before analysis.")

    else:
    # Create single observation dataframe
        obs = pd.DataFrame([{
        'amount': amount,
        'type': tx_type,
        'nameOrig': name_Orig,
        'step': step
        }])
    
        # Apply Path 2 Logic
        obs['orig_count'] = obs['nameOrig'].map(meta['train_counts']).fillna(1).astype(int)
        obs['new_dest'] = new_dest_value
        obs['amt_deviation'] = obs['amount'] / (meta['global_train_mean'] + 1)
        obs['hour_of_day'] = obs['step'] % 24
        obs['amount_log'] = np.log1p(obs['amount'])
        obs['new_account_high_val'] = ((obs['orig_count'] < 2) & (obs['amt_deviation'] > 2)).astype(int)

        # Static/Calculated features based on your training list
        obs['is_high_value'] = (obs['amount'] > 100000).astype(int) 
        obs['amount_diff_avg'] = obs['amount'] - (meta['global_train_mean'])
        obs['amt_to_avg_ratio'] = obs['amount'] / (meta['avg_amounts'] + 1)
        obs['type_relative_amt'] = obs['amount'] / (meta['global_train_mean'] + 1) # Simplification for app
    
         # One-Hot Encode and Align
        obs = pd.get_dummies(obs, columns=['type'])
        for col in meta['model_columns']:
            if col not in obs.columns:
                obs[col] = 0
        final_input = obs[meta['model_columns']]
    
        # Prediction
        prob = model.predict_proba(final_input)[:, 1][0]
        is_fraud = (prob >= meta['threshold'])
    
    # --- UI DISPLAY ---
        st.subheader("Analysis Result")
    
        col1, col2 = st.columns([1,1])
        with col1:
            st.write("#### Raw Transaction Data")
            st.dataframe(final_input, width='stretch')
    
        with col2:
            if is_fraud:
                st.markdown(f"""
             <div class="fraud-card">
             <h2>🚨 Fraudulent Transaction Detected</h2>
             <h3>Risk Probability: {prob*100:.2f}%</h3>
             <p>
             This transaction exhibits suspicious behavioral patterns
             and should be reviewed immediately.
             </p>
             </div>
            """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
             <div class="safe-card">
             <h2>✅ Transaction Safe</h2>
             <h3>Risk Probability: {prob*100:.2f}%</h3>
             <p>
             This transaction does not exhibit any suspicious behavioral patterns.
             </p>
             </div>
             """, unsafe_allow_html=True)
                
        # Risk Level

        st.write("### Risk Meter")

        st.progress(float(prob))

        if prob < 0.3:
            st.success("🟢 Low Risk")

        elif prob < 0.7:
            st.warning("🟠 Medium Risk")

        else:
            st.error("🔴 High Risk")

        
        # Show behavioral breakdown
        with st.expander("View Behavioral Breakdown"):
            st.write(f"- **Account Frequency:** {int(obs['orig_count'].iloc[0])} previous transactions")
            st.write(f"- **Amount Outlier:** {float(obs['amt_deviation'].iloc[0]):.2f}x global average")
            st.write(f"- **Time of Day:** Hour {int(obs['hour_of_day'].iloc[0])}")