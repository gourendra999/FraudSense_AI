# 🚨 REAL-TIME FRAUD DETECTION ENGINE (FraudSense AI)

An enterprise-grade, high-recall machine learning pipeline built to detect and flag fraudulent peer-to-peer mobile money transactions in **real-time**.

---

## 📖 Overview
FraudSense AI leverages the **PaySim dataset** (6.3 million records) to train a robust fraud detection model.  
The dataset is highly imbalanced: **99.87% non-fraud** vs. **0.13% fraud**.  

By targeting high-risk transactional vectors (`TRANSFER` and `CASH_OUT`), the engine eliminates noise and focuses classification power precisely where fraud occurs.  
Unlike static threshold systems, FraudSense AI employs an **XGBoost Classifier Pipeline** under a **Security-First paradigm**, maximizing recall to catch fraudulent activity before capital extraction occurs.

---

## 🏗️ Key Architectural Pillars

### 1. Chronological Out-of-Time Splitting (Time-Based Validation)
- **Problem**: Random k-fold validation introduces chronological leakage (training on “future” data to predict the “past”).  
- **Solution**: Sequential split along the transaction timeline.  
  - **Training Block**: Early hours reserved for learning baseline behavioral distributions.  
  - **OOT Test Set**: Later hours isolated for evaluation, simulating true production roll-out.  
- **Leakage-Free Aggregations**: Historical frequencies (e.g., recipient popularity, sender velocities) calculated strictly within the training window and mapped forward.

---

### 2. Zero-Leakage Feature Engineering
Raw account balance histories are excluded to prevent leakage. Instead, the engine relies on dynamic operational behaviors:

- **Contextual Amount Scales**:  
  - `amt_deviation`: Global anomaly detection.  
  - `type_relative_amt`: Peer-group relative transaction scale.  

- **Internal Account Velocity**:  
  - `amt_to_avg_ratio`: Tracks sudden spending spikes relative to customer history.  

- **Network Link Analysis**:  
  - `new_dest`: Flags transfers to brand-new, unverified mule accounts.  

---

### 3. Operational Risk Gating
- Outputs **continuous risk probabilities** instead of rigid binary classifications.  
- Production threshold tuned at **0.6**, optimizing defensive risk mitigation.  
- Breaks past traditional precision-recall trade-offs to maximize fraud detection.

---

## 📊 Core Metric Performance
Evaluated against strict chronological test splits (true production simulation):

- **Recall**: `0.60` → Catches 60% of active fraud attacks.  
- **Precision**: `0.64` → 64% of alerts are actual fraud, reducing alert fatigue.  
- **F1-Score**: `0.62` → Balanced performance under extreme imbalance.  
- **Global Accuracy**: `98%` → High stability across the transactional ecosystem.  

---

## 🛠 Tech Stack
- **Python 3.12**
- **scikit-learn**
- **XGBoost**
- **pandas / NumPy**
- **Streamlit / Flask** (for deployment & visualization)
