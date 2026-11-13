import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="BGFAD Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    margin: 5px;
}
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
.fraud-card  { background: linear-gradient(135deg, #7b1f1f, #c0392b); }
.safe-card   { background: linear-gradient(135deg, #1a5c2a, #27ae60); }
.warn-card   { background: linear-gradient(135deg, #7d5a00, #f39c12); }
.info-card   { background: linear-gradient(135deg, #1a3a5c, #2980b9); }
.section-title { font-size:1.3rem; font-weight:700; color:#2d6a9f;
                 border-left:4px solid #2d6a9f; padding-left:10px; margin:15px 0 10px 0; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Load all data (cached)
# -------------------------------------------------------
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    def p(f): return os.path.join(base, f)
    d = {}
    d["fusion"]      = pd.read_csv(p("dataset/fusion_data.csv"))
    d["predictions"] = pd.read_csv(p("dataset/final_predictions.csv"))
    d["ablation"]    = pd.read_csv(p("results/tables/ablation_study.csv"))
    d["comparison"]  = pd.read_csv(p("results/tables/model_comparison.csv"))
    d["performance"] = pd.read_csv(p("results/tables/model_performance.csv"))
    d["preprocess"]  = pd.read_csv(p("results/tables/preprocessing_summary.csv"))
    d["graph_stats"] = pd.read_csv(p("results/tables/graph_statistics.csv"))
    d["bcs_stats"]   = pd.read_csv(p("results/tables/behavior_score_results.csv"))
    d["aas_stats"]   = pd.read_csv(p("results/tables/anomaly_score_results.csv"))
    d["fusion_stats"]= pd.read_csv(p("results/tables/fusion_results.csv"))
    d["gts_stats"]   = pd.read_csv(p("results/tables/graph_trust_results.csv"))
    d["importance"]  = pd.read_csv(p("results/tables/feature_importance.csv"))
    d["sl_curve"]    = pd.read_csv(p("results/tables/self_learning_curve.csv"))
    d["sl_results"]  = pd.read_csv(p("results/tables/self_learning_results.csv"))
    d["cm"]          = pd.read_csv(p("results/tables/confusion_matrix.csv"), index_col=0)
    return d

data = load_data()

# -------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=60)
st.sidebar.title("BGFAD")
st.sidebar.caption("Behavior-Aware Graph Fusion\nAdaptive Detection")
st.sidebar.markdown("---")

pages = [
    "🏠 Home",
    "📊 Dataset & Preprocessing",
    "🧠 BGFAD Scores",
    "🔬 Ablation Study",
    "🤖 Model Comparison",
    "🎯 Live Prediction",
    "📈 Self-Learning",
    "💡 Explainable AI",
]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")
st.sidebar.caption("IEEE-CIS Fraud Detection Dataset")

# -------------------------------------------------------
# Helper: metric card
# -------------------------------------------------------
def card(label, value, css=""):
    st.markdown(f"""
    <div class="metric-card {css}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

# ================================================================
# PAGE: HOME
# ================================================================
if page == "🏠 Home":
    st.title("🛡️ BGFAD — Behavior-Aware Graph Fusion Adaptive Detection")
    st.markdown("**A Novel Explainable AI Framework for Real-Time Identity Theft and Account Takeover Detection in Fintech Ecosystems**")
    st.markdown("---")

    perf = data["performance"].set_index("Metric")["Value"]
    fusion_df = data["fusion"]
    total = len(fusion_df)
    fraud = int(fusion_df["isFraud"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: card("Total Transactions", f"{total:,}", "info-card")
    with c2: card("Fraud Cases", f"{fraud:,}", "fraud-card")
    with c3: card("ROC-AUC", f"{perf['ROC-AUC']:.4f}", "safe-card")
    with c4: card("Recall", f"{perf['Recall']:.4f}", "warn-card")
    with c5: card("PR-AUC", f"{perf['PR-AUC']:.4f}", "info-card")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        section("Pipeline Architecture")
        steps = [
            ("1", "Preprocessing", "Clean, encode, scale raw data"),
            ("2", "Feature Engineering", "10 domain-specific features"),
            ("3", "BCS", "Behavioral Consistency Score"),
            ("4", "Graph Construction", "Heterogeneous transaction graph"),
            ("5", "GTS", "Graph Trust Score propagation"),
            ("6", "AAS", "Adaptive Anomaly Score"),
            ("7", "Fusion", "Context Fusion Score"),
            ("8", "LightGBM", "Adaptive Risk Detection"),
            ("9", "Self-Learning", "Incremental feedback loop"),
            ("10", "SHAP", "Explainable AI"),
        ]
        for num, name, desc in steps:
            st.markdown(f"**Step {num} — {name}**: {desc}")

    with c2:
        section("Risk Decision Tiers")
        tiers = pd.DataFrame({
            "Risk Score": ["< 30", "30 – 59", "60 – 79", ">= 80"],
            "Decision":   ["✅ Safe", "📱 OTP Verification", "🔐 Biometric Verification", "🚫 Blocked"],
            "Color":      ["#27ae60", "#f39c12", "#e67e22", "#c0392b"]
        })
        for _, row in tiers.iterrows():
            st.markdown(
                f"<div style='background:{row['Color']};color:white;padding:8px 14px;"
                f"border-radius:8px;margin:4px 0;font-weight:600'>"
                f"{row['Risk Score']} &nbsp;→&nbsp; {row['Decision']}</div>",
                unsafe_allow_html=True
            )

        st.markdown("")
        section("Key SHAP Features")
        imp = data["importance"].head(7)
        fig = px.bar(imp, x="SHAP_Mean", y="Feature", orientation="h",
                     color="SHAP_Mean", color_continuous_scale="Blues",
                     labels={"SHAP_Mean": "Mean |SHAP|"})
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# PAGE: DATASET & PREPROCESSING
# ================================================================
elif page == "📊 Dataset & Preprocessing":
    st.title("📊 Dataset & Preprocessing")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        section("Preprocessing Summary")
        pre = data["preprocess"]
        st.dataframe(pre, use_container_width=True, hide_index=True)

        section("Graph Statistics")
        gs = data["graph_stats"]
        st.dataframe(gs, use_container_width=True, hide_index=True)

    with c2:
        section("Class Distribution")
        fusion_df = data["fusion"]
        counts = fusion_df["isFraud"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        counts["Class"] = counts["Class"].map({0: "Normal", 1: "Fraud"})
        fig = px.pie(counts, names="Class", values="Count",
                     color="Class",
                     color_discrete_map={"Normal": "#2196F3", "Fraud": "#F44336"},
                     hole=0.45)
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

        section("Transaction Amount Distribution")
        if "TransactionAmt" in fusion_df.columns:
            fig2 = px.histogram(fusion_df, x="TransactionAmt", nbins=60,
                                color_discrete_sequence=["#2196F3"])
            fig2.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    section("Fraud vs Normal — Score Comparison")
    score_cols = ["BCS", "GraphTrustScore", "AAS", "FusionScore"]
    available = [c for c in score_cols if c in fusion_df.columns]
    sel = st.selectbox("Select Score", available)
    fig3 = px.histogram(fusion_df, x=sel, color=fusion_df["isFraud"].map({0:"Normal",1:"Fraud"}),
                        nbins=50, barmode="overlay", opacity=0.7,
                        color_discrete_map={"Normal":"#2196F3","Fraud":"#F44336"},
                        labels={"color":"Class"})
    fig3.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ================================================================
# PAGE: BGFAD SCORES
# ================================================================
elif page == "🧠 BGFAD Scores":
    st.title("🧠 BGFAD Component Scores")
    st.markdown("---")
    fusion_df = data["fusion"]

    tab1, tab2, tab3, tab4 = st.tabs(["BCS", "GTS", "AAS", "Fusion Score"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(fusion_df, x="BCS", nbins=50,
                               color_discrete_sequence=["#2196F3"],
                               title="Behavioral Consistency Score Distribution")
            fig.add_vline(x=fusion_df["BCS"].mean(), line_dash="dash",
                          line_color="red", annotation_text=f"Mean={fusion_df['BCS'].mean():.3f}")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            section("BCS Statistics")
            st.dataframe(data["bcs_stats"], use_container_width=True, hide_index=True)
            st.markdown("**Formula:**")
            st.latex(r"BCS = \frac{S_{beh} + S_{dev} + S_{loc}}{3}")

    with tab2:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(fusion_df, x="GraphTrustScore", nbins=50,
                               color_discrete_sequence=["#4CAF50"],
                               title="Graph Trust Score Distribution")
            fig.add_vline(x=fusion_df["GraphTrustScore"].mean(), line_dash="dash",
                          line_color="red",
                          annotation_text=f"Mean={fusion_df['GraphTrustScore'].mean():.3f}")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            section("GTS Statistics")
            st.dataframe(data["gts_stats"], use_container_width=True, hide_index=True)
            st.markdown("**Formula:**")
            st.latex(r"GTS = 0.40 \cdot T_0 + 0.30 \cdot C + 0.30 \cdot T_N")

    with tab3:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(fusion_df, x="AAS", nbins=50,
                               color_discrete_sequence=["#FF9800"],
                               title="Adaptive Anomaly Score Distribution")
            fig.add_vline(x=fusion_df["AAS"].mean(), line_dash="dash",
                          line_color="red",
                          annotation_text=f"Mean={fusion_df['AAS'].mean():.3f}")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            section("AAS Statistics")
            st.dataframe(data["aas_stats"], use_container_width=True, hide_index=True)
            st.markdown("**Formula:**")
            st.latex(r"AAS = \frac{D_b + R_t + (1-GTS) + (1-U_r)}{4}")

    with tab4:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(fusion_df, x="FusionScore", nbins=50,
                               color_discrete_sequence=["#9C27B0"],
                               title="Context Fusion Score Distribution")
            fig.add_vline(x=fusion_df["FusionScore"].mean(), line_dash="dash",
                          line_color="red",
                          annotation_text=f"Mean={fusion_df['FusionScore'].mean():.3f}")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            section("Fusion Statistics")
            st.dataframe(data["fusion_stats"], use_container_width=True, hide_index=True)
            st.markdown("**Formula:**")
            st.latex(r"F = 0.40 \cdot BCS + 0.35 \cdot GTS + 0.25 \cdot (1-AAS)")

# ================================================================
# PAGE: ABLATION STUDY
# ================================================================
elif page == "🔬 Ablation Study":
    st.title("🔬 Ablation Study — Incremental Component Contribution")
    st.markdown("Each row adds one BGFAD component. Same LightGBM + same split.")
    st.markdown("---")

    abl = data["ablation"]

    section("Table V — Ablation Results")
    styled = abl.style.background_gradient(
        subset=["F1 Score","Recall","ROC-AUC","PR-AUC"], cmap="YlGn"
    ).background_gradient(
        subset=["FPR","FNR"], cmap="YlOrRd_r"
    ).format(precision=4)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        section("F1 / Recall / ROC-AUC / PR-AUC")
        metrics = ["F1 Score", "Recall", "ROC-AUC", "PR-AUC"]
        fig = go.Figure()
        colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0"]
        for m, col in zip(metrics, colors):
            fig.add_trace(go.Bar(name=m, x=abl["Experiment"], y=abl[m],
                                 marker_color=col))
        fig.update_layout(barmode="group", height=380,
                          xaxis_tickangle=-20,
                          legend=dict(orientation="h", y=1.1),
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("FPR vs FNR Across Experiments")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="FPR", x=abl["Experiment"], y=abl["FPR"],
                              marker_color="#F44336"))
        fig2.add_trace(go.Bar(name="FNR", x=abl["Experiment"], y=abl["FNR"],
                              marker_color="#FF9800"))
        fig2.update_layout(barmode="group", height=380,
                           xaxis_tickangle=-20,
                           legend=dict(orientation="h", y=1.1),
                           margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    section("ROC-AUC Progression")
    fig3 = px.line(abl, x="Experiment", y="ROC-AUC", markers=True,
                   color_discrete_sequence=["#2196F3"])
    fig3.update_traces(line_width=3, marker_size=10)
    fig3.update_layout(height=300, xaxis_tickangle=-15,
                       margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3, use_container_width=True)

    section("Key Insight")
    st.info(
        "**AAS contributes the largest single gain**: adding AAS improves ROC-AUC by +0.1442 "
        "and F1 by +0.0690. The full BGFAD system achieves the best PR-AUC (0.0807) and "
        "lowest FPR (0.0206), confirming each component adds measurable value."
    )

# ================================================================
# PAGE: MODEL COMPARISON
# ================================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Baseline Model Comparison")
    st.markdown("All models trained on full BGFAD features with class balancing.")
    st.markdown("---")

    cmp = data["comparison"]

    section("Table IV — Model Comparison (8 Metrics)")
    styled2 = cmp.style.background_gradient(
        subset=["F1 Score","Recall","ROC-AUC","PR-AUC"], cmap="YlGn"
    ).background_gradient(
        subset=["FPR","FNR"], cmap="YlOrRd_r"
    ).format(precision=4)
    st.dataframe(styled2, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        section("Accuracy / F1 / ROC-AUC")
        fig = go.Figure()
        for m, col in zip(["Accuracy","F1 Score","ROC-AUC"],
                          ["#2196F3","#4CAF50","#FF9800"]):
            fig.add_trace(go.Bar(name=m, x=cmp["Experiment"], y=cmp[m],
                                 marker_color=col))
        fig.update_layout(barmode="group", height=360,
                          xaxis_tickangle=-15,
                          legend=dict(orientation="h", y=1.1),
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Precision vs Recall Trade-off")
        fig2 = px.scatter(cmp, x="Recall", y="Precision",
                          text="Experiment", size="F1 Score",
                          color="ROC-AUC", color_continuous_scale="Viridis",
                          size_max=40)
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        section("ROC-AUC vs PR-AUC")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="ROC-AUC", x=cmp["Experiment"],
                              y=cmp["ROC-AUC"], marker_color="#2196F3"))
        fig3.add_trace(go.Bar(name="PR-AUC", x=cmp["Experiment"],
                              y=cmp["PR-AUC"], marker_color="#FF9800"))
        fig3.update_layout(barmode="group", height=300,
                           xaxis_tickangle=-15,
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        section("Confusion Matrix — LightGBM BGFAD")
        cm = data["cm"]
        fig4 = px.imshow(cm.values,
                         labels=dict(x="Predicted", y="Actual", color="Count"),
                         x=["Normal","Fraud"], y=["Normal","Fraud"],
                         color_continuous_scale="Blues",
                         text_auto=True)
        fig4.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)

# ================================================================
# PAGE: LIVE PREDICTION
# ================================================================
elif page == "🎯 Live Prediction":
    st.title("🎯 Live Transaction Risk Prediction")
    st.markdown("Fill in the **real transaction details** — all scores are computed automatically.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔢 Manual Input", "📂 Batch from CSV"])

    with tab1:
        st.markdown("### Transaction Details")
        st.caption("Enter the raw transaction information below. BCS, GTS, AAS and FusionScore will be computed automatically.")

        c1, c2, c3 = st.columns(3)
        with c1:
            section("Transaction Info")
            txn_amount   = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=150.0, step=0.01)
            txn_hour     = st.slider("Transaction Hour (0–23)", 0, 23, 14)
            card_id      = st.number_input("Card ID (numeric)", min_value=1, max_value=20000, value=5000)
            prev_txn_gap = st.number_input("Time Since Last Transaction (seconds)", min_value=0, max_value=86400, value=3600)

        with c2:
            section("User & Device Info")
            past_txn_count  = st.number_input("Past Transactions on this Card", min_value=1, max_value=500, value=20)
            past_fraud_count= st.number_input("Past Fraud Incidents on this Card", min_value=0, max_value=50, value=0)
            device_type     = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "Unknown"])
            browser         = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Edge", "Unknown"])

        with c3:
            section("Network Info")
            ip_txn_count    = st.number_input("Transactions from this IP", min_value=1, max_value=1000, value=5)
            merchant_risk   = st.selectbox("Merchant Category Risk", ["Low", "Medium", "High"])
            email_match     = st.selectbox("Billing Email Matches Account?", ["Yes", "No"])
            addr_match      = st.selectbox("Billing Address Matches?", ["Yes", "Partial", "No"])

        st.markdown("---")

        if st.button("🔍 Analyse & Predict Risk", type="primary", use_container_width=True):
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import train_test_split

            # --------------------------------------------------
            # Step 1: Compute raw features
            # --------------------------------------------------
            max_amt = 50000.0
            txn_risk = round(min(txn_amount / max_amt, 1.0), 4)

            session_risk = 1 if txn_hour <= 5 else 0

            login_freq = past_txn_count

            user_reputation = round(1.0 - (past_fraud_count / max(past_txn_count, 1)), 4)
            user_reputation = max(0.0, min(1.0, user_reputation))

            device_map = {"Mobile": 0.8, "Desktop": 0.9, "Tablet": 0.7, "Unknown": 0.4}
            device_trust = device_map[device_type]

            browser_map = {"Chrome": 0.9, "Firefox": 0.85, "Safari": 0.85, "Edge": 0.8, "Unknown": 0.3}
            browser_cons = browser_map[browser]

            identity_score = (1.0 if email_match == "Yes" else 0.5 if email_match == "No" else 0.3)
            addr_score     = (1.0 if addr_match  == "Yes" else 0.6 if addr_match  == "Partial" else 0.2)
            identity_score = round((identity_score + addr_score) / 2, 4)

            behavior_drift = round((txn_risk + session_risk + (1 - user_reputation)) / 3, 4)

            # --------------------------------------------------
            # Step 2: Compute BCS
            # --------------------------------------------------
            behavior_sim = round((login_freq / 500 + 0.5 + (1 - behavior_drift)) / 3, 4)
            device_sim   = round((device_trust + browser_cons) / 2, 4)
            location_sim = round((identity_score + (1 - session_risk)) / 2, 4)

            def norm(v, lo=0.0, hi=1.0):
                return round(max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-8))), 4)

            bcs = round((norm(behavior_sim) + norm(device_sim) + norm(location_sim)) / 3, 4)

            # --------------------------------------------------
            # Step 3: Compute GTS
            # --------------------------------------------------
            initial_trust   = 0.90
            connectivity    = round(min(ip_txn_count / 100.0, 1.0), 4)
            neighbor_trust  = round((user_reputation + device_trust) / 2, 4)
            gts = round(0.40 * initial_trust + 0.30 * connectivity + 0.30 * neighbor_trust, 4)

            # --------------------------------------------------
            # Step 4: Compute AAS
            # --------------------------------------------------
            merchant_risk_map = {"Low": 0.1, "Medium": 0.4, "High": 0.8}
            m_risk = merchant_risk_map[merchant_risk]
            aas_raw = (behavior_drift + txn_risk + (1 - gts) + (1 - user_reputation)) / 4
            aas = round(min(max(aas_raw, 0.0), 1.0), 4)

            # --------------------------------------------------
            # Step 5: Compute FusionScore
            # --------------------------------------------------
            fusion_score = round(0.40 * bcs + 0.35 * gts + 0.25 * (1 - aas), 4)

            # --------------------------------------------------
            # Step 6: Show computed scores
            # --------------------------------------------------
            st.markdown("### Computed BGFAD Scores")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("BCS", f"{bcs:.4f}",
                          help="Behavioral Consistency Score")
            with s2:
                st.metric("GTS", f"{gts:.4f}",
                          help="Graph Trust Score")
            with s3:
                st.metric("AAS", f"{aas:.4f}",
                          help="Adaptive Anomaly Score")
            with s4:
                st.metric("Fusion Score", f"{fusion_score:.4f}",
                          help="Context Fusion Score")

            # --------------------------------------------------
            # Step 7: Train model and predict
            # --------------------------------------------------
            fusion_df = data["fusion"]
            features  = ["BCS","GraphTrustScore","AAS","FusionScore",
                         "LoginFrequency","TransactionRisk","UserReputation",
                         "DeviceTrustScore","BrowserConsistency"]
            X = fusion_df[features]
            y = fusion_df["isFraud"]
            X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
            neg = (y_tr == 0).sum()
            pos = (y_tr == 1).sum()
            spw = round(neg / pos, 2)
            model = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   max_depth=8, scale_pos_weight=spw,
                                   random_state=42, verbose=-1)
            model.fit(X_tr, y_tr)

            inp = pd.DataFrame([[bcs, gts, aas, fusion_score, login_freq,
                                  txn_risk, user_reputation, device_trust, browser_cons]],
                               columns=features)
            prob  = model.predict_proba(inp)[0][1]
            score = round(prob * 100, 2)

            if score < 30:
                decision, color, icon = "Safe", "#27ae60", "✅"
            elif score < 60:
                decision, color, icon = "OTP Verification", "#f39c12", "📱"
            elif score < 80:
                decision, color, icon = "Biometric Verification", "#e67e22", "🔐"
            else:
                decision, color, icon = "Blocked", "#c0392b", "🚫"

            st.markdown("---")
            st.markdown("### Prediction Result")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Fraud Probability", f"{prob:.4f}")
            with r2:
                st.metric("Risk Score", f"{score} / 100")
            with r3:
                st.markdown(
                    f"<div style='background:{color};color:white;padding:20px;"
                    f"border-radius:12px;text-align:center;font-size:1.4rem;"
                    f"font-weight:700;margin-top:8px'>{icon} {decision}</div>",
                    unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  30], "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fef9e7"},
                        {"range": [60, 80], "color": "#fdebd0"},
                        {"range": [80,100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4},
                                  "thickness": 0.75, "value": score}
                }
            ))
            fig.update_layout(height=340, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Score breakdown bar
            st.markdown("### Score Breakdown")
            breakdown = pd.DataFrame({
                "Score": ["BCS", "GTS", "AAS", "FusionScore", "TransactionRisk", "UserReputation"],
                "Value": [bcs,   gts,   aas,   fusion_score,  txn_risk,          user_reputation]
            })
            fig2 = px.bar(breakdown, x="Score", y="Value",
                          color="Value", color_continuous_scale="RdYlGn_r",
                          range_y=[0, 1])
            fig2.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section("Batch Predictions from final_predictions.csv")
        pred_df = data["predictions"]
        st.dataframe(pred_df.head(200), use_container_width=True, hide_index=True)

        decision_counts = pred_df["Decision"].value_counts().reset_index()
        decision_counts.columns = ["Decision", "Count"]
        color_map = {
            "Safe": "#27ae60",
            "OTP Verification": "#f39c12",
            "Biometric Verification": "#e67e22",
            "Blocked": "#c0392b"
        }
        fig = px.bar(decision_counts, x="Decision", y="Count",
                     color="Decision", color_discrete_map=color_map,
                     title="Risk Decision Distribution")
        fig.update_layout(height=320, showlegend=False,
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# PAGE: SELF-LEARNING
# ================================================================
elif page == "📈 Self-Learning":
    st.title("📈 Self-Learning Module")
    st.markdown("Incremental online learning with analyst feedback loop.")
    st.markdown("---")

    sl_r = data["sl_results"].set_index("Metric")["Value"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Final Accuracy",  f"{sl_r['Accuracy']:.4f}",  "info-card")
    with c2: card("Final Recall",    f"{sl_r['Recall']:.4f}",    "safe-card")
    with c3: card("Feedback Samples",f"{int(sl_r['Feedback Samples']):,}", "warn-card")
    with c4: card("Total Batches",   f"{int(sl_r['Total Batches'])}", "info-card")

    st.markdown("---")
    curve = data["sl_curve"]

    c1, c2 = st.columns(2)
    with c1:
        section("Learning Curve — Accuracy & F1")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["Batch"], y=curve["Accuracy"],
                                 name="Accuracy", line=dict(color="#2196F3", width=2)))
        fig.add_trace(go.Scatter(x=curve["Batch"], y=curve["F1 Score"],
                                 name="F1 Score", line=dict(color="#4CAF50", width=2)))
        fig.update_layout(height=320, xaxis_title="Batch",
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Learning Curve — Precision & Recall")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=curve["Batch"], y=curve["Precision"],
                                  name="Precision", line=dict(color="#FF9800", width=2)))
        fig2.add_trace(go.Scatter(x=curve["Batch"], y=curve["Recall"],
                                  name="Recall", line=dict(color="#F44336", width=2)))
        fig2.update_layout(height=320, xaxis_title="Batch",
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    section("How the Feedback Loop Works")
    st.markdown("""
    1. **Incremental Training** — SGDClassifier trained via `partial_fit` across **43 mini-batches** (500 samples each)
    2. **Feedback Trigger** — Transactions with fraud probability ≥ 0.70 are flagged for analyst review
    3. **Retraining** — **2,069 confirmed samples** are fed back into the model via another `partial_fit` call
    4. **Adaptation** — Model continuously adapts to new fraud patterns without full retraining
    """)

# ================================================================
# PAGE: EXPLAINABLE AI
# ================================================================
elif page == "💡 Explainable AI":
    st.title("💡 Explainable AI — SHAP Analysis")
    st.markdown("SHAP (SHapley Additive exPlanations) values explain each feature's contribution.")
    st.markdown("---")

    imp = data["importance"]

    c1, c2 = st.columns(2)

    with c1:
        section("Feature Importance — Mean |SHAP|")
        fig = px.bar(imp, x="SHAP_Mean", y="Feature", orientation="h",
                     color="SHAP_Mean", color_continuous_scale="Blues",
                     labels={"SHAP_Mean": "Mean |SHAP| Value"})
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
                          coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("LightGBM Split Gain Importance")
        fig2 = px.bar(imp, x="LGB_Importance", y="Feature", orientation="h",
                      color="LGB_Importance", color_continuous_scale="Greens",
                      labels={"LGB_Importance": "Split Gain"})
        fig2.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
                           coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    section("SHAP Summary Plot")
    shap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results/graphs/shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, use_column_width=True)

    section("Feature Importance Table")
    st.dataframe(imp, use_container_width=True, hide_index=True)

    section("Interpretation")
    st.markdown("""
    | Feature | Interpretation |
    |---|---|
    | **UserReputation** | Historical fraud rate per card — highest SHAP impact (3.61) |
    | **BCS** | Behavioral consistency — second most important (1.11) |
    | **AAS** | Anomaly score — highest LightGBM split gain (2005) |
    | **FusionScore** | Combined signal — strong split gain (1746) |
    | **LoginFrequency** | Transaction count per card — moderate impact |
    | **TransactionRisk** | Normalized amount — moderate impact |
    | **GraphTrustScore** | Network trust — contributes via graph structure |
    | **DeviceTrustScore** | Zero impact — collapsed to single value post-IQR |
    | **BrowserConsistency** | Zero impact — collapsed to single value post-IQR |
    """)
