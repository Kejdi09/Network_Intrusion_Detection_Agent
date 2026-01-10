import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Page config ---
st.set_page_config(
    page_title="Network Intrusion Detection AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("⚡ Network Intrusion Detection AI")
st.sidebar.markdown("""
Select a trained model, upload your dataset, and visualize predictions with metrics and charts.
""")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "Stage 1 RF": MODELS_DIR / "stage1_rf.pkl",
    "Stage 1 XGB": MODELS_DIR / "stage1_xgb.pkl",
    "Stage 2 XGB": MODELS_DIR / "stage2_xgb.pkl"
}

models = {}
for name, path in MODEL_PATHS.items():
    try:
        models[name] = joblib.load(path)
    except Exception as e:
        st.sidebar.warning(f"Failed to load {name}: {e}")

# --- Model selection ---
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
if model_choice in models:
    model_data = models[model_choice]
    if isinstance(model_data, tuple):
        model, encoder = model_data
    else:
        model = model_data
        encoder = None
else:
    st.stop()

# --- Upload CSV ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Clean features ---
    FEATURES = [
        'L4_SRC_PORT','L4_DST_PORT','PROTOCOL','L7_PROTO',
        'IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS',
        'TCP_FLAGS','CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS',
        'FLOW_DURATION_MILLISECONDS','DURATION_IN',
        'MIN_TTL','MAX_TTL','LONGEST_FLOW_PKT','SHORTEST_FLOW_PKT',
        'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN',
        'SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES',
        'SRC_TO_DST_AVG_THROUGHPUT','DST_TO_SRC_AVG_THROUGHPUT',
        'NUM_PKTS_UP_TO_128_BYTES',
        'TCP_WIN_MAX_IN','TCP_WIN_MAX_OUT',
        'ICMP_TYPE','ICMP_IPV4_TYPE',
        'DNS_QUERY_ID','DNS_QUERY_TYPE','DNS_TTL_ANSWER',
        'FTP_COMMAND_RET_CODE'
    ]
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e9, 1e9).astype(np.float32)

    # --- Predict ---
    preds = model.predict(X)
    if encoder is not None:
        preds = encoder.inverse_transform(preds)
    df["Prediction"] = preds

    # --- Metrics cards ---
    total = len(df)
    attacks = np.sum(df["Prediction"] != 0)
    benign = total - attacks
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", total, delta_color="off")
    col2.metric("Benign", benign, delta=f"{(benign/total*100):.2f}%", delta_color="normal")
    col3.metric("Attacks Detected", attacks, delta=f"{(attacks/total*100):.2f}%", delta_color="inverse")

    # --- Feature importance (if XGB or RF) ---
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(20)

        st.markdown("### 🔑 Top 20 Feature Importances")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h", color="Importance",
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

    # --- Confusion matrix ---
    if "Label" in df.columns:
        cm = confusion_matrix(df["Label"], df["Prediction"])
        fig, ax = plt.subplots(figsize=(5,5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Oranges, ax=ax)
        st.pyplot(fig)

    # --- Predictions table with styling ---
    st.markdown("### 📊 Predictions")
    st.dataframe(df.style.background_gradient(subset=["Prediction"], cmap="RdYlGn"), height=400)

    # --- Download button ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True
    )
