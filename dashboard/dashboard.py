"""
Streamlit Visualization Dashboard for ML Bot Detection System
Interactive dashboard for live detection, model metrics, and data exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="ML Bot Detection Dashboard",
    page_icon="Bot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ML Bot Detection Dashboard")
st.markdown("Real-time bot vs. human traffic classification powered by machine learning.")

# Sidebar sliders
st.sidebar.header("Live Prediction")
request_rate = st.sidebar.slider("Request Rate (req/s)", 0.0, 100.0, 5.0)
request_interval_std = st.sidebar.slider("Request Interval Std", 0.0, 15.0, 2.0)
ua_bot_score = st.sidebar.slider("UA Bot Score", 0.0, 1.0, 0.1)
ua_entropy = st.sidebar.slider("UA Entropy", 0.0, 6.0, 4.0)
mouse_movement_score = st.sidebar.slider("Mouse Movement Score", 0.0, 1.0, 0.8)
click_interval_std = st.sidebar.slider("Click Interval Std", 0.0, 5.0, 1.0)
ip_request_count = st.sidebar.slider("IP Request Count", 1.0, 2000.0, 30.0)
ip_unique_paths = st.sidebar.slider("IP Unique Paths", 1.0, 100.0, 15.0)
session_duration = st.sidebar.slider("Session Duration (s)", 1.0, 1800.0, 300.0)
error_rate = st.sidebar.slider("Error Rate", 0.0, 1.0, 0.02)
payload_size_avg = st.sidebar.slider("Avg Payload Size (bytes)", 64.0, 10000.0, 2000.0)
time_of_day = st.sidebar.slider("Time of Day", 0, 23, 14)

features = {
    "request_rate": request_rate,
    "request_interval_std": request_interval_std,
    "ua_bot_score": ua_bot_score,
    "ua_entropy": ua_entropy,
    "mouse_movement_score": mouse_movement_score,
    "click_interval_std": click_interval_std,
    "ip_request_count": ip_request_count,
    "ip_unique_paths": ip_unique_paths,
    "session_duration": session_duration,
    "error_rate": error_rate,
    "payload_size_avg": payload_size_avg,
    "time_of_day": float(time_of_day),
}


@st.cache_resource
def load_detector():
    try:
        from src.inference import BotDetector
        return BotDetector(), None
    except Exception as e:
        return None, str(e)


detector, load_error = load_detector()

if detector:
    result = detector.predict_single(features)
    label = result["classification"]
    prob = result["bot_probability"]
    st.sidebar.markdown("---")
    st.sidebar.metric("Classification", label.upper())
    st.sidebar.metric("Bot Probability", f"{prob:.4f}")
    st.sidebar.metric("Confidence", result["confidence"].upper())
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Bot Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if label == "bot" else "green"},
            "steps": [
                {"range": [0, 40], "color": "lightgreen"},
                {"range": [40, 60], "color": "lightyellow"},
                {"range": [60, 100], "color": "lightcoral"},
            ],
        },
    ))
    gauge.update_layout(height=250, margin=dict(t=40, b=0, l=0, r=0))
    st.sidebar.plotly_chart(gauge, use_container_width=True)
else:
    st.sidebar.warning(f"Model not loaded: {load_error}")
    st.sidebar.info("Run: python src/train.py")

tab1, tab2, tab3 = st.tabs(["Model Performance", "Data Explorer", "Simulation"])

with tab1:
    st.header("Model Performance Metrics")
    results_path = Path("models/training_results.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        rows = []
        for name, res in results.items():
            m = res.get("test_metrics", {})
            rows.append({
                "Model": name.replace("_", " ").title(),
                "Accuracy": round(m.get("accuracy", 0), 4),
                "Precision": round(m.get("precision", 0), 4),
                "Recall": round(m.get("recall", 0), 4),
                "F1-Score": round(m.get("f1", 0), 4),
                "ROC-AUC": round(m.get("roc_auc", 0), 4),
            })
        df_results = pd.DataFrame(rows)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(df_results.set_index("Model"), use_container_width=True)
        with col2:
            fig = px.bar(
                df_results.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Model", y="Score", color="Metric", barmode="group",
                title="Model Comparison",
            )
            fig.update_layout(yaxis_range=[0.7, 1.0])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training results found. Run python src/train.py first.")

with tab2:
    st.header("Dataset Explorer")
    data_path = Path("data/sample_data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        df["label_str"] = df["label"].map({1: "Bot", 0: "Human"})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", f"{len(df):,}")
        col2.metric("Bot Samples", f"{df['label'].sum():,}")
        col3.metric("Human Samples", f"{(1 - df['label']).sum():,}")
        col4.metric("Features", len(df.columns) - 2)
        feature_options = [c for c in df.columns if c not in ["label", "label_str"]]
        selected_feature = st.selectbox("Select Feature", feature_options)
        fig = px.histogram(
            df, x=selected_feature, color="label_str",
            barmode="overlay", nbins=50,
            title=f"Distribution of {selected_feature} by Label",
            color_discrete_map={"Bot": "red", "Human": "green"},
        )
        st.plotly_chart(fig, use_container_width=True)
        corr = df[feature_options].corr()
        fig_heat = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No dataset found. Run python data/generate_data.py first.")

with tab3:
    st.header("Real-Time Detection Simulation")
    n_simulate = st.slider("Number of requests", 10, 200, 50)
    bot_ratio = st.slider("Expected bot ratio", 0.0, 1.0, 0.4)
    if st.button("Run Simulation") and detector:
        from data.generate_data import generate_bot_features, generate_human_features
        n_bots = int(n_simulate * bot_ratio)
        n_humans = n_simulate - n_bots
        bot_df = pd.DataFrame(generate_bot_features(n_bots))
        human_df = pd.DataFrame(generate_human_features(n_humans))
        sim_df = pd.concat([bot_df, human_df]).sample(frac=1).reset_index(drop=True)
        true_labels = sim_df.pop("label")
        results_list = detector.predict(sim_df)
        sim_df["bot_probability"] = [r["bot_probability"] for r in results_list]
        sim_df["prediction"] = [r["classification"] for r in results_list]
        sim_df["true_label"] = true_labels.map({1: "bot", 0: "human"})
        sim_df["correct"] = sim_df["prediction"] == sim_df["true_label"]
        accuracy = sim_df["correct"].mean()
        st.metric("Simulation Accuracy", f"{accuracy:.2%}")
        fig = px.scatter(
            sim_df.reset_index(), x="index", y="bot_probability",
            color="prediction", title="Request Stream - Bot Probabilities",
            color_discrete_map={"bot": "red", "human": "green"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
    elif not detector:
        st.warning("Model not loaded - run python src/train.py first")
