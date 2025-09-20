import sys
from pathlib import Path
import plotly.graph_objects as go

# --- make src importable ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from src.otdr_parser import load_trace, detect_breaks, smooth_power

st.set_page_config(page_title="Fiber OTDR ML – Demo", layout="wide")
st.title("Fiber Cut Detection – Demo")

# Sidebar controls
st.sidebar.header("Detection Params")
slope_th = st.sidebar.slider("Slope threshold (dB/km)", 0.5, 6.0, 2.0, 0.1)
smooth_w = st.sidebar.slider("Smoothing window (odd)", 3, 21, 7, 2)

# Data loader
uploaded = st.file_uploader("Upload OTDR CSV (distance_km,power_db)", type=["csv"])
if uploaded is None:
    st.info("Using sample_data/trace_sample.csv")
    df = load_trace("sample_data/trace_sample.csv")
else:
    df = pd.read_csv(uploaded)

# Detection
events = detect_breaks(df, slope_threshold=slope_th, window=smooth_w)

# Severity gate for automation
fiber_break_detected = len(events) > 0 and (
    (events["slope"] <= -3.0).any() or (events["power_db"] <= -16).any()
)

# Download results
if len(events):
    st.download_button(
        "Download detected events (CSV)",
        data=events.to_csv(index=False).encode(),
        file_name="detected_events.csv",
        mime="text/csv",
    )

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Trace")
    df_plot = df.copy()
    df_plot["power_smooth"] = smooth_power(df_plot["power_db"].to_numpy(), window=smooth_w)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot["distance_km"], y=df_plot["power_db"],
                             name="power_db", mode="lines"))
    fig.add_trace(go.Scatter(x=df_plot["distance_km"], y=df_plot["power_smooth"],
                             name="power_smooth", mode="lines"))

    # event markers
    for _, r in events.iterrows():
        fig.add_vline(x=r["distance_km"], line_dash="dot")
        fig.add_trace(go.Scatter(
            x=[r["distance_km"]], y=[r["power_db"]],
            mode="markers+text", text=[f"{r['distance_km']:.2f} km"],
            textposition="top center", name="event"
        ))

    fig.update_layout(
        xaxis_title="Distance (km)", yaxis_title="Power (dB)",
        height=420, legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Detected Events")
    if len(events) == 0:
        st.success("No sharp drops detected.")
    else:
        st.dataframe(events)

# Closed-loop action (simulated)
st.subheader("Closed-loop Automation (Simulated)")

def reroute_traffic(api: str, path: str):
    st.write(f"Calling **{api}** to reroute via `{path}` ... ✅")

if fiber_break_detected:
    st.warning("Break detected! Triggering reroute...")
    reroute_traffic(api="SDN_Controller", path="alt_path")
else:
    st.info("Network normal – no action taken.")
