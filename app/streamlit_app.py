import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.fiber_otdr_ml.preprocess import denoise_savgol
from src.fiber_otdr_ml.changepoint import simple_break_detector

st.set_page_config(page_title="Fiber OTDR ML", layout="wide")
st.title("Fiber Cut Detection (OTDR + ML)")

uploaded = st.file_uploader("Upload OTDR CSV (distance_km,power_db)", type=["csv"])
if uploaded is None:
    st.info("Using sample_data/trace_sample.csv")
    df = pd.read_csv("sample_data/trace_sample.csv")
else:
    df = pd.read_csv(uploaded)

x = df["distance_km"].values
y = df["power_db"].values
idx, grad = simple_break_detector(x, y, grad_thresh=-6.0)
bk_km = float(x[idx]) if idx is not None else None

col1, col2 = st.columns([3,1])
with col1:
    fig, ax = plt.subplots()
    ax.plot(x, y, alpha=0.35, label="raw")
    ax.plot(x, denoise_savgol(y, 41, 3), label="smoothed")
    if bk_km is not None:
        ax.axvline(bk_km, color="r", linestyle="--", label=f"break ~{bk_km:.2f} km")
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Power (dB)")
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)
with col2:
    st.subheader("Detection")
    st.metric("Gradient (dB/km)", f"{grad:.2f}")
    if bk_km is not None:
        st.success(f"Break detected near **{bk_km:.2f} km**")
    else:
        st.warning("No significant break detected")

st.caption("Prototype: gradient threshold + Savitzky-Golay smoothing. Add CNN/LSTM for advanced classification.")
