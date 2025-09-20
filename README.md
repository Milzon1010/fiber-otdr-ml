# Fiber OTDR ML

Real‑case starter project to **detect fiber cuts** from OTDR traces using a hybrid of
**change‑point statistics** and an optional **1D‑CNN** model. Includes a minimal
Streamlit dashboard to visualize traces and highlight detected break points.

<p align="center">
  <img src="assets/cheatsheet-preview.png" width="720"/>
</p>

## 🧭 What’s inside

```
fiber-otdr-ml/
├─ app/
│  └─ streamlit_app.py          # Dashboard (OTDR tab)
├─ sample_data/
│  └─ trace_sample.csv          # Synthetic trace with cut at ~5.2 km
├─ src/fiber_otdr_ml/
│  ├─ preprocess.py             # Denoise, normalize, derivative, windows
│  ├─ changepoint.py            # Simple change-point detection (ruptures optional)
│  ├─ model.py                  # 1D-CNN builder (TensorFlow)
│  └─ infer.py                  # End-to-end detection from CSV
├─ scripts/
│  └─ train_cnn1d.py            # Minimal training loop (toy example)
├─ models/                      # (place trained weights here)
├─ assets/                      # images for README
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

## ⚡ Quickstart

```bash
# 1) Create and activate virtual env (example: Linux/Mac)
python -m venv .venv && source .venv/bin/activate

# 2) Install deps (CPU)
pip install -r requirements.txt

# 3) Run quick inference on sample CSV
python -m src.fiber_otdr_ml.infer --csv sample_data/trace_sample.csv

# 4) Launch dashboard
streamlit run app/streamlit_app.py
```

You should see an **estimated break near 5.2 km** with ~18 dB loss on the sample trace.

## 🧪 Methods

- **Change‑point statistics**: gradient threshold + (optional) `ruptures` PELT.
- **CNN‑1D**: small ConvNet that learns event patterns (splice, connector, cut).  
  Use only if you have labeled windows; otherwise the stats detector works out‑of‑the‑box.

## 🧱 API (CLI)

```bash
python -m src.fiber_otdr_ml.infer --csv <path> [--use_ruptures] [--plt]
```

Outputs JSON with the top candidate break and prints a short summary.

## 📦 Dataset Hints

- Real OTDR exports are usually **CSV/TXT** distance vs power (dB).  
- If your device embeds units/headers, write a small parser to map columns to `{{distance_km, power_db}}`.

## 🗺 Roadmap

- Add LSTM baseline for comparison
- SDN controller integration (closed‑loop)
- TimescaleDB storage + Grafana panel
- Unit tests and CI

---

**Author:** @milzon1010  
**License:** MIT
