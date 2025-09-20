![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.x-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


# Fiber OTDR ML

From manual OTDR reading → to **AI-powered fiber cut detection** and self-healing automation.  

This project demonstrates how OTDR traces can be reframed as **time-series anomaly detection** instead of manual noisy-trace interpretation. The system detects sharp loss events, visualizes them, and simulates **closed-loop reroute** with SDN controllers.  

---

## 🚀 Features
- **Data ingestion**: Load OTDR CSV (distance_km, power_db).
- **Preprocessing**: Smoothing with Savitzky–Golay / rolling mean.
- **Anomaly detection**: Change-point by slope threshold.
- **Interactive dashboard** (Streamlit):
  - Upload CSV or use sample trace.
  - Adjustable **slope threshold** & **smoothing window**.
  - **Plotly trace visualization** with event markers.
  - **Download detected events** (CSV).
- **Closed-loop simulation**: Auto-reroute trigger via SDN controller when severe event is detected.

---

## 🖥️ Demo (screenshot)

![Dashboard Screenshot](docs/demo_screenshot.png)

> Example: Detected cut at ~5 km, auto-trigger reroute to alternate path.

---

## 📂 Project Structure
fiber-otdr-ml/
│
├── app/
│ └── dashboard_demo.py # Streamlit dashboard
│
├── src/
│ ├── init.py
│ └── otdr_parser.py # Parser + smoothing + break detection
│
├── sample_data/
│ └── trace_sample.csv # Example OTDR trace
│
├── notebooks/
│ └── model_training.ipynb # 1D CNN skeleton (experimental)
│
├── requirements.txt
└── README.md


---

## ⚡ Quickstart
```bash
# clone repo
git clone https://github.com/Milzon1010/fiber-otdr-ml.git
cd fiber-otdr-ml

# create & activate venv (optional)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run dashboard
streamlit run app/dashboard_demo.py

📊 Sample Output

Detected Events Table
distance_km  power_db  slope
4.5          -14.3     -2.43
5.0          -18.0     -2.09
5.5          -18.2     -2.40
6.0          -18.4     -2.71

Break detected! Triggering reroute...
Calling SDN_Controller to reroute via alt_path ... ✅

🧠 Lessons Learned

Clean, structured data is the foundation — garbage in, garbage out.

Define the problem right: anomaly detection in time-series, not just plotting traces.

Strongest results often come from combining classical statistics with deep learning.

AI augments engineers — faster decisions, reduced downtime.

🛠️ Next Deploy

Geo-enabled map integration (Streamlit + Pydeck/Folium):

Upload fiber route GeoJSON (LineString).

Project OTDR cut distance onto the map.

Show blinking red markers at detected cut points.

Tooltip with distance_km and loss details.

Export cut locations as GeoJSON for NOC systems.

Future extension: Connect to live OTDR + real SDN API for self-healing network.

📌 Roadmap

 Baseline slope-based detection (done)
 Streamlit interactive dashboard (done)
 GeoJSON route + blinking cut markers (next deploy)
 Real OTDR integration (vendor API) (next)
 Model training with 1D CNN for robust event classification (next)
 SDN API hook for live rerouting (next)