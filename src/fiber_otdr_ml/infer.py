from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from .preprocess import denoise_savgol
from .changepoint import simple_break_detector, ruptures_pelt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: distance_km,power_db")
    ap.add_argument("--use_ruptures", action="store_true", help="Try ruptures PELT as well")
    ap.add_argument("--plt", action="store_true", help="Show matplotlib plot")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    x = df["distance_km"].values
    y = df["power_db"].values

    idx, grad = simple_break_detector(x, y, grad_thresh=-6.0)
    result = {"method": "gradient_threshold", "break_km": None, "gradient_db_per_km": grad}
    if idx is not None:
        result["break_km"] = float(x[idx])

    extra = {}
    if args.use_ruptures:
        bkpts = ruptures_pelt(x, denoise_savgol(y, 41, 3), penalty=8.0)
        extra["ruptures_indices"] = bkpts
        extra["ruptures_km"] = [float(x[i]) for i in bkpts]

    print(json.dumps({"result": result, "extra": extra}, indent=2))

    if args.plt:
        import matplotlib.pyplot as plt
        ys = denoise_savgol(y, 41, 3)
        plt.figure()
        plt.plot(x, y, alpha=0.3, label="raw")
        plt.plot(x, ys, label="smoothed")
        if result["break_km"] is not None:
            bk = result["break_km"]
            plt.axvline(bk, color="r", linestyle="--", label=f"break ~{bk:.2f} km")
        if args.use_ruptures and extra.get("ruptures_km"):
            for km in extra["ruptures_km"]:
                plt.axvline(km, color="orange", linestyle=":")
        plt.xlabel("Distance (km)"); plt.ylabel("Power (dB)"); plt.legend(); plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
