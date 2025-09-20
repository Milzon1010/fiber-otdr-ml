from __future__ import annotations
import numpy as np
from .preprocess import denoise_savgol, derivative

def simple_break_detector(distance_km: np.ndarray, power_db: np.ndarray, grad_thresh: float = -6.0):
    """
    Return (index, gradient) of strongest negative slope if steeper than grad_thresh (dB/km),
    else (None, min_gradient).
    """
    y_smooth = denoise_savgol(power_db, window=41, poly=3)
    grad = derivative(distance_km, y_smooth)
    idx = int(np.argmin(grad))
    return (idx, float(grad[idx])) if grad[idx] < grad_thresh else (None, float(grad[idx]))

def ruptures_pelt(distance_km: np.ndarray, power_db: np.ndarray, penalty: float = 10.0):
    """Optional change-point detection using ruptures PELT model; returns list of indices."""
    try:
        import ruptures as rpt
    except Exception:
        return []
    signal = power_db.reshape(-1,1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    bkpts = algo.predict(pen=penalty)
    return [b-1 for b in bkpts]  # convert to index
