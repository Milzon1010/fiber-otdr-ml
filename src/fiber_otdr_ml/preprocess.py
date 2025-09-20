from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

def denoise_savgol(y: np.ndarray, window: int = 31, poly: int = 3) -> np.ndarray:
    """Savitzky–Golay smoothing for noisy OTDR traces."""
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window, poly, mode="interp")

def normalize_db(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return (y - np.median(y)) / (np.std(y) + 1e-8)

def derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Numerical derivative dy/dx, robust to uneven spacing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.gradient(x)
    dy = np.gradient(y)
    return dy / (dx + 1e-12)
