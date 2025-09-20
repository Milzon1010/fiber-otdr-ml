"""
OTDR trace utilities: load, smooth, and detect break points.
Usage:
    from src.otdr_parser import load_trace, detect_breaks
"""
from __future__ import annotations
import pandas as pd
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

def load_trace(path: str) -> pd.DataFrame:
    """Load CSV with columns: distance_km, power_db"""
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    if 'distance_km' not in cols or 'power_db' not in cols:
        # try common variants
        rename = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ('distance', 'distance(km)', 'km'):
                rename[c] = 'distance_km'
            if lc in ('power', 'power(db)', 'db', 'loss_db'):
                rename[c] = 'power_db'
        if rename:
            df = df.rename(columns=rename)
    # Final check
    assert 'distance_km' in df.columns and 'power_db' in df.columns, \
        "CSV must contain distance_km and power_db columns"
    df = df.dropna().sort_values('distance_km').reset_index(drop=True)
    return df

def smooth_power(power: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Smooth power curve with Savitzky–Golay if available, else rolling mean."""
    n = len(power)
    if n < window + 2:
        return power
    if _HAS_SCIPY and window % 2 == 1:  # savgol requires odd window
        return savgol_filter(power, window_length=window, polyorder=poly, mode="interp")
    # Fallback: centered rolling mean
    s = pd.Series(power).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return s

def detect_breaks(df: pd.DataFrame, slope_threshold: float = 2.0, window: int = 7) -> pd.DataFrame:
    """
    Simple change-point detection using first derivative threshold.
    slope_threshold: absolute d(power_db)/d(distance_km) threshold (dB per km) for a 'break'.
    Returns DataFrame with distance_km, power_db, slope columns for detected points.
    """
    x = df['distance_km'].to_numpy()
    y = smooth_power(df['power_db'].to_numpy(), window=window)
    # derivative (slope) as diff of smoothed power over diff distance
    dy = np.diff(y)
    dx = np.diff(x) + 1e-9
    slope = dy / dx
    # prepend first slope to align lengths
    slope = np.insert(slope, 0, slope[0])
    df_out = df.copy()
    df_out['slope'] = slope
    # sharp negative drop => slope << -threshold
    events = df_out[df_out['slope'] <= -slope_threshold][['distance_km','power_db','slope']]
    return events.reset_index(drop=True)
