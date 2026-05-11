"""
Stage 2 — Feature Extraction
Reads raw_samples.csv and builds a feature matrix using a sliding window.
Each window of N samples becomes one training row.

Run:  python features.py
Output: data/features.csv
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
RAW_FILE     = os.path.join("data", "raw_samples.csv")
FEAT_FILE    = os.path.join("data", "features.csv")
WINDOW_SIZE  = 20    # samples per window (at 2 Hz → 10 seconds of context)
STEP_SIZE    = 5     # slide step (overlap windows for more training data)

FEATURE_COLS = [
    "rssi_mean", "rssi_std", "rssi_range", "rssi_roc", "rssi_peak_cnt",
    "lat_mean",  "lat_std",  "lat_max",    "lat_roc",  "lat_peak_cnt",
    "rssi_lat_corr",
]
# ────────────────────────────────────────────────────────────────────────────


def extract_window_features(rssi_win: np.ndarray, lat_win: np.ndarray) -> dict:
    """
    Given a window of RSSI and latency values, compute statistical features.
    These features are what the ML model actually sees.
    """

    # ── RSSI features ──
    rssi_clean = rssi_win[rssi_win > 0]   # drop -1 error readings
    rssi_mean  = rssi_clean.mean()  if len(rssi_clean) else 0
    rssi_std   = rssi_clean.std()   if len(rssi_clean) else 0
    rssi_min   = rssi_clean.min()   if len(rssi_clean) else 0
    rssi_max   = rssi_clean.max()   if len(rssi_clean) else 0
    rssi_range = rssi_max - rssi_min

    # Rate of change: how fast is signal fluctuating?
    rssi_roc   = np.abs(np.diff(rssi_clean)).mean() if len(rssi_clean) > 1 else 0

    # Peak count: how many local spikes/dips? (indicates movement bursts)
    rssi_peaks, _ = find_peaks(rssi_clean, prominence=2)
    rssi_peak_cnt = len(rssi_peaks)

    # ── Latency features ──
    lat_clean  = lat_win[lat_win > 0]     # drop -1 error readings
    lat_mean   = lat_clean.mean() if len(lat_clean) else 0
    lat_std    = lat_clean.std()  if len(lat_clean) else 0
    lat_max    = lat_clean.max()  if len(lat_clean) else 0
    lat_roc    = np.abs(np.diff(lat_clean)).mean() if len(lat_clean) > 1 else 0

    lat_peaks, _ = find_peaks(lat_clean, prominence=3)
    lat_peak_cnt = len(lat_peaks)

    # ── Cross-signal features ──
    # Correlation between RSSI and latency fluctuation
    if len(rssi_clean) > 3 and len(lat_clean) > 3:
        min_len  = min(len(rssi_clean), len(lat_clean))
        corr_val = np.corrcoef(rssi_clean[:min_len], lat_clean[:min_len])[0, 1]
        if np.isnan(corr_val):
            corr_val = 0.0
    else:
        corr_val = 0.0

    return {
        "rssi_mean":      round(rssi_mean,  4),
        "rssi_std":       round(rssi_std,   4),
        "rssi_range":     round(rssi_range, 4),
        "rssi_roc":       round(rssi_roc,   4),
        "rssi_peak_cnt":  rssi_peak_cnt,
        "lat_mean":       round(lat_mean,   4),
        "lat_std":        round(lat_std,    4),
        "lat_max":        round(lat_max,    4),
        "lat_roc":        round(lat_roc,    4),
        "lat_peak_cnt":   lat_peak_cnt,
        "rssi_lat_corr":  round(corr_val,   4),
    }


def build_features():
    print(f"[INFO] Reading {RAW_FILE} ...")
    df = pd.read_csv(RAW_FILE)

    print(f"[INFO] Total raw samples : {len(df)}")
    print(f"[INFO] Window size       : {WINDOW_SIZE}  Step: {STEP_SIZE}")

    rssi_arr  = df["rssi_pct"].values.astype(float)
    lat_arr   = df["latency_ms"].values.astype(float)
    label_arr = df["label"].values.astype(int)

    rows = []
    n = len(df)

    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE

        rssi_win  = rssi_arr[start:end]
        lat_win   = lat_arr[start:end]
        label_win = label_arr[start:end]

        # Majority vote label for the window
        label = int(np.round(label_win.mean()))

        feats = extract_window_features(rssi_win, lat_win)
        feats["label"] = label
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    feat_df.to_csv(FEAT_FILE, index=False)

    pos = (feat_df["label"] == 1).sum()
    neg = (feat_df["label"] == 0).sum()
    print(f"[DONE] {len(feat_df)} windows  →  {FEAT_FILE}")
    print(f"       Present (1): {pos}  |  Empty (0): {neg}")

    if min(pos, neg) < 10:
        print("[WARN] Very few samples in one class. Collect more data first!")

    return feat_df


if __name__ == "__main__":
    build_features()