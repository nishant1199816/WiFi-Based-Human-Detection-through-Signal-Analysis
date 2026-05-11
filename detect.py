"""
Stage 4 — Real-time Detection
Runs the trained Random Forest in real-time, prints a live dashboard.

Run:  python detect.py --router 192.168.1.1 --model rf
      python detect.py --router 192.168.1.1 --model lstm
"""

import subprocess, re, time, pickle, argparse, os, collections
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import torch

# Reuse feature extraction from features.py
from features import extract_window_features, WINDOW_SIZE, FEATURE_COLS

MODEL_DIR   = "models"
RF_PATH     = os.path.join(MODEL_DIR, "rf_model.pkl")
LSTM_PATH   = os.path.join(MODEL_DIR, "lstm_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ── same helpers as collect.py ────────────────────────────────────────────

def get_rssi() -> float:
    try:
        out = subprocess.check_output(
            ["netsh", "wlan", "show", "interfaces"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode("utf-8", errors="ignore")
        m = re.search(r"Signal\s*:\s*(\d+)\s*%", out)
        return float(m.group(1)) if m else -1.0
    except Exception:
        return -1.0


def get_latency(ip: str) -> float:
    try:
        out = subprocess.check_output(
            ["ping", "-n", "1", "-w", "800", ip],
            stderr=subprocess.DEVNULL, timeout=3
        ).decode("utf-8", errors="ignore")
        m = re.search(r"Average\s*=\s*(\d+)\s*ms", out)
        if not m:
            m = re.search(r"time[=<](\d+)ms", out)
        return float(m.group(1)) if m else -1.0
    except Exception:
        return -1.0

# ─────────────────────────────────────────────────────────────────────────────

def load_models(model_type: str):
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    if model_type == "rf":
        with open(RF_PATH, "rb") as f:
            model = pickle.load(f)
        return model, scaler, "rf"

    elif model_type == "lstm":
        from train import PresenceLSTM, SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = PresenceLSTM(len(FEATURE_COLS), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
        m.load_state_dict(torch.load(LSTM_PATH, map_location=device))
        m.eval()
        return (m, device), scaler, "lstm"

    raise ValueError(f"Unknown model type: {model_type}")


def predict_rf(model, scaler, window_rssi, window_lat):
    feats = extract_window_features(
        np.array(window_rssi), np.array(window_lat)
    )
    x = np.array([[feats[c] for c in FEATURE_COLS]])
    x = scaler.transform(x)
    prob  = model.predict_proba(x)[0][1]
    label = int(prob > 0.5)
    return label, prob


def predict_lstm(model_tuple, scaler, feat_history):
    """feat_history: deque of feature dicts, length = SEQ_LEN"""
    from train import SEQ_LEN
    if len(feat_history) < SEQ_LEN:
        return 0, 0.0
    model, device = model_tuple
    seq = np.array([[f[c] for c in FEATURE_COLS] for f in feat_history], dtype=np.float32)
    seq = scaler.transform(seq)
    x   = torch.tensor(seq).unsqueeze(0).to(device)   # [1, SEQ_LEN, features]
    with torch.no_grad():
        prob = model(x).item()
    return int(prob > 0.5), prob


def draw_dashboard(label: int, prob: float, rssi: float, lat: float, n: int):
    """Simple terminal dashboard."""
    bar_len = 30
    filled  = int(prob * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)

    status  = "👤 PERSON PRESENT" if label == 1 else "🔲 ROOM EMPTY    "
    colour  = "\033[92m" if label == 1 else "\033[90m"
    reset   = "\033[0m"

    print(f"\r{colour}{status}{reset}  "
          f"[{bar}] {prob*100:5.1f}%  "
          f"RSSI={rssi:5.1f}%  Lat={lat:5.1f}ms  "
          f"n={n:04d}", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="192.168.1.1")
    parser.add_argument("--model",  default="rf", choices=["rf", "lstm"])
    args = parser.parse_args()

    print(f"[INFO] Loading {args.model.upper()} model ...")
    model, scaler, model_type = load_models(args.model)
    print(f"[INFO] Model loaded. Starting real-time detection ...\n")

    rssi_buf  = collections.deque(maxlen=WINDOW_SIZE)
    lat_buf   = collections.deque(maxlen=WINDOW_SIZE)

    # For LSTM: keep a rolling history of extracted feature dicts
    from train import SEQ_LEN
    feat_history = collections.deque(maxlen=SEQ_LEN)

    sample_n = 0
    try:
        while True:
            t0 = time.time()

            rssi = get_rssi()
            lat  = get_latency(args.router)
            rssi_buf.append(rssi)
            lat_buf.append(lat)
            sample_n += 1

            if len(rssi_buf) == WINDOW_SIZE:
                if model_type == "rf":
                    label, prob = predict_rf(model, scaler, list(rssi_buf), list(lat_buf))
                else:
                    feats = extract_window_features(
                        np.array(list(rssi_buf)), np.array(list(lat_buf))
                    )
                    feat_history.append(feats)
                    label, prob = predict_lstm(model, scaler, list(feat_history))

                draw_dashboard(label, prob, rssi, lat, sample_n)

            elapsed = time.time() - t0
            time.sleep(max(0, 0.5 - elapsed))   # 2 Hz

    except KeyboardInterrupt:
        print("\n[DONE] Detection stopped.")


if __name__ == "__main__":
    main()