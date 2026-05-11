"""
Stage 3 — Model Training
Trains two models on features.csv:
  1. Random Forest  — fast baseline, interpretable
  2. LSTM (PyTorch) — captures temporal patterns, uses GTX 1650 if available

Run:  python train.py
Output: models/rf_model.pkl  and  models/lstm_model.pt
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── CONFIG ──────────────────────────────────────────────────────────────────
FEAT_FILE   = os.path.join("data", "features.csv")
MODEL_DIR   = "models"
RF_PATH     = os.path.join(MODEL_DIR, "rf_model.pkl")
LSTM_PATH   = os.path.join(MODEL_DIR, "lstm_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# LSTM hyperparams
SEQ_LEN     = 10      # feed N consecutive feature vectors as a sequence
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-3
# ────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rssi_mean", "rssi_std", "rssi_range", "rssi_roc", "rssi_peak_cnt",
    "lat_mean",  "lat_std",  "lat_max",    "lat_roc",  "lat_peak_cnt",
    "rssi_lat_corr",
]


# ══ Random Forest ════════════════════════════════════════════════════════════

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n── Random Forest ──────────────────────────────")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",   # handles imbalanced data
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Empty", "Present"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = sorted(
        zip(FEATURE_COLS, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop features:")
    for name, imp in importances[:5]:
        bar = "█" * int(imp * 40)
        print(f"  {name:<20} {bar}  {imp:.3f}")

    return rf


# ══ LSTM Model ═══════════════════════════════════════════════════════════════

class PresenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)   # hn: [num_layers, batch, hidden]
        out = hn[-1]                 # last layer's hidden state
        return self.classifier(out).squeeze(1)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Stack feature rows into overlapping sequences for LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])   # label is the last sample's label
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(X_train, X_test, y_train, y_test):
    print("\n── LSTM (PyTorch) ─────────────────────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    if device.type == "cuda":
        print(f"   GPU   : {torch.cuda.get_device_name(0)}")

    # Build sequences
    X_seq_train, y_seq_train = make_sequences(X_train, y_train, SEQ_LEN)
    X_seq_test,  y_seq_test  = make_sequences(X_test,  y_test,  SEQ_LEN)

    if len(X_seq_train) < BATCH_SIZE:
        print("[WARN] Not enough data for LSTM training. Skipping.")
        return None

    train_ds = TensorDataset(
        torch.tensor(X_seq_train), torch.tensor(y_seq_train)
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = PresenceLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_seq_test).to(device)
            yt = torch.tensor(y_seq_test).to(device)
            val_preds = (model(Xt) > 0.5).float()
            acc = (val_preds == yt).float().mean().item()
            scheduler.step(total_loss)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), LSTM_PATH + ".best")

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/len(train_dl):.4f}  val_acc={acc:.3f}")

    # Load best weights
    model.load_state_dict(torch.load(LSTM_PATH + ".best"))
    print(f"\n   Best val accuracy: {best_acc:.3f}")
    torch.save(model.state_dict(), LSTM_PATH)
    return model


# ══ MAIN ═════════════════════════════════════════════════════════════════════

def main():
    print(f"[INFO] Loading {FEAT_FILE} ...")
    df = pd.read_csv(FEAT_FILE)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    print(f"[INFO] Samples: {len(df)}  |  Features: {len(FEATURE_COLS)}")
    print(f"[INFO] Class balance — 0: {(y==0).sum()}  1: {(y==1).sum()}")

    if len(df) < 50:
        print("[ERROR] Collect more data before training. Need at least 50 windows.")
        return

    # Normalize
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save scaler (needed for inference)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Scaler saved → {SCALER_PATH}")

    # ── Train RF ──
    rf = train_random_forest(X_train, X_test, y_train, y_test)
    with open(RF_PATH, "wb") as f:
        pickle.dump(rf, f)
    print(f"[DONE] RF model saved → {RF_PATH}")

    # ── Train LSTM ──
    train_lstm(X_train, X_test, y_train, y_test)
    if os.path.exists(LSTM_PATH):
        print(f"[DONE] LSTM model saved → {LSTM_PATH}")


if __name__ == "__main__":
    main()