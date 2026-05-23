# 📡 WiFi-Based-Human-Detection-through-Signal-Analysis

> Detect human presence in a room using **only WiFi signals** — no cameras, no special sensors, no extra hardware.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red?logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Windows-informational?logo=windows)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

---

## 🧠 What is this?

This project detects whether a **human is present** in a room by analyzing fluctuations in WiFi signal patterns — caused by the human body absorbing and reflecting radio frequency signals.

When a person is in a room, their body subtly disturbs the WiFi environment. These disturbances show up as microscopic changes in **signal strength (RSSI)** and **ping latency**. This system captures those changes, extracts statistical patterns, and uses **Machine Learning** to classify them in real time.

Inspired by research systems like [MIT WiTrack](http://witrack.csail.mit.edu/) and [Meta WiFi DensePose](https://ai.meta.com/blog/wifi-human-sensing/) — but built entirely with **consumer hardware and free software**.

---

## ✅ What it can do

| Capability | Status |
|---|---|
| Human presence detection (present / absent) | ✅ Working |
| Real-time inference with live terminal dashboard | ✅ Working |
| Auto-labeling via webcam during data collection | ✅ Working |
| GPU-accelerated LSTM training (CUDA) | ✅ Working |
| Through-wall detection | ⏳ Future (needs CSI hardware) |
| Activity recognition (walking, sitting) | ⏳ Future |

---

## 🛠️ Hardware Used

| Component | Details |
|---|---|
| Laptop | ASUS TUF F15 — GTX 1650, 8GB RAM |
| Router | Any standard home WiFi router |
| Webcam | Built-in laptop webcam (for labeling only) |
| Special sensors | ❌ None |

> **Zero extra cost.** If you have a laptop and a WiFi router, you can run this.

---

## 🔬 How It Works

```
Router ──ping──► Laptop
                    │
             RSSI + Latency
             sampled at 2 Hz
                    │
           Sliding window
           (last 20 samples)
                    │
        11 statistical features
        (std, variance, peaks, ROC...)
                    │
         Random Forest / LSTM
                    │
        "👤 PERSON PRESENT — 91.5%"
```

### Why does it work?
The human body is an RF (radio frequency) absorber and reflector. When a person enters a room, WiFi signals bounce differently — causing measurable (but invisible to the naked eye) changes in:
- **Latency** — how long a ping takes to reach the router
- **RSSI** — received signal strength

These patterns differ between an occupied and empty room. ML learns these patterns from labeled data and generalizes to new readings.

---

## 📁 Project Structure

```
wifi_detection/
├── collect.py      # Stage 1 — RSSI + latency collection + webcam auto-labeling
├── features.py     # Stage 2 — Sliding window feature extraction
├── train.py        # Stage 3 — Random Forest + LSTM (PyTorch) training
├── detect.py       # Stage 4 — Real-time inference + terminal dashboard
├── requirements.txt
├── data/
│   ├── raw_samples.csv    # Raw collected signal data
│   └── features.csv       # Extracted feature windows
└── models/
    ├── rf_model.pkl        # Trained Random Forest
    ├── lstm_model.pt       # Trained LSTM weights
    └── scaler.pkl          # Feature normalizer
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/wifi_detection.git
cd wifi_detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install PyTorch (with CUDA for GPU training)
```bash
# For NVIDIA GPU (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch
```

---

## 🚀 Usage — Step by Step

### Step 1 — Find your router IP
```bash
ipconfig
# Look for "Default Gateway" → usually 192.168.1.1
```

### Step 2 — Collect training data
```bash
python collect.py --router 192.168.1.1 --duration 600
```
- Run for **at least 10 minutes**
- Spend ~5 min in the room (person present) and ~5 min outside (room empty)
- Webcam **automatically labels** frames using motion detection
- No camera? Use `--no-cam` flag and label manually

Data saved to `data/raw_samples.csv`

### Step 3 — Extract features
```bash
python features.py
```
Converts raw signal readings into 11 statistical features using sliding windows.
Output → `data/features.csv`

### Step 4 — Train models
```bash
python train.py
```
Trains two models:
- **Random Forest** — fast, interpretable baseline
- **LSTM (PyTorch)** — captures temporal patterns, uses GPU if available

Models saved to `models/`

### Step 5 — Real-time detection
```bash
# Random Forest (recommended for low data)
python detect.py --router 192.168.1.1 --model rf

# LSTM (better with 1000+ training samples)
python detect.py --router 192.168.1.1 --model lstm
```

**Live output:**
```
👤 PERSON PRESENT  [████████████████████░░░░░░░░░░]  91.5%  RSSI=82.0%  Lat=2.0ms  n=0114
🔲 ROOM EMPTY      [████░░░░░░░░░░░░░░░░░░░░░░░░░░]  28.2%  RSSI=82.0%  Lat=1.0ms  n=0195
```

---

## 📊 Features Extracted

| Feature | Description |
|---|---|
| `rssi_mean` | Average signal strength in window |
| `rssi_std` | How much signal fluctuated |
| `rssi_range` | Max − Min signal value |
| `rssi_roc` | Rate of change (speed of fluctuation) |
| `rssi_peak_cnt` | Number of signal spikes |
| `lat_mean` | Average ping latency |
| `lat_std` | Latency variance |
| `lat_max` | Worst latency in window |
| `lat_roc` | Latency change rate |
| `lat_peak_cnt` | Number of latency spikes |
| `rssi_lat_corr` | Correlation between RSSI and latency |

---

## 📈 Results

Tested on ASUS TUF F15 with a standard home router:

| Model | Accuracy | Training Data |
|---|---|---|
| Random Forest | ~69% | 668 samples (~10 min) |
| LSTM | ~53% | Needs more data (1000+ samples) |

> Accuracy improves significantly with more training data. 30+ minutes of varied movement data can push RF accuracy above 85%.

**Top predictive features** (Random Forest importance):

```
lat_std       ████████████████████  0.275
lat_mean      ███████████████████   0.261
lat_roc       ██████████████        0.207
lat_max       ███████████           0.157
lat_peak_cnt  ███████               0.100
```

Latency features dominate because RSSI stays relatively stable on consumer hardware, while latency shows more sensitivity to human presence.

---

## 🔭 Why not CSI?

Real research-grade WiFi sensing (WiTrack, DensePose) uses **Channel State Information (CSI)** — raw physical layer data with amplitude and phase across multiple frequency subcarriers. Far richer than RSSI.

CSI requires special hardware (Intel 5300 NIC, ESP32 with modified firmware). This project intentionally avoids that to stay **zero-cost and fully accessible**.

---

## 🗺️ Future Roadmap

- [ ] Collect 2000+ samples and retrain for 85%+ accuracy
- [ ] Add CSI support via ESP32 (optional hardware upgrade path)
- [ ] Activity recognition: walking vs sitting vs stationary
- [ ] Multi-person occupancy counting
- [ ] Web dashboard (Flask + React) for visualization
- [ ] Integration with smart home systems (Home Assistant, MQTT)
- [ ] Paranormal activity detection (EMF anomaly classification)

---

## 🧪 Tech Stack

| Tool | Use |
|---|---|
| Python 3.10+ | Core language |
| OpenCV | Webcam motion-based labeling |
| NumPy / Pandas | Data processing |
| SciPy | Peak detection, signal processing |
| Scikit-learn | Random Forest, preprocessing |
| PyTorch | LSTM model + CUDA training |
| Windows `netsh` | RSSI extraction |
| Windows `ping` | Latency measurement |

---

## ⚠️ Limitations

- Detection is **coarse** (present/absent), not fine-grained activity recognition
- Consumer RSSI is often stable — model relies mainly on latency variance
- Accuracy varies with room size, router placement, and wall materials
- Through-wall detection not possible without CSI hardware
- Currently **Windows only** (uses `netsh` and `ping` commands)

---

## 📚 References & Inspiration

- [MIT WiTrack — Through-wall Human Tracking](http://witrack.csail.mit.edu/)
- [Meta AI — WiFi-Based Human Pose Estimation](https://ai.meta.com/blog/wifi-human-sensing/)
- [DensePose From WiFi — CMU / Meta Research Paper](https://arxiv.org/abs/2301.00250)

---

👨‍💻 Author
Nishant Singh

📧 Email: ns1199816@gmail.com

🔗 GitHub: https://github.com/nishant1199816

🔗 LinkedIn: https://www.linkedin.com/in/nishant-singh-tech/ 
