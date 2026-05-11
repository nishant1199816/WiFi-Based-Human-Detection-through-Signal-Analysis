"""
Stage 1 — Data Collection
Collects WiFi signal strength (RSSI), ping latency, and webcam-based
presence labels simultaneously. Saves everything to data/raw_samples.csv

Run:  python collect.py --router 192.168.1.1 --duration 300
      (adjust --router to your gateway IP, find it via: ipconfig)
"""

import subprocess, re, time, csv, argparse, os
from datetime import datetime
import cv2
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ  = 2          # samples per second (2 Hz is safe on Windows)
MOTION_THRESH   = 25         # pixel diff threshold for webcam motion label
OUTPUT_FILE     = os.path.join("data", "raw_samples.csv")
CSV_HEADER      = ["timestamp", "rssi_pct", "latency_ms", "label"]
# ────────────────────────────────────────────────────────────────────────────


def get_rssi() -> float:
    """
    Returns WiFi signal quality (0-100%) via netsh on Windows.
    Maps roughly to RSSI: quality% ≈ 2*(RSSI_dBm + 100)
    Returns -1.0 on failure.
    """
    try:
        out = subprocess.check_output(
            ["netsh", "wlan", "show", "interfaces"],
            stderr=subprocess.DEVNULL,
            timeout=2
        ).decode("utf-8", errors="ignore")
        match = re.search(r"Signal\s*:\s*(\d+)\s*%", out)
        return float(match.group(1)) if match else -1.0
    except Exception:
        return -1.0


def get_latency(router_ip: str) -> float:
    """
    Pings the router once and returns RTT in milliseconds.
    Returns -1.0 on timeout or failure.
    """
    try:
        out = subprocess.check_output(
            ["ping", "-n", "1", "-w", "800", router_ip],
            stderr=subprocess.DEVNULL,
            timeout=3
        ).decode("utf-8", errors="ignore")
        match = re.search(r"Average\s*=\s*(\d+)\s*ms", out)
        if not match:
            match = re.search(r"time[=<](\d+)ms", out)
        return float(match.group(1)) if match else -1.0
    except Exception:
        return -1.0


class WebcamLabeler:
    """
    Simple motion-based presence labeler using the laptop webcam.
    Returns label = 1 (person present / moving) or 0 (empty room).
    Uses frame differencing — no ML required here.
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check camera permissions.")
        self.prev_gray = None

    def get_label(self) -> int:
        ret, frame = self.cap.read()
        if not ret:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0
        delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, thresh = cv2.threshold(delta, MOTION_THRESH, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(thresh)
        return 1 if motion_pixels > 500 else 0

    def release(self):
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="WiFi Presence Data Collector")
    parser.add_argument("--router",   default="192.168.1.1", help="Router/gateway IP")
    parser.add_argument("--duration", type=int, default=300,  help="Collection time in seconds")
    parser.add_argument("--no-cam",   action="store_true",    help="Skip webcam labeling (label=manual)")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    interval = 1.0 / SAMPLE_RATE_HZ

    print(f"[INFO] Router IP   : {args.router}")
    print(f"[INFO] Duration    : {args.duration}s  ({args.duration * SAMPLE_RATE_HZ} samples)")
    print(f"[INFO] Output file : {OUTPUT_FILE}")
    print("[INFO] Starting collection — press Ctrl+C to stop early\n")

    # ── manual label prompt if no cam ──
    if args.no_cam:
        print("Webcam disabled. Enter label manually:")
        print("  Press ENTER each time person ENTERS or LEAVES the room.")
        current_label = int(input("Start label (0=empty, 1=person present): ").strip() or "0")
        labeler = None
    else:
        labeler = WebcamLabeler()
        current_label = None  # will come from cam

    file_exists = os.path.exists(OUTPUT_FILE)
    start_time  = time.time()
    sample_count = 0

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)

        try:
            while (time.time() - start_time) < args.duration:
                t0 = time.time()

                ts      = datetime.now().isoformat(timespec="milliseconds")
                rssi    = get_rssi()
                latency = get_latency(args.router)

                if labeler:
                    label = labeler.get_label()
                else:
                    label = current_label  # last manual entry

                writer.writerow([ts, rssi, latency, label])
                sample_count += 1

                status = f"\r[{sample_count:04d}] RSSI={rssi:5.1f}%  Latency={latency:6.1f}ms  Label={label}"
                print(status, end="", flush=True)

                elapsed = time.time() - t0
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")

    if labeler:
        labeler.release()

    print(f"\n[DONE] {sample_count} samples saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()