import os
import math
import joblib
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --------------------
# Config (ENV VARS)
# --------------------
WORKER_CHANNEL_ID = os.getenv("WORKER_CHANNEL_ID", "2436533")
WORKER_READ_KEY   = os.getenv("WORKER_READ_KEY", "")  # keep secret in Render env vars
RESULT_VERSION    = "v1"

# --------------------
# Load models once (cold start)
# --------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

worker_if = safe_load(os.path.join(MODEL_DIR, "worker_anomaly_model.pkl"))
worker_cols = safe_load(os.path.join(MODEL_DIR, "worker_feature_columns.pkl"))  # list of feature names or None

app = FastAPI(title="Safety ML API", version=RESULT_VERSION)

# Allow Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo; later lock to your Vercel domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "worker_model_loaded": worker_if is not None,
        "cols_loaded": worker_cols is not None,
        "version": RESULT_VERSION
    }

def fetch_latest_worker():
    if not WORKER_READ_KEY:
        raise RuntimeError("WORKER_READ_KEY not set")
    url = f"https://api.thingspeak.com/channels/{WORKER_CHANNEL_ID}/feeds.json"
    r = requests.get(url, params={"api_key": WORKER_READ_KEY, "results": 1}, timeout=10)
    r.raise_for_status()
    feeds = r.json().get("feeds", [])
    if not feeds:
        return None
    f = feeds[0]
    # ThingSpeak fields mapping:
    # field1 HR, field2 SpO2, field3 Temp, field4 Accel, field5 Fall, field6 Presence
    def num(x):
        try:
            return float(x)
        except Exception:
            return None
    return {
        "created_at": f.get("created_at"),
        "hr": num(f.get("field1")),
        "spo2": num(f.get("field2")),
        "temp": num(f.get("field3")),
        "accel": num(f.get("field4")),
        "fall": int(float(f.get("field5") or 0)),
        "presence": int(float(f.get("field6") or 0)),
    }

def demo_haci(v):
    """Simple HACI scoring. Replace/align with your paper definition later."""
    hr = v["hr"] if v["hr"] and v["hr"] > 0 else None
    spo2 = v["spo2"] if v["spo2"] and v["spo2"] >= 70 else None
    temp = v["temp"] if v["temp"] and v["temp"] >= 34 else None
    accel = v["accel"] if v["accel"] and v["accel"] > 0 else None
    fall = v["fall"]

    score = 0
    # SpO2 (35)
    if spo2 is None: score += 24
    elif spo2 >= 95: score += 35
    elif spo2 >= 92: score += 28
    elif spo2 >= 88: score += 20
    else: score += 12

    # HR (25)
    if hr is None: score += 18
    elif 60 <= hr <= 100: score += 25
    elif hr <= 120: score += 18
    else: score += 12

    # Temp (20)
    if temp is None: score += 14
    elif temp <= 37.5: score += 20
    elif temp <= 38.0: score += 14
    else: score += 10

    # Accel (20)
    if fall == 1: score += 0
    elif accel is None: score += 12
    elif 0.9 <= accel <= 1.1: score += 20
    elif accel <= 1.35: score += 16
    elif accel <= 1.8: score += 12
    else: score += 8

    if fall == 1:
        score -= 25

    return int(max(0, min(100, round(score))))

def decide_action(v, haci, anomaly_flag):
    if v["presence"] == 0:
        return {"label": "N/A", "detail": "Worker not detected near machine."}

    if v["fall"] == 1:
        return {"label": "EMERGENCY", "detail": "Fall detected. Call supervisor immediately."}

    if v["spo2"] is not None and v["spo2"] < 92:
        return {"label": "ALERT", "detail": "Low SpO₂. Move to safe zone immediately."}

    if v["temp"] is not None and v["temp"] > 38.0:
        return {"label": "ALERT", "detail": "Heat stress risk. Rest + hydrate."}

    if haci < 60 or (anomaly_flag and haci < 75):
        return {"label": "MICRO-BREAK", "detail": "Take 3–5 minutes rest + hydrate."}

    return {"label": "NORMAL", "detail": "Stable. Continue monitoring."}

def anomaly_predict(v):
    """
    If IsolationForest model exists: run it.
    Else: fallback to simple rule.
    """
    if v["presence"] == 0:
        return False, "presence=0"

    # Build features
    feat = {
        "hr": v["hr"] or 0,
        "spo2": v["spo2"] or 0,
        "temp": v["temp"] or 0,
        "accel": v["accel"] or 0,
        "fall": v["fall"],
        "presence": v["presence"],
    }

    if worker_if is not None and worker_cols is not None:
        row = [feat.get(c, 0) for c in worker_cols]
        X = np.array([row], dtype=float)
        pred = worker_if.predict(X)[0]   # -1 anomaly, 1 normal
        return (pred == -1), "isoforest"
    else:
        # fallback
        rule = (v["fall"] == 1) or ((v["accel"] or 0) > 2.2) or ((v["hr"] or 0) > 130) or ((v["spo2"] or 100) < 90)
        return bool(rule), "rule"

@app.get("/api/worker/latest")
def worker_latest():
    v = fetch_latest_worker()
    if v is None:
        return {"ok": False, "error": "No data"}

    haci = demo_haci(v)
    anomaly_flag, anomaly_source = anomaly_predict(v)
    action = decide_action(v, haci, anomaly_flag)

    return {
        "ok": True,
        "data": v,
        "haci": haci,
        "anomaly": {"flag": anomaly_flag, "source": anomaly_source},
        "action": action
    }
