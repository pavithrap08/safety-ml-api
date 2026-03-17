import requests
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# ThingSpeak Worker Channel
# -----------------------------
WORKER_CHANNEL_ID = "2436533"
WORKER_READ_KEY = "IEHPXGUC6U1K4NJX"

# -----------------------------
# Load trained worker ML model
# -----------------------------
worker_if = joblib.load("worker_anomaly_model.pkl")
wcols = joblib.load("worker_feature_columns.pkl")

def safe_float(x):
    try:
        if x is None or x == "":
            return np.nan
        return float(x)
    except:
        return np.nan

def clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))

# -----------------------------
# Clean invalid values (based on your logs)
# -----------------------------
def clean_worker_values(hr, spo2, temp, accel):
    hr = np.nan if (np.isnan(hr) or hr <= 0 or hr > 220) else hr
    spo2 = np.nan if (np.isnan(spo2) or spo2 < 70 or spo2 > 100) else spo2
    temp = np.nan if (np.isnan(temp) or temp < 34 or temp > 42) else temp
    accel = np.nan if (np.isnan(accel) or accel <= 0) else accel
    return hr, spo2, temp, accel

# -----------------------------
# HACI scoring tuned for your accel in g
# -----------------------------
def spo2_score(spo2):
    if np.isnan(spo2): return 70
    if spo2 >= 95: return 100
    if 92 <= spo2 <= 94: return 85
    if 88 <= spo2 <= 91: return 65
    return 40

def hr_score(hr):
    if np.isnan(hr): return 70
    if 60 <= hr <= 100: return 100
    if 50 <= hr <= 59 or 101 <= hr <= 120: return 75
    if 40 <= hr <= 49 or 121 <= hr <= 140: return 55
    return 40

def temp_score(temp):
    if np.isnan(temp): return 70
    if 36.5 <= temp <= 37.5: return 100
    if 37.6 <= temp <= 38.0: return 80
    if 38.1 <= temp <= 39.0: return 55
    return 40

def activity_score_from_accel_g(accel_g, fall):
    # Your normal is ~0.94–1.06g, fall spike example 3.2g with fall=1
    if not np.isnan(fall) and fall >= 1:
        return 10
    if np.isnan(accel_g):
        return 70
    if 0.90 <= accel_g <= 1.10: return 100
    if 1.10 < accel_g <= 1.35 or 0.75 <= accel_g < 0.90: return 85
    if 1.35 < accel_g <= 1.80: return 65
    if 1.80 < accel_g < 2.70: return 45
    return 25

def haci_band(haci):
    if np.isnan(haci): return "N/A (No presence)"
    if haci >= 80: return "SAFE"
    if haci >= 60: return "MODERATE"
    if haci >= 40: return "WARNING"
    return "CRITICAL"

def compute_haci(hr, spo2, temp, accel, fall, presence, anomaly_flag, anomaly_score,
                 w=(0.35,0.25,0.20,0.20), gamma=12.0, delta=8.0):
    """
    HACI uses worker-only ML here:
    - anomaly_flag (-1/1) and anomaly_score from worker Isolation Forest
    gamma/delta penalize abnormality.
    """
    if np.isnan(presence) or presence <= 0:
        return np.nan

    s = (
        w[0]*spo2_score(spo2) +
        w[1]*hr_score(hr) +
        w[2]*temp_score(temp) +
        w[3]*activity_score_from_accel_g(accel, fall)
    )

    # ML penalty
    penalty_flag = gamma if int(anomaly_flag) == -1 else 0
    penalty_score = 0.0
    # decision_function: lower/negative -> more abnormal
    if anomaly_score is not None:
        penalty_score = delta * max(0.0, -float(anomaly_score))

    # fall penalty
    fall_penalty = 20 if (not np.isnan(fall) and fall >= 1) else 0

    return clamp(s - penalty_flag - penalty_score - fall_penalty, 0, 100)

# -----------------------------
# Fetch latest ThingSpeak entry
# -----------------------------
url = f"https://api.thingspeak.com/channels/{WORKER_CHANNEL_ID}/feeds.json"
r = requests.get(url, params={"api_key": WORKER_READ_KEY, "results": 1}, timeout=20)
r.raise_for_status()
feed = r.json()["feeds"][0]

hr = safe_float(feed.get("field1"))
spo2 = safe_float(feed.get("field2"))
temp = safe_float(feed.get("field3"))
accel = safe_float(feed.get("field4"))
fall = safe_float(feed.get("field5"))
presence = safe_float(feed.get("field6"))

# Clean invalid based on your real values
hr, spo2, temp, accel = clean_worker_values(hr, spo2, temp, accel)

# If worker not present, skip ML/HACI
if np.isnan(presence) or presence <= 0:
    print("Presence=0 → Worker not detected near machine. HACI: N/A")
    raise SystemExit

# Worker ML anomaly prediction
X = pd.DataFrame([[hr, spo2, temp, accel]], columns=wcols).astype(float)
anomaly_flag = int(worker_if.predict(X)[0])  # -1 anomaly, 1 normal
anomaly_score = float(worker_if.decision_function(X)[0])

# HACI
haci = compute_haci(hr, spo2, temp, accel, fall, presence, anomaly_flag, anomaly_score)
def recommend_action(hr, spo2, temp, fall, presence, anomaly_flag, haci):
    if np.isnan(presence) or presence <= 0:
        return "N/A (Worker not present)"

    if not np.isnan(fall) and fall >= 1:
        return "EMERGENCY: Fall detected — notify supervisor"

    # Medical safety
    if not np.isnan(spo2) and spo2 < 92:
        return "ALERT: Low SpO2 — move worker to safe zone"

    if not np.isnan(temp) and temp > 38.0:
        return "ALERT: Possible heat stress — hydration + rest"

    # Micro-break logic
    if not np.isnan(haci) and haci < 60:
        return "MICRO-BREAK: Recommend 3–5 min rest + hydration"

    if anomaly_flag == -1 and (np.isnan(haci) or haci < 75):
        return "MICRO-BREAK: Unusual pattern detected — short rest advised"

    return "NORMAL: Continue monitoring"

print("Worker Latest:")
print(" HR:", hr, "SpO2:", spo2, "Temp:", temp, "Accel(g):", accel, "Fall:", fall, "Presence:", presence)
print("Worker ML Anomaly:", "YES" if anomaly_flag == -1 else "NO", "| score:", round(anomaly_score, 3))
print("HACI:", round(haci, 1), "/100 |", haci_band(haci))
action = recommend_action(hr, spo2, temp, fall, presence, anomaly_flag, haci)
print("ACTION:", action)

# Suggested action
if fall >= 1:
    print("ACTION: EMERGENCY (Fall detected)")
elif anomaly_flag == -1 and haci < 60:
    print("ACTION: ALERT (Anomalous + low HACI)")
elif haci < 40:
    print("ACTION: ALERT (Critical HACI)")
elif anomaly_flag == -1:
    print("ACTION: MONITOR (Anomaly detected)")
else:
    print("ACTION: NORMAL")