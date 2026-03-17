import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="ConSafe • Central Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- ThingSpeak channels ----
ENV_CHANNEL_ID = "2451818"
ENV_READ_KEY = "AP36OTQMUVKAHQGA"

# You can add more workers here later (startup-ready)
WORKERS = [
    {
        "name": "Ram",
        "worker_id": "ind_110",
        "industry": "Construction Site",
        "shift": "Morning Shift",
        "channel_id": "2436533",
        "read_key": "IEHPXGUC6U1K4NJX",
        # Mapping: field1 HR, field2 SpO2, field3 BodyTemp, field4 Accel, field5 Fall, field6 Presence
    }
]

# ---- refresh ----
DEFAULT_REFRESH_SEC = 8  # polling interval (ThingSpeak)
THINGSPEAK_RESULTS = 50  # history points to pull for charts (per request)

# ---- Thresholds (tune for demo) ----
TH = {
    "spo2_low": 92,
    "temp_high": 38.0,
    "hr_high": 120,
    "accel_high": 2.2,
    "env_noise_spike": 2000,
    "env_air_poor": 800,
    "env_dust_high": 900,
}

# ---- Optional ML model paths ----
MODEL_PATHS = {
    "env_anomaly": "anomaly_model.pkl",              # uploaded by you ✅
    "worker_anomaly": "worker_anomaly_model.pkl",    # uploaded by you ✅
    "label_encoder": "label_encoder.pkl",            # uploaded by you ✅
    "env_risk": "risk_model.pkl",                    # optional
    "worker_risk": "worker_risk_model.pkl",          # optional
}

# =========================
# UI THEME (Startup/Corporate)
# =========================

CSS = """
<style>
/* App background */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 10% 10%, rgba(59,130,246,0.10), transparent 55%),
              radial-gradient(900px 500px at 90% 20%, rgba(16,185,129,0.10), transparent 55%),
              linear-gradient(180deg, rgba(15,23,42,0.03), rgba(15,23,42,0.00));
}

/* Reduce top padding */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(2,6,23,0.92), rgba(2,6,23,0.80));
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92) !important; }

/* Cards */
.card{
  border: 1px solid rgba(2,6,23,0.10);
  border-radius: 18px;
  padding: 16px 16px;
  background: rgba(255,255,255,0.72);
  box-shadow: 0 18px 60px rgba(2,6,23,0.08);
  backdrop-filter: blur(10px);
}
.card h4{ margin: 0 0 10px 0; font-size: 13px; letter-spacing: .5px; text-transform: uppercase; color: rgba(2,6,23,0.60); }
.kpi{
  display:flex; align-items:flex-start; justify-content:space-between; gap:12px;
}
.kpi .big{ font-size: 32px; font-weight: 900; color: rgba(2,6,23,0.92); line-height:1; }
.kpi .sub{ font-size: 12px; color: rgba(2,6,23,0.60); margin-top: 4px; }
.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 12px;
  border: 1px solid rgba(2,6,23,0.10);
  background: rgba(2,6,23,0.03);
}
.pill.good{ color:#16a34a; border-color: rgba(22,163,74,0.25); background: rgba(22,163,74,0.08); }
.pill.warn{ color:#d97706; border-color: rgba(217,119,6,0.25); background: rgba(217,119,6,0.10); }
.pill.bad{  color:#dc2626; border-color: rgba(220,38,38,0.25); background: rgba(220,38,38,0.10); }

.hr{ height:1px; background: rgba(2,6,23,0.10); margin: 10px 0 0 0; }

.smallnote{
  font-size: 12px; color: rgba(2,6,23,0.62); line-height: 1.5;
}

.headline{
  font-weight: 950;
  letter-spacing: -0.6px;
  color: rgba(2,6,23,0.92);
  font-size: 22px;
}
.subheadline{
  color: rgba(2,6,23,0.60);
  font-size: 13px;
  margin-top: 2px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================
# DATA + ML HELPERS
# =========================

def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

def thingspeak_fetch(channel_id: str, read_key: str, results: int = 1) -> pd.DataFrame:
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    params = {"api_key": read_key, "results": results}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    feeds = js.get("feeds", [])
    if not feeds:
        return pd.DataFrame()
    df = pd.DataFrame(feeds)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for k, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                models[k] = joblib.load(path)
            except Exception:
                models[k] = None
        else:
            models[k] = None
    return models

MODELS = load_models()

def compute_worker_anomaly(hr, spo2, temp, accel, fall) -> Tuple[str, Optional[float]]:
    """
    Returns ("YES"/"NO", score_or_none)
    If worker_anomaly_model exists and supports decision_function/predict, use it.
    Otherwise fallback to rule-based.
    """
    model = MODELS.get("worker_anomaly")
    x = np.array([[hr, spo2, temp, accel, fall]], dtype=float)

    # Model path
    if model is not None:
        try:
            # IsolationForest: predict -> -1 anomaly, 1 normal
            pred = model.predict(x)[0]
            score = None
            if hasattr(model, "decision_function"):
                score = float(model.decision_function(x)[0])
            return ("YES" if pred == -1 else "NO", score)
        except Exception:
            pass

    # Fallback rules
    anomaly = (
        (fall == 1) or
        (accel is not None and accel > TH["accel_high"]) or
        (hr is not None and hr > 130) or
        (spo2 is not None and spo2 < 90)
    )
    return ("YES" if anomaly else "NO", None)

def haci_score(hr, spo2, temp, accel, fall, anomaly_flag) -> int:
    # same style as your worker app: demo-friendly, stable
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

    # Activity (20)
    if fall == 1:
        score += 0
    elif accel is not None and 0.9 <= accel <= 1.1:
        score += 20
    elif accel is not None and accel <= 1.35:
        score += 16
    elif accel is not None and accel <= 1.8:
        score += 12
    else:
        score += 8

    if anomaly_flag == "YES": score -= 8
    if fall == 1: score -= 25

    return int(max(0, min(100, round(score))))

def band_from_score(s: int) -> Tuple[str, str]:
    # label, pillClass
    if s >= 80: return "SAFE", "good"
    if s >= 60: return "MODERATE", "warn"
    if s >= 40: return "WARNING", "warn"
    return "CRITICAL", "bad"

def env_notes(env: Dict[str, Optional[float]]) -> List[str]:
    notes = []
    if env.get("flame") == 1:
        notes.append("Flame detected")
    if env.get("sound") is not None and (env["sound"] >= TH["env_noise_spike"] or env["sound"] == 4095):
        notes.append("Noise spike")
    if env.get("air") is not None and env["air"] >= TH["env_air_poor"]:
        notes.append("Air quality poor")
    if env.get("dust") is not None and env["dust"] >= TH["env_dust_high"]:
        notes.append("Dust high")
    return notes

def env_anomaly(env: Dict[str, Optional[float]]) -> Tuple[str, Optional[float]]:
    """
    ("YES"/"NO", score_or_none)
    Uses anomaly_model.pkl if available, otherwise rule-based.
    """
    model = MODELS.get("env_anomaly")
    vals = [
        env.get("temp") if env.get("temp") is not None else -1,
        env.get("hum") if env.get("hum") is not None else -1,
        env.get("sound") if env.get("sound") is not None else -1,
        env.get("air") if env.get("air") is not None else -1,
        env.get("dust") if env.get("dust") is not None else -1,
        env.get("flame") if env.get("flame") is not None else 0,
    ]
    x = np.array([vals], dtype=float)

    if model is not None:
        try:
            pred = model.predict(x)[0]  # -1 anomaly, 1 normal
            score = None
            if hasattr(model, "decision_function"):
                score = float(model.decision_function(x)[0])
            return ("YES" if pred == -1 else "NO", score)
        except Exception:
            pass

    # fallback:
    notes = env_notes(env)
    return ("YES" if len(notes) > 0 else "NO", None)

# =========================
# ALERTS / INCIDENTS (demo store)
# =========================

def ensure_state():
    if "alerts" not in st.session_state:
        st.session_state.alerts = []  # list of dicts
    if "incidents" not in st.session_state:
        st.session_state.incidents = []
    if "last_tick" not in st.session_state:
        st.session_state.last_tick = None

ensure_state()

def push_alert(source: str, severity: str, title: str, detail: str):
    st.session_state.alerts.insert(0, {
        "time": pd.Timestamp.now(tz="Asia/Kolkata"),
        "source": source,
        "severity": severity,
        "title": title,
        "detail": detail,
        "status": "NEW",
        "owner": "",
    })

def resolve_alert(idx: int):
    try:
        st.session_state.alerts[idx]["status"] = "RESOLVED"
    except Exception:
        pass

def acknowledge_alert(idx: int, owner: str):
    try:
        st.session_state.alerts[idx]["status"] = "ACK"
        st.session_state.alerts[idx]["owner"] = owner
    except Exception:
        pass

# =========================
# DATA PULL (single tick)
# =========================

def pull_env_latest() -> Dict[str, Optional[float]]:
    df = thingspeak_fetch(ENV_CHANNEL_ID, ENV_READ_KEY, results=1)
    if df.empty:
        return {"temp": None, "hum": None, "sound": None, "air": None, "dust": None, "flame": None, "ts": None}

    f = df.iloc[0]
    temp = safe_float(f.get("field1"))
    hum = safe_float(f.get("field2"))
    sound = safe_float(f.get("field3"))
    air = safe_float(f.get("field4"))
    dust = safe_float(f.get("field5"))
    flame = safe_float(f.get("field6"))

    # Treat negatives as missing (your data uses -1)
    def clean(v):
        if v is None: return None
        if v < 0: return None
        return v

    env = {
        "temp": clean(temp),
        "hum": clean(hum),
        "sound": clean(sound),
        "air": clean(air),
        "dust": clean(dust),
        "flame": int(flame) if flame is not None else None,
        "ts": f.get("created_at"),
    }
    return env

def pull_worker_latest(worker_cfg: Dict) -> Dict:
    df = thingspeak_fetch(worker_cfg["channel_id"], worker_cfg["read_key"], results=1)
    base = {
        "name": worker_cfg["name"],
        "worker_id": worker_cfg["worker_id"],
        "industry": worker_cfg["industry"],
        "shift": worker_cfg["shift"],
        "presence": 0,
        "hr": None,
        "spo2": None,
        "temp": None,
        "accel": None,
        "fall": 0,
        "ts": None,
    }
    if df.empty:
        return base

    f = df.iloc[0]

    hr = safe_float(f.get("field1"))
    spo2 = safe_float(f.get("field2"))
    temp = safe_float(f.get("field3"))
    accel = safe_float(f.get("field4"))
    fall = safe_float(f.get("field5"))
    presence = safe_float(f.get("field6"))

    # clean
    hr = hr if (hr is not None and hr > 0) else None
    spo2 = spo2 if (spo2 is not None and spo2 >= 70) else None
    temp = temp if (temp is not None and temp >= 34) else None
    accel = accel if (accel is not None and accel > 0) else None
    fall = int(fall) if fall is not None else 0
    presence = int(presence) if presence is not None else 0

    base.update({
        "presence": presence,
        "hr": hr,
        "spo2": spo2,
        "temp": temp,
        "accel": accel,
        "fall": fall,
        "ts": f.get("created_at"),
    })
    return base

def evaluate_and_generate_alerts(env: Dict, workers: List[Dict]):
    # ENV alerts
    env_flags = env_notes(env)
    env_anom, _ = env_anomaly(env)
    if env_anom == "YES" and env_flags:
        push_alert(
            source="ENV • Site",
            severity="WARNING" if len(env_flags) == 1 else "CRITICAL",
            title="Environmental hazard detected",
            detail=", ".join(env_flags)
        )

    # Worker alerts
    for w in workers:
        if w["presence"] == 0:
            continue

        anomaly_flag, _ = compute_worker_anomaly(
            w["hr"] or -1, w["spo2"] or -1, w["temp"] or -1, w["accel"] or -1, w["fall"]
        )
        haci = haci_score(w["hr"], w["spo2"], w["temp"], w["accel"], w["fall"], anomaly_flag)
        label, _ = band_from_score(haci)

        if w["fall"] == 1:
            push_alert(
                source=f"WORKER • {w['worker_id']}",
                severity="EMERGENCY",
                title="Fall detected",
                detail=f"{w['name']} reported a fall event."
            )
            continue

        if w["spo2"] is not None and w["spo2"] < TH["spo2_low"]:
            push_alert(
                source=f"WORKER • {w['worker_id']}",
                severity="CRITICAL",
                title="Low SpO₂",
                detail=f"{w['name']} SpO₂={w['spo2']}%. Move to safe zone + rest."
            )

        if w["temp"] is not None and w["temp"] > TH["temp_high"]:
            push_alert(
                source=f"WORKER • {w['worker_id']}",
                severity="CRITICAL",
                title="Heat stress risk",
                detail=f"{w['name']} Temp={w['temp']}°C. Rest + hydrate."
            )

        if label in ["WARNING", "CRITICAL"] or (anomaly_flag == "YES" and haci < 75):
            push_alert(
                source=f"WORKER • {w['worker_id']}",
                severity="WARNING" if label == "WARNING" else "CRITICAL",
                title="Fatigue/strain risk",
                detail=f"HACI={haci}, anomaly={anomaly_flag}. Recommend micro-break."
            )

# =========================
# SIDEBAR NAV + CONTROLS
# =========================

st.sidebar.markdown("## 🛡️ SafetyOps")
st.sidebar.markdown("**Centralized Dashboard**")
st.sidebar.markdown("Startup-grade • Real-time monitoring + ML signals")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Workers", "Environment", "Alerts", "Analytics", "Devices", "Settings"],
    index=0
)

refresh_sec = st.sidebar.slider("Auto refresh (seconds)", 5, 30, DEFAULT_REFRESH_SEC, 1)
do_autorefresh = st.sidebar.toggle("Auto refresh", value=True)
owner_name = st.sidebar.text_input("Shift Supervisor Name", value="Supervisor A")

st.sidebar.markdown("---")
st.sidebar.caption("Polling model: ThingSpeak → Dashboard (near real-time)")

# =========================
# LIVE DATA TICK
# =========================

def run_tick(force: bool = False):
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    last = st.session_state.last_tick

    if (not force) and do_autorefresh and last is not None:
        # Prevent over-refresh if user navigates quickly
        if (now - last).total_seconds() < max(2, refresh_sec - 2):
            return None

    env = pull_env_latest()
    workers = [pull_worker_latest(w) for w in WORKERS]

    # score each worker
    for w in workers:
        if w["presence"] == 0:
            w["anomaly"] = "—"
            w["haci"] = None
            w["band"] = "OFFLINE"
            continue

        anom, anom_score = compute_worker_anomaly(
            w["hr"] or -1, w["spo2"] or -1, w["temp"] or -1, w["accel"] or -1, w["fall"]
        )
        w["anomaly"] = anom
        w["anomaly_score"] = anom_score
        w["haci"] = haci_score(w["hr"], w["spo2"], w["temp"], w["accel"], w["fall"], anom)
        w["band"], w["band_class"] = band_from_score(w["haci"])

    env_anom, env_score = env_anomaly(env)
    env["anomaly"] = env_anom
    env["anomaly_score"] = env_score
    env["notes"] = env_notes(env)

    # Create alerts (demo behavior)
    evaluate_and_generate_alerts(env, workers)

    st.session_state.last_tick = now
    return env, workers

# Manual refresh button
if st.sidebar.button("🔄 Refresh now"):
    data = run_tick(force=True)
else:
    data = run_tick(force=False)

# If no tick happened due to throttle, do a safe fetch on first load
if data is None and st.session_state.last_tick is None:
    data = run_tick(force=True)

# Fallback safe defaults
if data is None:
    env_live = {"temp": None, "hum": None, "sound": None, "air": None, "dust": None, "flame": None, "notes": [], "anomaly": "—"}
    workers_live = []
else:
    env_live, workers_live = data

# Auto refresh mechanism (Streamlit-native)
if do_autorefresh:
    time.sleep(0.01)
    st.caption(f"Last update: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%d %b %Y, %I:%M:%S %p')}  •  Auto refresh: {refresh_sec}s")
    st.autorefresh_interval = refresh_sec * 1000  # used by streamlit internals
    # NOTE: Streamlit doesn't have st.meta_refresh in older versions; use st.experimental_rerun pattern:
    st.markdown(f"<meta http-equiv='refresh' content='{refresh_sec}'>", unsafe_allow_html=True)

# =========================
# PAGE RENDERERS
# =========================

def card(title: str, body_html: str):
    st.markdown(f"""
    <div class="card">
      <h4>{title}</h4>
      {body_html}
    </div>
    """, unsafe_allow_html=True)

def pill(label: str, cls: str):
    return f"<span class='pill {cls}'>{label}</span>"

# -------------------------
# Overview
# -------------------------
if page == "Overview":
    st.markdown("<div class='headline'>Central Command</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheadline'>Live safety status across workers and environment • actionable insights • ML anomaly flags</div>", unsafe_allow_html=True)

    # KPIs
    online = sum(1 for w in workers_live if w.get("presence") == 1)
    offline = len(workers_live) - online
    critical_workers = sum(1 for w in workers_live if w.get("band") in ["CRITICAL"])
    warning_workers = sum(1 for w in workers_live if w.get("band") in ["WARNING"])

    env_sev = "SAFE"
    env_cls = "good"
    if env_live.get("notes"):
        env_sev = "WARNING" if len(env_live["notes"]) == 1 else "CRITICAL"
        env_cls = "warn" if env_sev == "WARNING" else "bad"
    if env_live.get("flame") == 1:
        env_sev, env_cls = "CRITICAL", "bad"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Workers Online", f"<div class='kpi'><div><div class='big'>{online}</div><div class='sub'>Online / detected</div></div>{pill('LIVE','good')}</div>")
    with c2:
        card("Workers Offline", f"<div class='kpi'><div><div class='big'>{offline}</div><div class='sub'>Not detected</div></div>{pill('CHECK','warn')}</div>")
    with c3:
        card("Worker Risk", f"<div class='kpi'><div><div class='big'>{critical_workers}</div><div class='sub'>Critical workers</div></div>{pill('CRITICAL','bad')}</div><div class='hr'></div><div class='smallnote'>Warnings: {warning_workers}</div>")
    with c4:
        card("Environment Status", f"<div class='kpi'><div><div class='big'>{env_sev}</div><div class='sub'>Zone conditions</div></div>{pill(env_live.get('anomaly','—'),'warn' if env_live.get('anomaly')=='YES' else 'good')}</div><div class='hr'></div><div class='smallnote'>{' • '.join(env_live.get('notes',[])) or 'Stable conditions'}</div>")

    st.markdown("### 🔥 Live Priority Queue")
    # show top 8 latest alerts
    alerts_df = pd.DataFrame(st.session_state.alerts[:8])
    if alerts_df.empty:
        st.info("No active alerts yet. As data streams in, alerts will appear here.")
    else:
        alerts_df["time"] = alerts_df["time"].dt.strftime("%I:%M:%S %p")
        st.dataframe(alerts_df[["time", "severity", "source", "title", "detail", "status", "owner"]], use_container_width=True, hide_index=True)

    st.markdown("### ⚡ Quick Actions")
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("Clear demo alerts"):
            st.session_state.alerts = []
            st.success("Cleared alerts.")
    with qa2:
        if st.button("Create demo incident from latest alert"):
            if st.session_state.alerts:
                a = st.session_state.alerts[0]
                st.session_state.incidents.insert(0, {
                    "time": pd.Timestamp.now(tz="Asia/Kolkata"),
                    "severity": a["severity"],
                    "source": a["source"],
                    "title": a["title"],
                    "detail": a["detail"],
                    "status": "OPEN",
                    "owner": owner_name,
                })
                st.success("Incident created.")
            else:
                st.warning("No alerts available.")
    with qa3:
        st.caption("Tip: In real deployments, replace polling with MQTT/WebSocket for true streaming updates.")

# -------------------------
# Workers
# -------------------------
elif page == "Workers":
    st.markdown("<div class='headline'>Worker Fleet</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheadline'>Monitor vitals, detect anomalies, rank risk, and track micro-break recommendations</div>", unsafe_allow_html=True)

    if not workers_live:
        st.warning("No worker configuration found.")
    else:
        df = pd.DataFrame(workers_live)

        # Filters
        colf1, colf2, colf3 = st.columns([1,1,2])
        with colf1:
            only_online = st.toggle("Only online workers", value=False)
        with colf2:
            risk_filter = st.selectbox("Risk filter", ["All", "SAFE", "MODERATE", "WARNING", "CRITICAL"], index=0)
        with colf3:
            search = st.text_input("Search (name / ID)", value="")

        view = df.copy()
        if only_online:
            view = view[view["presence"] == 1]
        if risk_filter != "All":
            view = view[view["band"] == risk_filter]
        if search.strip():
            s = search.strip().lower()
            view = view[view["name"].str.lower().str.contains(s) | view["worker_id"].str.lower().str.contains(s)]

        # Risk ranking
        view["haci_sort"] = view["haci"].fillna(999)
        view = view.sort_values(["haci_sort", "presence"], ascending=[True, False])

        # Table
        show_cols = ["name","worker_id","industry","shift","presence","hr","spo2","temp","accel","fall","anomaly","haci","band","ts"]
        view_display = view[show_cols].copy()
        if "ts" in view_display.columns:
            view_display["ts"] = pd.to_datetime(view_display["ts"], errors="coerce").dt.strftime("%d-%b %I:%M:%S %p")
        st.dataframe(view_display, use_container_width=True, hide_index=True)

        st.markdown("### 📈 Worker Drilldown")
        chosen = st.selectbox("Select worker", df["worker_id"].tolist(), index=0)
        wcfg = next((x for x in WORKERS if x["worker_id"] == chosen), None)
        if wcfg:
            hist = thingspeak_fetch(wcfg["channel_id"], wcfg["read_key"], results=THINGSPEAK_RESULTS)
            if hist.empty:
                st.info("No history available yet.")
            else:
                hist["hr"] = pd.to_numeric(hist.get("field1"), errors="coerce")
                hist["spo2"] = pd.to_numeric(hist.get("field2"), errors="coerce")
                hist["temp"] = pd.to_numeric(hist.get("field3"), errors="coerce")
                hist["accel"] = pd.to_numeric(hist.get("field4"), errors="coerce")
                hist["fall"] = pd.to_numeric(hist.get("field5"), errors="coerce")
                hist = hist.dropna(subset=["created_at"]).sort_values("created_at")

                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.line(hist, x="created_at", y=["hr","spo2"], title="Heart Rate & SpO₂ (trend)")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = px.line(hist, x="created_at", y=["temp","accel"], title="Body Temp & Activity (trend)")
                    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Environment
# -------------------------
elif page == "Environment":
    st.markdown("<div class='headline'>Environment Control</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheadline'>Zone-level sensing • thresholds • anomaly flagging • trend visibility</div>", unsafe_allow_html=True)

    # Live cards
    e1, e2, e3 = st.columns(3)
    with e1:
        card("Site Temperature", f"<div class='kpi'><div><div class='big'>{env_live.get('temp','--')}</div><div class='sub'>°C</div></div>{pill('LIVE','good')}</div>")
    with e2:
        card("Humidity", f"<div class='kpi'><div><div class='big'>{env_live.get('hum','--')}</div><div class='sub'>%</div></div>{pill('LIVE','good')}</div>")
    with e3:
        cls = "warn" if env_live.get("anomaly") == "YES" else "good"
        card("Env Anomaly", f"<div class='kpi'><div><div class='big'>{env_live.get('anomaly','—')}</div><div class='sub'>IsolationForest</div></div>{pill('MODEL',cls)}</div><div class='hr'></div><div class='smallnote'>{' • '.join(env_live.get('notes',[])) or 'No hazard pattern detected'}</div>")

    # Trend charts
    hist = thingspeak_fetch(ENV_CHANNEL_ID, ENV_READ_KEY, results=THINGSPEAK_RESULTS)
    if hist.empty:
        st.info("No environment history available yet.")
    else:
        hist = hist.dropna(subset=["created_at"]).sort_values("created_at")
        for f, col in [("field1","temp"),("field2","hum"),("field3","sound"),("field4","air"),("field5","dust"),("field6","flame")]:
            hist[col] = pd.to_numeric(hist.get(f), errors="coerce")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(hist, x="created_at", y=["temp","hum"], title="Temperature & Humidity"), use_container_width=True)
            st.plotly_chart(px.line(hist, x="created_at", y=["air","dust"], title="Air Quality & Dust"), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(hist, x="created_at", y=["sound"], title="Noise / Sound level"), use_container_width=True)
            st.plotly_chart(px.area(hist, x="created_at", y=["flame"], title="Flame status (0/1)"), use_container_width=True)

# -------------------------
# Alerts
# -------------------------
elif page == "Alerts":
    st.markdown("<div class='headline'>Alerts & Incidents</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheadline'>Triaging, acknowledgements, assignments, and audit-ready history</div>", unsafe_allow_html=True)

    if not st.session_state.alerts:
        st.info("No alerts yet. Once thresholds/anomalies occur, they’ll show here.")
    else:
        df = pd.DataFrame(st.session_state.alerts)
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%d-%b %I:%M:%S %p")

        # quick filter
        sev = st.selectbox("Severity", ["All","EMERGENCY","CRITICAL","WARNING","INFO"], index=0)
        status = st.selectbox("Status", ["All","NEW","ACK","RESOLVED"], index=0)
        view = df.copy()
        if sev != "All":
            view = view[view["severity"] == sev]
        if status != "All":
            view = view[view["status"] == status]

        st.dataframe(view[["time","severity","source","title","detail","status","owner"]], use_container_width=True, hide_index=True)

        st.markdown("### Manage latest alert")
        idx = st.number_input("Alert index (0 = latest)", min_value=0, max_value=max(0, len(st.session_state.alerts)-1), value=0, step=1)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Acknowledge"):
                acknowledge_alert(int(idx), owner_name)
                st.success("Acknowledged.")
        with col2:
            if st.button("🧾 Create Incident"):
                a = st.session_state.alerts[int(idx)]
                st.session_state.incidents.insert(0, {
                    "time": pd.Timestamp.now(tz="Asia/Kolkata"),
                    "severity": a["severity"],
                    "source": a["source"],
                    "title": a["title"],
                    "detail": a["detail"],
                    "status": "OPEN",
                    "owner": owner_name,
                })
                st.success("Incident created.")
        with col3:
            if st.button("🟢 Resolve"):
                resolve_alert(int(idx))
                st.success("Resolved.")

    st.markdown("### Incidents (Audit log)")
    inc = pd.DataFrame(st.session_state.incidents)
    if inc.empty:
        st.caption("No incidents created yet.")
    else:
        inc2 = inc.copy()
        inc2["time"] = pd.to_datetime(inc2["time"]).dt.strftime("%d-%b %I:%M:%S %p")
        st.dataframe(inc2, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download incidents CSV", inc2.to_csv(index=False).encode("utf-8"), file_name="incidents.csv")

# -------------------------
# Analytics
# -------------------------
elif page == "Analytics":
    st.markdown("<div class='headline'>Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheadline'>Shift/zone insights • trend summaries • exportable results</div>", unsafe_allow_html=True)

    # Simple analytics from alerts/incidents (demo)
    alerts = pd.DataFrame(st.session_state.alerts)
    if alerts.empty:
        st.info("No alert history yet. Trigger a few alerts and revisit this page.")
    else:
        alerts["time"] = pd.to_datetime(alerts["time"])
        alerts["day"] = alerts["time"].dt.date
        by_sev = alerts.groupby("severity").size().reset_index(name="count").sort_values("count", ascending=False)
        by_source = alerts.groupby("source").size().reset_index(name="count").sort_values("count", ascending=False).head(10)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(by_sev, x="severity", y="count", title="Alerts by severity"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(by_source, x="count", y="source", orientation="h", title="Top alert sources"), use_container_width=True)

        by_day = alerts.groupby("day").size().reset_index(name="count")
        st.plotly_chart(px.line(by_day, x="day", y="count", title="Alerts over time"), use_container_width=True)

# -------------------------
# Devices
# -------------------------
elif page == "Devices":
    st.markdown("<div class='headline'>Device & Data Health</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheadline'>Heartbeat, freshness, missing data, and flatline detection</div>", unsafe_allow_html=True)

    rows = []
    # Environment device
    rows.append({
        "device": "ENV • Site Node",
        "channel": ENV_CHANNEL_ID,
        "last_seen": str(env_live.get("ts")),
        "anomaly": env_live.get("anomaly"),
        "notes": ", ".join(env_live.get("notes", [])) or "—",
    })
    # Worker devices
    for w in workers_live:
        rows.append({
            "device": f"WORKER • {w['worker_id']}",
            "channel": next((x["channel_id"] for x in WORKERS if x["worker_id"] == w["worker_id"]), "—"),
            "last_seen": str(w.get("ts")),
            "anomaly": w.get("anomaly", "—"),
            "notes": f"presence={w.get('presence')} fall={w.get('fall')}",
        })

    ddf = pd.DataFrame(rows)
    st.dataframe(ddf, use_container_width=True, hide_index=True)

    st.markdown("### Data quality tips (what supervisors love)")
    st.markdown("""
- **Flatline detection:** if a sensor value repeats exactly for long time → sensor stuck / wiring issue  
- **Missing values:** treat `-1` or empty as missing, don’t plot them as real values  
- **Freshness:** if last update > 30 seconds (for your demo) → device offline indicator  
""")

# -------------------------
# Settings
# -------------------------
elif page == "Settings":
    st.markdown("<div class='headline'>Settings</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheadline'>Threshold tuning + deployment notes</div>", unsafe_allow_html=True)

    st.markdown("### Thresholds")
    col1, col2, col3 = st.columns(3)
    with col1:
        TH["spo2_low"] = st.number_input("SpO₂ low (%)", 80, 98, TH["spo2_low"])
        TH["temp_high"] = st.number_input("Body temp high (°C)", 36.0, 42.0, float(TH["temp_high"]))
    with col2:
        TH["hr_high"] = st.number_input("HR high (bpm)", 80, 180, TH["hr_high"])
        TH["accel_high"] = st.number_input("Accel high (g)", 1.0, 5.0, float(TH["accel_high"]))
    with col3:
        TH["env_air_poor"] = st.number_input("Air poor threshold", 100, 5000, TH["env_air_poor"])
        TH["env_dust_high"] = st.number_input("Dust high threshold", 100, 5000, TH["env_dust_high"])

    st.markdown("### ML Model status")
    mrows = []
    for k, path in MODEL_PATHS.items():
        mrows.append({"model": k, "file": path, "loaded": bool(MODELS.get(k))})
    st.dataframe(pd.DataFrame(mrows), use_container_width=True, hide_index=True)

    st.markdown("### Deployment note (startup-style)")
    st.markdown("""
For a **true real-time** startup deployment:
- ESP32 → **MQTT broker** (Mosquitto / EMQX)  
- Backend → FastAPI/Node consumes MQTT and publishes **WebSocket** updates  
- Dashboard → subscribes to WebSocket for instant updates (no polling)  
ThingSpeak polling = **near real-time**, good for demo & academic review.
""")