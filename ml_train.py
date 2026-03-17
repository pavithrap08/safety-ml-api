import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# =============================
# 1. LOAD DATA FROM THINGSPEAK
# =============================
CHANNEL_ID = "2451818"
READ_API_KEY = "AP36OTQMUVKAHQGA"
RESULTS = 800  # increase later

url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={RESULTS}"

resp = requests.get(url, timeout=15)
resp.raise_for_status()
feeds = resp.json().get("feeds", [])

df = pd.DataFrame(feeds)

# Keep required fields (6 fields)
df = df[["created_at", "field1", "field2", "field3", "field4", "field5", "field6"]].copy()
df.columns = ["time", "temperature", "humidity", "gas", "dust", "sound", "flame"]

# Convert to numeric
for c in ["temperature", "humidity", "gas", "dust", "sound", "flame"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna()

# Optional: force sound/flame to 0/1
df["sound"] = (df["sound"] >= 1).astype(int)
df["flame"] = (df["flame"] >= 1).astype(int)

print("Data loaded:", df.shape)
print(df.tail(3))

# =============================
# 2. CREATE RISK LABELS (RULE-BASED)
# =============================
# label meanings: Low / Medium / High  (safe / warning / danger)

def risk_label(row):
    # Flame = immediate HIGH
    if row["flame"] == 1:
        return "High"

    score = 0

    # Temperature
    if row["temperature"] >= 35:
        score += 1

    # Humidity
    if row["humidity"] >= 75:
        score += 1

    # Gas (MQ135 ADC) — use your ESP32 thresholds
    if row["gas"] >= 2000:
        score += 2
    elif row["gas"] >= 1200:
        score += 1

    # Dust (EMA raw-ish) — tune based on your values
    if row["dust"] >= 700:
        score += 2
    elif row["dust"] >= 250:
        score += 1

    # Sound — if LM393 detects loud noise
    if row["sound"] == 1:
        score += 1

    # Map score -> label
    if score <= 1:
        return "Low"
    elif score <= 3:
        return "Medium"
    else:
        return "High"

df["risk"] = df.apply(risk_label, axis=1)

# Encode labels
le = LabelEncoder()
df["risk_encoded"] = le.fit_transform(df["risk"])
print("Label classes:", list(le.classes_))
print(df["risk"].value_counts())

# =============================
# 3. TRAIN RANDOM FOREST
# =============================
X = df[["temperature", "humidity", "gas", "dust", "sound", "flame"]]
y = df["risk_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nRandom Forest Accuracy:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# =============================
# 4. TRAIN ISOLATION FOREST (ANOMALY)
# =============================
# Unsupervised: learns "normal" patterns, flags unusual patterns as anomalies
iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)
iso_model.fit(X)

# =============================
# 5. SAVE MODELS
# =============================
joblib.dump(rf_model, "risk_model.pkl")
joblib.dump(iso_model, "anomaly_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n✅ Models saved: risk_model.pkl, anomaly_model.pkl, label_encoder.pkl")
