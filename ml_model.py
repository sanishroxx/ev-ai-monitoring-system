import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import json
import os

# ── Data Storage ─────────────────────────────────────────────
data_file = "readings.csv"

def save_reading(data):
    df_new = pd.DataFrame([data])
    
    if os.path.exists(data_file):
        df_new.to_csv(data_file, mode='a', 
                      header=False, index=False)
    else:
        df_new.to_csv(data_file, index=False)

def load_data():
    if os.path.exists(data_file):
        return pd.read_csv(data_file)
    return None

# ── Train AI Model ────────────────────────────────────────────
def train_model():
    df = load_data()
    
    if df is None or len(df) < 10:
        print("Not enough data to train!")
        return None, None
    
    print(f"Training on {len(df)} readings...")
    
    # Features
    features = ['energy_kwh', 'voltage', 
                'current', 'temperature', 'soc_percent']
    
    # Check columns exist
    for f in features:
        if f not in df.columns:
            print(f"Column missing: {f}")
            return None, None
    
    X = df[features].dropna()
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train IsolationForest
    model = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100
    )
    model.fit(X_scaled)
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model trained and saved!")
    return model, scaler

# ── Predict ───────────────────────────────────────────────────
def predict(data):
    # Load model
    if not os.path.exists('model.pkl'):
        print("Model not trained yet!")
        return "Unknown"
    
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare input
    features = [[
        data['energy_kwh'],
        data['voltage'],
        data['current'],
        data['temperature'],
        data['soc_percent']
    ]]
    
    # Scale
    scaled = scaler.transform(features)
    
    # Predict
    result = model.predict(scaled)[0]
    score = float(model.score_samples(scaled)[0])
    
    return {
        "status": "ANOMALY" if result == -1 else "Normal",
        "score": round(score, 4)
    }

# ── EDA ───────────────────────────────────────────────────────
def get_stats():
    df = load_data()
    
    if df is None or len(df) == 0:
        return None
    
    stats = {
        "total_readings": len(df),
        "avg_voltage": round(df['voltage'].mean(), 2),
        "avg_temperature": round(df['temperature'].mean(), 2),
        "avg_energy": round(df['energy_kwh'].mean(), 2),
        "avg_current": round(df['current'].mean(), 2),
        "max_temperature": round(df['temperature'].max(), 2),
        "min_voltage": round(df['voltage'].min(), 2),
    }
    
    if 'ai_status' in df.columns:
        anomalies = len(df[df['ai_status'] == 'ANOMALY'])
        stats['total_anomalies'] = anomalies
        stats['normal_readings'] = len(df) - anomalies
    
    return stats

if __name__ == "__main__":
    print("ML Model Ready!")
    train_model()