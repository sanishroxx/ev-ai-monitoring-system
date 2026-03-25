import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
import threading
import time
from ml_model import train_model, predict, save_reading, get_stats, load_data

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Tech OVN — EV Charger AI Monitor",
    page_icon="⚡",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0f1e; }
   .stMetric {
    background: #1a3a6e !important;
    border: 2px solid #00d4ff !important;
    border-radius: 10px !important;
    padding: 10px !important;
    color: #ffffff !important;
}

[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-size: 36px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-weight: 600 !important;
}
    .anomaly-alert {
        background: rgba(255,68,102,0.1);
        border: 2px solid #ff4466;
        border-radius: 10px;
        padding: 15px;
        color: #ff4466;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .normal-alert {
        background: rgba(0,255,136,0.1);
        border: 2px solid #00ff88;
        border-radius: 10px;
        padding: 15px;
        color: #00ff88;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────
st.title("⚡ Tech OVN — EV Charger AI Monitor")
st.markdown("**OCPP Battery Monitoring System — Powered by Tech OVN**")
st.divider()

# ── Session State ─────────────────────────────────────────────
if 'readings' not in st.session_state:
    st.session_state.readings = []

if 'running' not in st.session_state:
    st.session_state.running = False

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
st.sidebar.divider()

# Train Model Button
if st.sidebar.button("🤖 Train AI Model"):
    with st.spinner("Training AI Model..."):
        model, scaler = train_model()
        if model:
            st.sidebar.success("✅ Model Trained!")
        else:
            st.sidebar.warning("Need more data first!")

# Charger Selection
charger = st.sidebar.selectbox(
    "Select Charger",
    ["OVN-EV-001", "OVN-EV-002", "OVN-EV-003"]
)

st.sidebar.divider()
st.sidebar.markdown("**System Info**")
st.sidebar.markdown("🔌 OCPP v1.6")
st.sidebar.markdown("🤖 IsolationForest AI")
st.sidebar.markdown("📊 Real-time Monitor")

# ── Main Dashboard ────────────────────────────────────────────

# Stats Row
stats = get_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    total = stats['total_readings'] if stats else 0
    st.metric("📊 Total Readings", total)

with col2:
    normal = stats.get('normal_readings', 0) if stats else 0
    st.metric("✅ Normal", normal)

with col3:
    anomalies = stats.get('total_anomalies', 0) if stats else 0
    st.metric("⚠️ Anomalies", anomalies)

with col4:
    avg_temp = stats['avg_temperature'] if stats else 0
    st.metric("🌡️ Avg Temperature", f"{avg_temp}°C")

st.divider()

# ── Live Data Section ─────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 Live Battery Data")
    
    df = load_data()
    
    if df is not None and len(df) > 0:
        # Filter by charger
        charger_df = df[df['charger_id'] == charger]
        
        if len(charger_df) > 0:
            # Charts
            tab1, tab2, tab3 = st.tabs([
                "⚡ Voltage", 
                "🌡️ Temperature", 
                "🔋 Energy"
            ])
            
            with tab1:
                st.line_chart(
                    charger_df.tail(50)['voltage'],
                    color="#00d4ff"
                )
            
            with tab2:
                st.line_chart(
                    charger_df.tail(50)['temperature'],
                    color="#ff4466"
                )
            
            with tab3:
                st.line_chart(
                    charger_df.tail(50)['energy_kwh'],
                    color="#00ff88"
                )
        else:
            st.info(f"No data for {charger} yet!")
    else:
        st.info("No data yet — Start simulator first!")

with col_right:
    st.subheader("🔴 Latest Readings")
    
    df = load_data()
    
    if df is not None and len(df) > 0:
        latest = df.tail(5)[['charger_id', 
                              'voltage', 
                              'temperature',
                              'status']].iloc[::-1]
        
        for _, row in latest.iterrows():
            if row.get('status') == 'ANOMALY':
                st.markdown(f"""
                <div class="anomaly-alert">
                ⚠️ {row['charger_id']}<br/>
                {row['voltage']}V | {row['temperature']}°C<br/>
                ANOMALY!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-alert">
                ✅ {row['charger_id']}<br/>
                {row['voltage']}V | {row['temperature']}°C<br/>
                Normal
                </div>
                """, unsafe_allow_html=True)
            st.write("")
    else:
        st.info("Waiting for data...")

st.divider()

# ── AI Predictor ──────────────────────────────────────────────
st.subheader("🤖 AI Anomaly Detector — Test Manually")

col1, col2, col3 = st.columns(3)

with col1:
    voltage = st.number_input("Voltage (V)", 
                               value=400.0, 
                               step=1.0)
    energy = st.number_input("Energy (kWh)", 
                              value=15.0, 
                              step=0.1)

with col2:
    current = st.number_input("Current (A)", 
                               value=32.0, 
                               step=0.1)
    temperature = st.number_input("Temperature (°C)", 
                                   value=35.0, 
                                   step=0.1)

with col3:
    soc = st.number_input("SOC (%)", 
                           value=60.0, 
                           step=1.0)
    st.write("")
    st.write("")
    predict_btn = st.button("⚡ ANALYZE", 
                             use_container_width=True)

if predict_btn:
    result = predict({
        'energy_kwh': energy,
        'voltage': voltage,
        'current': current,
        'temperature': temperature,
        'soc_percent': soc
    })
    
    if isinstance(result, dict):
        if result['status'] == 'ANOMALY':
            st.markdown("""
            <div class="anomaly-alert">
            ⚠️ ANOMALY DETECTED! Check charger immediately!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="normal-alert">
            ✅ All readings normal — Charger operating fine!
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Train model first!")

st.divider()

# ── Data Table ────────────────────────────────────────────────
st.subheader("📋 All Readings")

df = load_data()

if df is not None and len(df) > 0:
    st.dataframe(
        df.tail(20).iloc[::-1],
        use_container_width=True
    )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Data",
        csv,
        "charger_data.csv",
        "text/csv"
    )
else:
    st.info("No data yet!")

# ── Auto Refresh ──────────────────────────────────────────────
st.divider()
auto_refresh = st.checkbox("🔄 Auto Refresh (every 5 seconds)")

if auto_refresh:
    time.sleep(5)
    st.rerun()