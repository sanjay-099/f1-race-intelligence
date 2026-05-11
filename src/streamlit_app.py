"""
F1 Race Intelligence System — Streamlit Dashboard
Cloud-optimized version using pre-extracted CSV data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1a2e; padding: 1rem; border-radius: 10px; }
    h1 { color: #FF1E1E; font-weight: bold; }
    h2 { color: #00d2ff; }
    h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ROOT / "data" / "exports"
MODELS_DIR = ROOT / "models"

# ── Data Loaders ─────────────────────────────────────────
@st.cache_data
def load_event_info():
    with open(EXPORTS_DIR / "bahrain_2024_event.json") as f:
        return json.load(f)

@st.cache_data
def load_laps():
    laps = pd.read_csv(EXPORTS_DIR / "bahrain_2024_laps.csv")
    laps['LapTime'] = pd.to_timedelta(laps['LapTime'], errors='coerce')
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    return laps

@st.cache_data
def load_telemetry():
    return pd.read_csv(EXPORTS_DIR / "bahrain_2024_fastest_telemetry.csv")

@st.cache_resource
def load_models():
    tire_model = joblib.load(MODELS_DIR / "tire_degradation_model.pkl")
    compound_encoder = joblib.load(MODELS_DIR / "compound_encoder.pkl")
    driver_encoder = joblib.load(MODELS_DIR / "driver_encoder.pkl")
    iso_forest = joblib.load(MODELS_DIR / "incident_isolation_forest.pkl")
    scaler = joblib.load(MODELS_DIR / "incident_scaler.pkl")
    return tire_model, compound_encoder, driver_encoder, iso_forest, scaler

# ── Header ───────────────────────────────────────────────
st.title("🏎️ F1 Race Intelligence System")
st.markdown("**Real-time race analytics powered by FastF1 telemetry + Machine Learning**")
st.markdown("---")

# ── Load Everything ──────────────────────────────────────
event_info = load_event_info()
laps = load_laps()
telemetry = load_telemetry()
tire_model, compound_encoder, driver_encoder, iso_forest, scaler = load_models()

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=150)
st.sidebar.title("🏁 Race Selection")
st.sidebar.markdown(f"**Race:** {event_info['event_name']}")
st.sidebar.markdown(f"**Year:** {event_info['event_year']}")
st.sidebar.markdown(f"**Circuit:** {event_info['location']}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Navigate",
    ["🏠 Race Overview", "📡 Telemetry Analysis", "🚨 Incident Detection", "🎯 Strategy Optimizer"]
)

# ════════════════════════════════════════════════════════
# 🏠 RACE OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Race Overview":
    st.header("🏠 Race Overview")
    st.success("✅ Race data loaded successfully!")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Laps", len(laps))
    col2.metric("Drivers", laps['Driver'].nunique())
    col3.metric("Race Date", event_info['date'])
    col4.metric("Location", event_info['location'])

# ════════════════════════════════════════════════════════
# 📡 TELEMETRY ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "📡 Telemetry Analysis":
    st.header("📡 Telemetry Analysis")
    st.markdown("Compare driver telemetry from their fastest lap")
    
    drivers_list = sorted(telemetry['Driver'].unique().tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.selectbox("🏎️ Driver 1", drivers_list, 
                               index=drivers_list.index('HAM') if 'HAM' in drivers_list else 0)
    with col2:
        driver2 = st.selectbox("🏎️ Driver 2 (comparison)", ['None'] + drivers_list, index=0)
    
    tel1 = telemetry[telemetry['Driver'] == driver1]
    
    st.subheader(f"📊 {driver1} — Fastest Lap Stats")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lap Time", str(tel1['LapTime'].iloc[0]).split('.')[0][-5:] if len(tel1) > 0 else "N/A")
    m2.metric("Lap Number", int(tel1['LapNumber'].iloc[0]) if len(tel1) > 0 else 0)
    m3.metric("Compound", tel1['Compound'].iloc[0] if len(tel1) > 0 else "N/A")
    m4.metric("Max Speed", f"{tel1['Speed'].max():.0f} km/h" if len(tel1) > 0 else "N/A")
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Speed (km/h)", "Throttle (%)", "Gear"),
        vertical_spacing=0.08
    )
    
    fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Speed'],
                             name=driver1, line=dict(color='#00d2ff', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['Throttle'],
                             line=dict(color='#00ff88', width=2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=tel1['Distance'], y=tel1['nGear'],
                             line=dict(color='#ff6b6b', width=2), showlegend=False), row=3, col=1)
    
    if driver2 != 'None':
        tel2 = telemetry[telemetry['Driver'] == driver2]
        fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Speed'],
                                 name=driver2, line=dict(color='#FF1E1E', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['Throttle'],
                                 line=dict(color='#ffaa00', width=2, dash='dash'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=tel2['Distance'], y=tel2['nGear'],
                                 line=dict(color='#aa00ff', width=2, dash='dash'), showlegend=False), row=3, col=1)
    
    fig.update_layout(template='plotly_dark', height=700, hovermode='x unified')
    fig.update_xaxes(title_text="Distance (m)", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════
# 🚨 INCIDENT DETECTION
# ════════════════════════════════════════════════════════
elif page == "🚨 Incident Detection":
    st.header("🚨 Incident Detection")
    st.markdown("ML-powered anomaly detection on lap times")
    
    clean_laps = laps[
        (laps['IsAccurate'] == True) &
        (laps['LapTimeSeconds'] > 85) &
        (laps['LapTimeSeconds'] < 120)
    ].copy()
    
    features_df = clean_laps[['LapTimeSeconds', 'TyreLife', 'LapNumber']].dropna()
    X_scaled = scaler.transform(features_df)
    predictions = iso_forest.predict(X_scaled)
    
    clean_laps_iso = clean_laps.dropna(subset=['LapTimeSeconds', 'TyreLife', 'LapNumber']).copy()
    clean_laps_iso['IsAnomaly'] = (predictions == -1)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Laps", len(clean_laps_iso))
    col2.metric("🚨 Incidents", int(clean_laps_iso['IsAnomaly'].sum()))
    col3.metric("Detection Rate", f"{clean_laps_iso['IsAnomaly'].sum() / len(clean_laps_iso) * 100:.1f}%")
    col4.metric("Drivers Flagged", clean_laps_iso[clean_laps_iso['IsAnomaly']]['Driver'].nunique())
    
    st.markdown("---")
    
    all_drivers = sorted(clean_laps_iso['Driver'].unique().tolist())
    selected_drivers = st.multiselect("🏎️ Filter drivers (empty = all)", all_drivers, default=[])
    plot_df = clean_laps_iso if not selected_drivers else clean_laps_iso[clean_laps_iso['Driver'].isin(selected_drivers)]
    
    fig = go.Figure()
    driver_order = plot_df.groupby('Driver')['LapTimeSeconds'].mean().sort_values().index.tolist()
    compound_colors = {'SOFT': '#FF3333', 'MEDIUM': '#FFFF00', 'HARD': '#CCCCCC'}
    
    for driver in driver_order:
        d_laps = plot_df[plot_df['Driver'] == driver]
        normal = d_laps[~d_laps['IsAnomaly']]
        fig.add_trace(go.Scatter(
            x=normal['LapNumber'], y=[driver] * len(normal),
            mode='markers',
            marker=dict(size=6, color=[compound_colors.get(c, '#888') for c in normal['Compound']], opacity=0.5),
            showlegend=False
        ))
        anom = d_laps[d_laps['IsAnomaly']]
        if len(anom) > 0:
            fig.add_trace(go.Scatter(
                x=anom['LapNumber'], y=[driver] * len(anom),
                mode='markers',
                marker=dict(size=14, color='red', symbol='circle-open', line=dict(width=3, color='red')),
                showlegend=False
            ))
    
    fig.update_layout(
        title='🚨 Race-Wide Incident Map<br><sub>Red circles = Detected Incidents | Colors = Compound</sub>',
        xaxis_title='Lap Number', yaxis_title='Driver',
        template='plotly_dark', height=700,
        yaxis=dict(categoryorder='array', categoryarray=driver_order[::-1])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Incident Report")
    incidents = clean_laps_iso[clean_laps_iso['IsAnomaly']][
        ['Driver', 'LapNumber', 'LapTimeSeconds', 'Compound', 'TyreLife', 'Stint']
    ].sort_values('LapNumber').reset_index(drop=True)
    st.dataframe(incidents, use_container_width=True, height=400)

# ════════════════════════════════════════════════════════
# 🎯 STRATEGY OPTIMIZER
# ════════════════════════════════════════════════════════
elif page == "🎯 Strategy Optimizer":
    st.header("🎯 Strategy Optimizer")
    st.markdown("ML-powered pit stop optimization")
    
    def predict_stint(driver, compound, stint_length, start_lap=1):
        driver_enc = driver_encoder.transform([driver])[0]
        compound_enc = compound_encoder.transform([compound])[0]
        stint_data = []
        for i in range(1, stint_length + 1):
            race_lap = start_lap + i - 1
            stint_data.append({
                'TyreLife': i, 'TyreLifeSquared': i ** 2,
                'CompoundEncoded': compound_enc, 'DriverEncoded': driver_enc,
                'CompoundAge': compound_enc * i, 'LapNumber': race_lap
            })
        return tire_model.predict(pd.DataFrame(stint_data))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        driver = st.selectbox("🏎️ Driver", sorted(driver_encoder.classes_),
                              index=list(sorted(driver_encoder.classes_)).index('HAM'))
    with col2:
        stint1_compound = st.selectbox("🛞 Stint 1", ['SOFT', 'HARD'], index=0)
    with col3:
        stint2_compound = st.selectbox("🛞 Stint 2", ['HARD', 'SOFT'], index=0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        race_length = st.number_input("🏁 Race Length", min_value=30, max_value=80, value=57)
    with col5:
        pit_loss = st.number_input("⏱️ Pit Loss (s)", min_value=15.0, max_value=30.0, value=22.0, step=0.5)
    with col6:
        pit_min, pit_max = st.slider("🎯 Search Range", 5, race_length - 5, (10, 45))
    
    if stint1_compound == stint2_compound:
        st.warning("⚠️ Compounds must differ!")
    
    if st.button("🚀 RUN OPTIMIZER", type="primary"):
        with st.spinner(f"Simulating strategies..."):
            strategies = []
            for pit_lap in range(pit_min, pit_max + 1):
                s1 = predict_stint(driver, stint1_compound, pit_lap, 1)
                s2 = predict_stint(driver, stint2_compound, race_length - pit_lap, pit_lap + 1)
                total = sum(s1) + pit_loss + sum(s2)
                strategies.append({
                    'PitLap': pit_lap, 'Stint1Time': round(sum(s1), 3),
                    'Stint2Time': round(sum(s2), 3), 'TotalRaceTime': round(total, 3)
                })
            
            strategies_df = pd.DataFrame(strategies)
            optimal = strategies_df.loc[strategies_df['TotalRaceTime'].idxmin()]
            top5 = strategies_df.nsmallest(5, 'TotalRaceTime')
            
            st.success(f"✅ Optimization complete!")
            st.subheader("🏆 OPTIMAL STRATEGY")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Pit Lap", int(optimal['PitLap']))
            r2.metric("Stint 1", f"{int(optimal['PitLap'])} laps {stint1_compound}")
            r3.metric("Stint 2", f"{race_length - int(optimal['PitLap'])} laps {stint2_compound}")
            r4.metric("Total Time", f"{optimal['TotalRaceTime']:.1f}s")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strategies_df['PitLap'], y=strategies_df['TotalRaceTime'],
                                     mode='lines+markers', line=dict(color='#00d2ff', width=3),
                                     marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=[optimal['PitLap']], y=[optimal['TotalRaceTime']],
                                     mode='markers', marker=dict(color='#00ff88', size=22, symbol='star')))
            fig.add_vrect(x0=top5['PitLap'].min() - 0.5, x1=top5['PitLap'].max() + 0.5,
                          fillcolor='#00ff88', opacity=0.1, line_width=0)
            fig.update_layout(title=f'🏎️ Pit Window — {driver}', xaxis_title='Pit Lap',
                              yaxis_title='Total Race Time (s)', template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Top 5")
            st.dataframe(top5.reset_index(drop=True), use_container_width=True)

# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.markdown("**Built by Sanjay Chowdary** | [GitHub](https://github.com/sanjay-099/f1-race-intelligence) | Powered by FastF1 + XGBoost")