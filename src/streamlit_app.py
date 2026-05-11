"""
F1 Race Intelligence System — Streamlit Dashboard
Real-time race analytics powered by FastF1 telemetry + ML.
"""
import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import joblib
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

# ── Custom CSS for F1 theme ──────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1a2e; padding: 1rem; border-radius: 10px; }
    h1 { color: #FF1E1E; font-weight: bold; }
    h2 { color: #00d2ff; }
    h3 { color: #ffffff; }
    .css-1d391kg { background-color: #16213e; }
</style>
""", unsafe_allow_html=True)

# ── Resolve project paths ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
MODELS_DIR = ROOT / "models"

# ── Load FastF1 + Models (cached) ────────────────────────
@st.cache_resource
def load_session():
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    session = fastf1.get_session(2024, 'Bahrain', 'R')
    session.load()
    return session

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

# ── Load Data ────────────────────────────────────────────
with st.spinner("Loading race data..."):
    session = load_session()
    tire_model, compound_encoder, driver_encoder, iso_forest, scaler = load_models()

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=150)
st.sidebar.title("🏁 Race Selection")
st.sidebar.markdown(f"**Race:** {session.event['EventName']}")
st.sidebar.markdown(f"**Year:** {session.event.year}")
st.sidebar.markdown(f"**Circuit:** {session.event['Location']}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Navigate",
    ["🏠 Race Overview", "📡 Telemetry Analysis", "🚨 Incident Detection", "🎯 Strategy Optimizer"]
)

# ── Page Routing (placeholders for now) ──────────────────
if page == "🏠 Race Overview":
    st.header("🏠 Race Overview")
    st.info("✅ Race data loaded successfully!")
    
    col1, col2, col3, col4 = st.columns(4)
    laps = session.laps
    col1.metric("Total Laps", len(laps))
    col2.metric("Drivers", laps['Driver'].nunique())
    col3.metric("Race Date", str(session.date.date()))
    col4.metric("Location", session.event['Location'])
    
elif page == "📡 Telemetry Analysis":
    st.header("📡 Telemetry Analysis")
    st.markdown("Compare driver telemetry from their fastest lap")
    
    # Driver selection
    drivers_list = sorted(session.laps['Driver'].unique().tolist())
    
    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.selectbox("🏎️ Driver 1", drivers_list, index=drivers_list.index('HAM'))
    with col2:
        driver2 = st.selectbox("🏎️ Driver 2 (comparison)", ['None'] + drivers_list, index=0)
    
    # Get telemetry
    try:
        lap1 = session.laps.pick_drivers(driver1).pick_fastest()
        tel1 = lap1.get_car_data().add_distance()
        
        # Metrics for driver 1
        st.subheader(f"📊 {driver1} — Fastest Lap Stats")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Lap Time", str(lap1['LapTime']).split('.')[0][-5:])
        m2.metric("Lap Number", int(lap1['LapNumber']))
        m3.metric("Compound", lap1['Compound'])
        m4.metric("Max Speed", f"{tel1['Speed'].max():.0f} km/h")
        
        # Build chart
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Speed (km/h)", "Throttle (%)", "Gear"),
            vertical_spacing=0.08
        )
        
        # Driver 1 speed
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Speed'],
            name=driver1, line=dict(color='#00d2ff', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Throttle'],
            name=f'{driver1} Throttle', line=dict(color='#00ff88', width=2),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['nGear'],
            name=f'{driver1} Gear', line=dict(color='#ff6b6b', width=2),
            showlegend=False
        ), row=3, col=1)
        
        # If driver 2 selected, add comparison
        if driver2 != 'None':
            lap2 = session.laps.pick_drivers(driver2).pick_fastest()
            tel2 = lap2.get_car_data().add_distance()
            
            fig.add_trace(go.Scatter(
                x=tel2['Distance'], y=tel2['Speed'],
                name=driver2, line=dict(color='#FF1E1E', width=2, dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=tel2['Distance'], y=tel2['Throttle'],
                line=dict(color='#ffaa00', width=2, dash='dash'),
                showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=tel2['Distance'], y=tel2['nGear'],
                line=dict(color='#aa00ff', width=2, dash='dash'),
                showlegend=False
            ), row=3, col=1)
            
            # Comparison metrics
            st.subheader(f"⚔️ {driver1} vs {driver2}")
            c1, c2, c3 = st.columns(3)
            delta_time = (lap1['LapTime'] - lap2['LapTime']).total_seconds()
            c1.metric("Time Delta", f"{delta_time:+.3f}s", 
                      delta=f"{driver1} {'slower' if delta_time > 0 else 'faster'}")
            c2.metric(f"{driver1} Max Speed", f"{tel1['Speed'].max():.0f} km/h")
            c3.metric(f"{driver2} Max Speed", f"{tel2['Speed'].max():.0f} km/h")
        
        fig.update_layout(
            template='plotly_dark',
            height=700,
            hovermode='x unified',
            showlegend=True
        )
        fig.update_xaxes(title_text="Distance (m)", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not load telemetry: {e}")
    
elif page == "🚨 Incident Detection":
    st.header("🚨 Incident Detection")
    st.markdown("ML-powered anomaly detection on lap times — flags incidents, pit push laps, and tire cliff events")
    
    # Get clean lap data
    laps = session.laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    clean_laps = laps[
        (laps['IsAccurate'] == True) &
        (laps['LapTimeSeconds'] > 85) &
        (laps['LapTimeSeconds'] < 120)
    ].copy()
    
    # Build features for Isolation Forest
    features_df = clean_laps[['LapTimeSeconds', 'TyreLife', 'LapNumber']].dropna()
    X_scaled = scaler.transform(features_df)
    predictions = iso_forest.predict(X_scaled)
    
    # Attach predictions
    clean_laps_iso = clean_laps.dropna(subset=['LapTimeSeconds', 'TyreLife', 'LapNumber']).copy()
    clean_laps_iso['IsAnomaly'] = (predictions == -1)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Laps", len(clean_laps_iso))
    col2.metric("🚨 Incidents Detected", clean_laps_iso['IsAnomaly'].sum())
    col3.metric("Detection Rate", f"{clean_laps_iso['IsAnomaly'].sum() / len(clean_laps_iso) * 100:.1f}%")
    col4.metric("Drivers Flagged", clean_laps_iso[clean_laps_iso['IsAnomaly']]['Driver'].nunique())
    
    st.markdown("---")
    
    # Driver filter
    all_drivers = sorted(clean_laps_iso['Driver'].unique().tolist())
    selected_drivers = st.multiselect(
        "🏎️ Filter drivers (leave empty to see all)",
        all_drivers,
        default=[]
    )
    
    plot_df = clean_laps_iso if not selected_drivers else clean_laps_iso[clean_laps_iso['Driver'].isin(selected_drivers)]
    
    # Build incident map
    fig = go.Figure()
    
    # Sort drivers by avg lap time
    driver_order = plot_df.groupby('Driver')['LapTimeSeconds'].mean().sort_values().index.tolist()
    
    compound_colors = {'SOFT': '#FF3333', 'MEDIUM': '#FFFF00', 'HARD': '#CCCCCC'}
    
    for driver in driver_order:
        d_laps = plot_df[plot_df['Driver'] == driver]
        
        normal = d_laps[~d_laps['IsAnomaly']]
        fig.add_trace(go.Scatter(
            x=normal['LapNumber'],
            y=[driver] * len(normal),
            mode='markers',
            marker=dict(
                size=6,
                color=[compound_colors.get(c, '#888') for c in normal['Compound']],
                opacity=0.5
            ),
            name=driver,
            showlegend=False,
            hovertemplate=f'{driver}<br>Lap %{{x}}<br>Time: %{{customdata:.3f}}s<extra></extra>',
            customdata=normal['LapTimeSeconds']
        ))
        
        anom = d_laps[d_laps['IsAnomaly']]
        if len(anom) > 0:
            fig.add_trace(go.Scatter(
                x=anom['LapNumber'],
                y=[driver] * len(anom),
                mode='markers',
                marker=dict(
                    size=14,
                    color='red',
                    symbol='circle-open',
                    line=dict(width=3, color='red')
                ),
                showlegend=False,
                hovertemplate=f'<b>🚨 INCIDENT</b><br>{driver}<br>Lap %{{x}}<br>Time: %{{customdata:.3f}}s<extra></extra>',
                customdata=anom['LapTimeSeconds']
            ))
    
    fig.update_layout(
        title='🚨 Race-Wide Incident Map<br><sub>Grey dots = Normal | Red circles = Detected Incidents | Color = Tyre Compound</sub>',
        xaxis_title='Lap Number',
        yaxis_title='Driver',
        template='plotly_dark',
        height=700,
        yaxis=dict(categoryorder='array', categoryarray=driver_order[::-1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Incident table
    st.subheader("📋 Detailed Incident Report")
    incidents = clean_laps_iso[clean_laps_iso['IsAnomaly']][
        ['Driver', 'LapNumber', 'LapTimeSeconds', 'Compound', 'TyreLife', 'Stint']
    ].sort_values('LapNumber').reset_index(drop=True)
    incidents['LapTimeSeconds'] = incidents['LapTimeSeconds'].round(3)
    incidents['LapNumber'] = incidents['LapNumber'].astype(int)
    incidents['TyreLife'] = incidents['TyreLife'].astype(int)
    
    st.dataframe(incidents, use_container_width=True, height=400)
    
elif page == "🎯 Strategy Optimizer":
    st.header("🎯 Strategy Optimizer")
    st.markdown("ML-powered pit stop optimization — find the optimal pit window for any driver")
    
    # Helper function — same as Set 5
    def predict_stint(driver, compound, stint_length, start_lap=1):
        driver_enc = driver_encoder.transform([driver])[0]
        compound_enc = compound_encoder.transform([compound])[0]
        
        stint_data = []
        for i in range(1, stint_length + 1):
            race_lap = start_lap + i - 1
            stint_data.append({
                'TyreLife': i,
                'TyreLifeSquared': i ** 2,
                'CompoundEncoded': compound_enc,
                'DriverEncoded': driver_enc,
                'CompoundAge': compound_enc * i,
                'LapNumber': race_lap
            })
        
        stint_df = pd.DataFrame(stint_data)
        predicted_times = tire_model.predict(stint_df)
        
        return pd.DataFrame({
            'StintLap': range(1, stint_length + 1),
            'PredictedLapTime': predicted_times,
            'CumulativeTime': np.cumsum(predicted_times)
        })
    
    # ── User controls ────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        driver = st.selectbox(
            "🏎️ Driver",
            sorted(driver_encoder.classes_),
            index=list(sorted(driver_encoder.classes_)).index('HAM')
        )
    with col2:
        stint1_compound = st.selectbox("🛞 Stint 1 Compound", ['SOFT', 'HARD'], index=0)
    with col3:
        stint2_compound = st.selectbox("🛞 Stint 2 Compound", ['HARD', 'SOFT'], index=0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        race_length = st.number_input("🏁 Race Length (laps)", min_value=30, max_value=80, value=57)
    with col5:
        pit_loss = st.number_input("⏱️ Pit Stop Loss (s)", min_value=15.0, max_value=30.0, value=22.0, step=0.5)
    with col6:
        pit_min, pit_max = st.slider("🎯 Pit Lap Search Range", 5, race_length - 5, (10, 45))
    
    if stint1_compound == stint2_compound:
        st.warning("⚠️ Stint 1 and Stint 2 compounds are the same — F1 rules require 2 different compounds!")
    
    if st.button("🚀 RUN OPTIMIZER", type="primary"):
        with st.spinner(f"Simulating {pit_max - pit_min + 1} strategies for {driver}..."):
            strategies = []
            
            for pit_lap in range(pit_min, pit_max + 1):
                stint1 = predict_stint(driver, stint1_compound, stint_length=pit_lap, start_lap=1)
                stint2 = predict_stint(driver, stint2_compound, stint_length=race_length - pit_lap, start_lap=pit_lap + 1)
                
                total = stint1['CumulativeTime'].iloc[-1] + pit_loss + stint2['CumulativeTime'].iloc[-1]
                
                strategies.append({
                    'PitLap': pit_lap,
                    'Stint1Time': round(stint1['CumulativeTime'].iloc[-1], 3),
                    'Stint2Time': round(stint2['CumulativeTime'].iloc[-1], 3),
                    'TotalRaceTime': round(total, 3)
                })
            
            strategies_df = pd.DataFrame(strategies)
            optimal = strategies_df.loc[strategies_df['TotalRaceTime'].idxmin()]
            top5 = strategies_df.nsmallest(5, 'TotalRaceTime')
            
            # ── Results display ──────────────────────────
            st.success(f"✅ Optimization complete! Best strategy found.")
            
            st.subheader("🏆 OPTIMAL STRATEGY")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Pit Lap", int(optimal['PitLap']))
            r2.metric("Stint 1", f"{int(optimal['PitLap'])} laps {stint1_compound}")
            r3.metric("Stint 2", f"{race_length - int(optimal['PitLap'])} laps {stint2_compound}")
            r4.metric("Total Race Time", f"{optimal['TotalRaceTime']:.1f}s")
            
            st.markdown("---")
            
            # ── Pit window chart ─────────────────────────
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=strategies_df['PitLap'],
                y=strategies_df['TotalRaceTime'],
                mode='lines+markers',
                name='Total Race Time',
                line=dict(color='#00d2ff', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=[optimal['PitLap']],
                y=[optimal['TotalRaceTime']],
                mode='markers',
                name='🏆 Optimal',
                marker=dict(color='#00ff88', size=22, symbol='star',
                            line=dict(width=2, color='white'))
            ))
            
            fig.add_vrect(
                x0=top5['PitLap'].min() - 0.5,
                x1=top5['PitLap'].max() + 0.5,
                fillcolor='#00ff88',
                opacity=0.1,
                line_width=0,
                annotation_text="Strategy Window",
                annotation_position="top left"
            )
            
            fig.update_layout(
                title=f'🏎️ Pit Window — {driver} | {stint1_compound} → {stint2_compound}',
                xaxis_title='Pit Lap',
                yaxis_title='Total Race Time (s)',
                template='plotly_dark',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ── Insights ─────────────────────────────────
            st.subheader("💡 Strategy Insights")
            i1, i2, i3 = st.columns(3)
            i1.info(f"**🎯 Pit Window:** Lap {int(top5['PitLap'].min())} – {int(top5['PitLap'].max())}\n\n5-lap flexibility band")
            i2.warning(f"**⚠️ Risk Zone (Early):** Before Lap {int(top5['PitLap'].min())}\n\nCosts >1s vs optimal")
            i3.warning(f"**⚠️ Risk Zone (Late):** After Lap {int(top5['PitLap'].max())}\n\nCosts >1s vs optimal")
            
            # ── Top 5 table ──────────────────────────────
            st.subheader("📊 Top 5 Strategies")
            st.dataframe(top5.reset_index(drop=True), use_container_width=True)

# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.markdown("**Built by Sanjay Chowdary** | [GitHub](https://github.com/sanjay-099/f1-race-intelligence) | Powered by FastF1 + XGBoost")