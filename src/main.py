"""
F1 Race Intelligence System
FastAPI + Jinja2 full-stack web application
Built by Sanjay Chowdary
"""
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import fastf1
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── App Setup ─────────────────────────────────────────────
app = FastAPI(title="F1 Race Intelligence", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

ROOT = BASE_DIR.parent
CACHE_DIR = ROOT / "data" / "cache"
MODELS_DIR = ROOT / "models"

# ── Load at Startup ───────────────────────────────────────
print("🏎️ Loading F1 session data...")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))
session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load(laps=True, telemetry=True, weather=False, messages=False)
laps = session.laps
print(f"✅ Session loaded: {len(laps)} laps, {laps['Driver'].nunique()} drivers")

print("🤖 Loading ML models...")
tire_model   = joblib.load(MODELS_DIR / "tire_degradation_model.pkl")
compound_enc = joblib.load(MODELS_DIR / "compound_encoder.pkl")
driver_enc   = joblib.load(MODELS_DIR / "driver_encoder.pkl")
iso_forest   = joblib.load(MODELS_DIR / "incident_isolation_forest.pkl")
scaler       = joblib.load(MODELS_DIR / "incident_scaler.pkl")
print("✅ Models loaded")

# ── Helper ────────────────────────────────────────────────
def predict_stint(driver, compound, stint_len, start_lap=1):
    d_enc = driver_enc.transform([driver])[0]
    c_enc = compound_enc.transform([compound])[0]
    data = [{
        'TyreLife': i, 'TyreLifeSquared': i**2,
        'CompoundEncoded': c_enc, 'DriverEncoded': d_enc,
        'CompoundAge': c_enc*i, 'LapNumber': start_lap+i-1
    } for i in range(1, stint_len+1)]
    return tire_model.predict(pd.DataFrame(data))

# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    lap_counts = laps.groupby('Driver').size().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=lap_counts.index.tolist(),
        y=lap_counts.values.tolist(),
        marker_color='#00d2ff',
        marker_line_color='#0088aa',
        marker_line_width=1
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#16213e',
        plot_bgcolor='#16213e',
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title='Driver',
        yaxis_title='Laps Completed'
    )
    return templates.TemplateResponse(request=request, name="index.html", context={
        "event_name": session.event['EventName'],
        "year": session.event.year,
        "location": session.event['Location'],
        "date": str(session.date.date()),
        "total_laps": len(laps),
        "total_drivers": laps['Driver'].nunique(),
        "chart_json": fig.to_json(),
        "drivers": sorted(laps['Driver'].unique().tolist()),
        "compounds": list(compound_enc.classes_)
    })

@app.get("/telemetry", response_class=HTMLResponse)
async def telemetry(request: Request, driver1: str = "HAM", driver2: str = "None"):
    drivers_list = sorted(laps['Driver'].unique().tolist())
    chart_json = "{}"
    stats = {}
    try:
        lap1 = laps.pick_drivers(driver1).pick_fastest()
        tel1 = lap1.get_car_data().add_distance()
        stats = {
            "driver1": driver1,
            "lap_time": str(lap1['LapTime']).split('.')[0][-5:],
            "lap_number": int(lap1['LapNumber']),
            "compound": lap1['Compound'],
            "max_speed": f"{tel1['Speed'].max():.0f}"
        }
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Speed (km/h)", "Throttle (%)", "Gear"),
            vertical_spacing=0.08
        )
        fig.add_trace(go.Scatter(
            x=tel1['Distance'].tolist(), y=tel1['Speed'].tolist(),
            name=driver1, line=dict(color='#00d2ff', width=2.5)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=tel1['Distance'].tolist(), y=tel1['Throttle'].tolist(),
            line=dict(color='#00ff88', width=2), showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=tel1['Distance'].tolist(), y=tel1['nGear'].tolist(),
            line=dict(color='#ff6b6b', width=2), showlegend=False
        ), row=3, col=1)
        if driver2 and driver2 != "None":
            lap2 = laps.pick_drivers(driver2).pick_fastest()
            tel2 = lap2.get_car_data().add_distance()
            delta = (lap1['LapTime'] - lap2['LapTime']).total_seconds()
            stats["driver2"] = driver2
            stats["delta"] = f"{delta:+.3f}"
            stats["max_speed2"] = f"{tel2['Speed'].max():.0f}"
            fig.add_trace(go.Scatter(
                x=tel2['Distance'].tolist(), y=tel2['Speed'].tolist(),
                name=driver2, line=dict(color='#FF1E1E', width=2.5, dash='dash')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=tel2['Distance'].tolist(), y=tel2['Throttle'].tolist(),
                line=dict(color='#ffaa00', width=2, dash='dash'), showlegend=False
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=tel2['Distance'].tolist(), y=tel2['nGear'].tolist(),
                line=dict(color='#aa00ff', width=2, dash='dash'), showlegend=False
            ), row=3, col=1)
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#16213e',
            plot_bgcolor='#16213e',
            height=700,
            hovermode='x unified',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_xaxes(title_text="Distance (m)", row=3, col=1)
        chart_json = fig.to_json()
    except Exception as e:
        print(f"Telemetry error: {e}")
    return templates.TemplateResponse(request=request, name="telemetry.html", context={
        "drivers": drivers_list,
        "driver1": driver1,
        "driver2": driver2,
        "stats": stats,
        "chart_json": chart_json
    })

@app.get("/incidents", response_class=HTMLResponse)
async def incidents(request: Request):
    laps_copy = laps.copy()
    laps_copy['LapTimeSeconds'] = laps_copy['LapTime'].dt.total_seconds()
    clean = laps_copy[
        (laps_copy['IsAccurate'] == True) &
        (laps_copy['LapTimeSeconds'] > 85) &
        (laps_copy['LapTimeSeconds'] < 120)
    ].copy()
    feats = clean[['LapTimeSeconds', 'TyreLife', 'LapNumber']].dropna()
    X_scaled = scaler.transform(feats)
    preds = iso_forest.predict(X_scaled)
    clean_iso = clean.dropna(subset=['LapTimeSeconds', 'TyreLife', 'LapNumber']).copy()
    clean_iso['IsAnomaly'] = (preds == -1)
    driver_order = clean_iso.groupby('Driver')['LapTimeSeconds'].mean().sort_values().index.tolist()
    compound_colors = {'SOFT': '#FF3333', 'MEDIUM': '#FFFF00', 'HARD': '#CCCCCC'}
    fig = go.Figure()
    for drv in driver_order:
        d = clean_iso[clean_iso['Driver'] == drv]
        normal = d[~d['IsAnomaly']]
        fig.add_trace(go.Scatter(
            x=normal['LapNumber'].tolist(), y=[drv]*len(normal),
            mode='markers',
            marker=dict(size=6, color=[compound_colors.get(c,'#888') for c in normal['Compound']], opacity=0.5),
            showlegend=False,
            hovertemplate=f'{drv}<br>Lap %{{x}}<br>%{{customdata:.3f}}s<extra></extra>',
            customdata=normal['LapTimeSeconds'].tolist()
        ))
        anom = d[d['IsAnomaly']]
        if len(anom) > 0:
            fig.add_trace(go.Scatter(
                x=anom['LapNumber'].tolist(), y=[drv]*len(anom),
                mode='markers',
                marker=dict(size=14, color='red', symbol='circle-open', line=dict(width=3, color='red')),
                showlegend=False,
                hovertemplate=f'<b>🚨 INCIDENT</b><br>{drv}<br>Lap %{{x}}<br>%{{customdata:.3f}}s<extra></extra>',
                customdata=anom['LapTimeSeconds'].tolist()
            ))
    fig.update_layout(
        title='Race-Wide Incident Map — 2024 Bahrain GP',
        template='plotly_dark',
        paper_bgcolor='#16213e',
        plot_bgcolor='#16213e',
        height=750,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(categoryorder='array', categoryarray=driver_order[::-1])
    )
    incidents_df = clean_iso[clean_iso['IsAnomaly']][[
        'Driver', 'LapNumber', 'LapTimeSeconds', 'Compound', 'TyreLife'
    ]].sort_values('LapNumber').reset_index(drop=True)
    incidents_df['LapNumber'] = incidents_df['LapNumber'].astype(int)
    incidents_df['TyreLife'] = incidents_df['TyreLife'].astype(int)
    incidents_df['LapTimeSeconds'] = incidents_df['LapTimeSeconds'].round(3)
    return templates.TemplateResponse(request=request, name="incidents.html", context={
        "total_laps": len(clean_iso),
        "total_incidents": int(clean_iso['IsAnomaly'].sum()),
        "detection_rate": f"{clean_iso['IsAnomaly'].sum()/len(clean_iso)*100:.1f}",
        "drivers_flagged": int(clean_iso[clean_iso['IsAnomaly']]['Driver'].nunique()),
        "chart_json": fig.to_json(),
        "incidents": incidents_df.to_dict(orient='records')
    })

@app.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request):
    return templates.TemplateResponse(request=request, name="strategy.html", context={
        "drivers": sorted(driver_enc.classes_.tolist()),
        "compounds": compound_enc.classes_.tolist()
    })

class StrategyRequest(BaseModel):
    driver: str = "HAM"
    stint1: str = "SOFT"
    stint2: str = "HARD"
    race_length: int = 57
    pit_loss: float = 22.0
    pit_min: int = 10
    pit_max: int = 45

@app.post("/strategy/optimize")
async def optimize_strategy(req: StrategyRequest):
    if req.stint1 == req.stint2:
        return {"error": "Compounds must differ"}
    strategies = []
    for pit_lap in range(req.pit_min, req.pit_max + 1):
        s1 = predict_stint(req.driver, req.stint1, pit_lap, 1)
        s2 = predict_stint(req.driver, req.stint2, req.race_length - pit_lap, pit_lap + 1)
        total = float(sum(s1)) + req.pit_loss + float(sum(s2))
        strategies.append({
            "pit_lap": pit_lap,
            "stint1_time": round(float(sum(s1)), 3),
            "stint2_time": round(float(sum(s2)), 3),
            "total": round(total, 3)
        })
    strategies.sort(key=lambda x: x['total'])
    optimal = strategies[0]
    top5 = strategies[:5]
    pit_window = [s['pit_lap'] for s in top5]
    all_pits = [s['pit_lap'] for s in strategies]
    all_totals = [s['total'] for s in strategies]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_pits, y=all_totals,
        mode='lines+markers',
        line=dict(color='#00d2ff', width=3),
        marker=dict(size=8), name='Total Race Time'
    ))
    fig.add_trace(go.Scatter(
        x=[optimal['pit_lap']], y=[optimal['total']],
        mode='markers',
        marker=dict(color='#00ff88', size=22, symbol='star'),
        name='🏆 Optimal'
    ))
    fig.add_vrect(
        x0=min(pit_window)-0.5, x1=max(pit_window)+0.5,
        fillcolor='#00ff88', opacity=0.1, line_width=0,
        annotation_text="Strategy Window"
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#16213e',
        plot_bgcolor='#16213e',
        height=450,
        margin=dict(l=20, r=20, t=40, b=40),
        xaxis_title='Pit Lap',
        yaxis_title='Total Race Time (s)',
        title=f'Pit Window — {req.driver} | {req.stint1} → {req.stint2}'
    )
    return {
        "optimal": optimal,
        "top5": top5,
        "pit_window_start": min(pit_window),
        "pit_window_end": max(pit_window),
        "chart_json": fig.to_json(),
        "driver": req.driver,
        "strategy": f"{req.stint1} → {req.stint2}"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "race": session.event['EventName'], "laps": len(laps)}