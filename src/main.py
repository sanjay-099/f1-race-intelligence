"""
F1 Race Intelligence System
FastAPI + Jinja2 full-stack web application
Built by Sanjay Chowdary
"""
from fastapi import FastAPI, Request, Body, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from src.tabnet_model import TabNetPredictor
import fastf1
import requests
import asyncio

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_original_send = requests.Session.send
def _custom_send(self, request, **kwargs):
    # Short timeout for Ergast/jolpi — it's unreliable, fail fast
    if 'jolpi.ca' in str(request.url) or 'ergast' in str(request.url):
        kwargs['timeout'] = 5.0  # fail fast, use cache
    else:
        kwargs['timeout'] = 30.0
    return _original_send(self, request, **kwargs)
requests.Session.send = _custom_send

_original_init = requests.Session.__init__
def _custom_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    # NO retries on timeout — just fail fast and use cache
    adapter = HTTPAdapter(max_retries=0)
    self.mount('http://', adapter)
    self.mount('https://', adapter)
requests.Session.__init__ = _custom_init
import pandas as pd
import numpy as np
import joblib
import builtins
builtins_round = builtins.round
import anthropic
from dotenv import load_dotenv
import os
from src.race_context import build_race_context
load_dotenv()
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import builtins
builtins_round = builtins.round
class SessionNotAvailableError(Exception):
    """Raised when a session has no data yet (future race)."""
    def __init__(self, year, round_number, session_type):
        self.year         = year
        self.round_number = round_number
        self.session_type = session_type
        super().__init__(f"No data available for {year} Round {round_number} {session_type}")

# ── App Setup ─────────────────────────────────────────────
app = FastAPI(
    title="F1 Race Intelligence API",
    version="2.0.0",
    description="""
## F1 Race Intelligence System
Production-grade Formula 1 analytics platform — FastF1 telemetry + XGBoost + Multi-Agent LLM (Claude Sonnet 4.6).

Built by **Sanjay Chowdary** · MS Data Science · University of Alabama at Birmingham

📄 [Technical Methodology](https://github.com/sanjay-099/f1-race-intelligence/blob/main/docs/F1_Race_Intelligence_Methodology.pdf)

### Key Capabilities
- **Race Predictions** — Quali→Race model trained on 1,083 historical samples (MAE ~2.1-2.5 positions)
- **Tyre Degradation** — XGBoost regressor (R²=0.962, MAE=1.229s, 170,351 laps)
- **Incident Detection** — Dual IsolationForest (race + practice sessions)
- **Driver Embeddings** — PCA on 8 telemetry-derived style features, 44 drivers indexed
- **Monte Carlo Simulation** — 500+ iteration race strategy optimizer
- **Circuit & Constructor Intelligence** — Historical analysis across 31 circuits, 2018-2026

### Data
- FastF1 telemetry API (2018-2026)
- 325,983 total laps across 31 circuits
- 10.5 GB FastF1 cache

    """,
    contact={
        "name": "APP",
        "url": "https://huggingface.co/spaces/sanjay1103/f1-race-intelligence",
    },
    openapi_tags=[
        {"name": "Analysis", "description": "Race telemetry, incidents, strategy analysis"},
        {"name": "Intelligence", "description": "Predictions, AI analyst, counterfactual analysis"},
        {"name": "Deep Dive", "description": "Session compare, tyre lab, driver styles, circuit & constructor intelligence"},
        {"name": "System", "description": "Pipeline monitor, model evaluation, health"},
    ]
)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

from fastapi.responses import HTMLResponse as _HTMLResponse
from fastapi import Request as _Request

@app.exception_handler(SessionNotAvailableError)
async def session_not_available_handler(request: _Request, exc: SessionNotAvailableError):
    seasons  = get_available_seasons()
    schedule = get_season_schedule(exc.year)
    return templates.TemplateResponse(request=request, name="no_data.html", status_code=200, context={
        "seasons":          seasons,
        "schedule":         schedule,
        "selected_year":    exc.year,
        "selected_round":   exc.round_number,
        "selected_session": exc.session_type,
        "active":           "none",
        "year":             exc.year,
        "round_number":     exc.round_number,
        "session_type":     exc.session_type,
    })

@app.exception_handler(Exception)
async def general_exception_handler(request: _Request, exc: Exception):
    from fastf1.core import DataNotLoadedError
    if isinstance(exc, (SessionNotAvailableError, DataNotLoadedError)):
        q = request.query_params
        try:
            year_int  = int(q.get('year'))
            round_int = int(q.get('round'))
        except:
            latest = get_latest_race()
            year_int, round_int = latest['year'], latest['round']
        session_type = q.get('session_type', 'R')
        fake_exc = SessionNotAvailableError(year_int, round_int, session_type)
        return await session_not_available_handler(request, fake_exc)
        
    import traceback
    print(f"Unhandled error on {request.url.path}: {exc}")
    traceback.print_exc()

    # Try to preserve the year/round the user actually requested
    q = request.query_params
    req_year   = q.get('year')
    req_round  = q.get('round')
    req_session = q.get('session_type', 'R')

    try:
        if req_year and req_round:
            year_int  = int(req_year)
            round_int = int(req_round)
        else:
            latest    = get_latest_race()
            year_int  = latest['year']
            round_int = latest['round']

        seasons  = get_available_seasons()
        schedule = get_season_schedule(year_int)
        context = {
            "seasons":          seasons,
            "schedule":         schedule,
            "selected_year":    year_int,
            "selected_round":   round_int,
            "selected_session": req_session,
            "active":           "none",
            "year":             year_int,
            "round_number":     round_int,
            "session_type":     req_session,
        }
        if not isinstance(exc, SessionNotAvailableError):
            context["error_detail"] = str(exc)
            
        return templates.TemplateResponse(request=request, name="no_data.html", status_code=200, context=context)
    except:
        return _HTMLResponse(content=f"<h1>Error</h1><p>{exc}</p>", status_code=500)
ROOT = BASE_DIR.parent
CACHE_DIR = ROOT / "data" / "cache"
MODELS_DIR = ROOT / "models"

# ── Incident Detection Models (pre-trained) ───────────────
iso_forest = joblib.load(MODELS_DIR / "incident_isolation_forest.pkl")
scaler     = joblib.load(MODELS_DIR / "incident_scaler.pkl")
iso_forest_prac = joblib.load(MODELS_DIR / "incident_isolation_forest_practice.pkl")
scaler_prac     = joblib.load(MODELS_DIR / "incident_scaler_practice.pkl")

# ── Session + Model Cache ─────────────────────────────────
session_cache = {}
model_cache = {}


from src.tabnet_model import TabNetPredictor, PostRacePredictor
tabnet_cache = {}
postrace_cache = {}

def get_postrace_model(year: int, round_number: int, session_type: str = 'R'):
    """
    Train XGBoost on previous races, predict on selected race.
    Avoids overfitting by never testing on training data.
    """
    cache_key = f"postrace_{year}_{round_number}_{session_type}"
    if cache_key in postrace_cache:
        return postrace_cache[cache_key]

    print(f"📊 Building post-race model for {year} Round {round_number}...")

    # Load previous races for training
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    # Only use races BEFORE the selected round
    prev_rounds = schedule[
        schedule['RoundNumber'] < round_number
    ]['RoundNumber'].tolist()

    # If no previous races, use last 3 races from previous season
    if len(prev_rounds) == 0:
        print(f"⚠️ No previous races in {year}, using {year-1} data")
        schedule_prev = fastf1.get_event_schedule(year-1, include_testing=False)
        prev_rounds_prev = schedule_prev['RoundNumber'].tolist()[-3:]
        train_sessions = []
        for r in prev_rounds_prev:
            try:
                s = load_race_session(year-1, int(r), session_type)
                train_sessions.append(s)
            except:
                continue
    else:
        # Use up to last 5 races for training
        train_rounds = prev_rounds[-5:]
        train_sessions = []
        for r in train_rounds:
            try:
                s = load_race_session(year, int(r), session_type)
                train_sessions.append(s)
            except:
                continue

    # Load selected race for prediction
    test_session = load_race_session(year, round_number, session_type)

    # Train + predict
    predictor = PostRacePredictor()
    metrics   = predictor.train_multi(train_sessions, test_session)

    postrace_cache[cache_key] = {
        'predictor': predictor,
        'metrics':   metrics
    }
    print(f"✅ Post-race model ready: {metrics}")
    return postrace_cache[cache_key]

def mean_absolute_error_for_race(xgb_data, year, round_number, session_type):
    """Calculate MAE for XGBoost on race data."""
    try:
        session = load_race_session(year, round_number, session_type)
        laps = session.laps.copy()
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        clean = laps[
            (laps['IsAccurate'] == True) &
            (laps['LapTimeSeconds'] > 60) &
            (laps['LapTimeSeconds'] < 150)
        ].dropna(subset=['LapTimeSeconds', 'TyreLife', 'LapNumber', 'Compound', 'Driver'])

        clean = clean.copy()
        clean['CompoundEncoded'] = xgb_data['compound_enc'].transform(clean['Compound'])
        clean['DriverEncoded']   = xgb_data['driver_enc'].transform(clean['Driver'])
        clean['CircuitEncoded']  = xgb_data['circuit_enc'].transform([session.event['Location']] * len(clean))
        clean['TyreLifeSquared'] = clean['TyreLife'] ** 2
        clean['CompoundAge']     = clean['CompoundEncoded'] * clean['TyreLife']
        clean['RegEra']          = 2 if year >= 2026 else (1 if year >= 2022 else 0)

        features = ['TyreLife', 'TyreLifeSquared', 'CompoundEncoded',
                    'DriverEncoded', 'CompoundAge', 'LapNumber','RegEra', 'CircuitEncoded']
        preds = xgb_data['model'].predict(clean[features])
        return mean_absolute_error(clean['LapTimeSeconds'], preds)
    except:
        return 0.0
# ── Race Schedule Helper ──────────────────────────────────
def get_available_seasons():
    """Returns all seasons FastF1 supports including current year."""
    from datetime import datetime
    current_year = datetime.now().year
    # FastF1 supports from 2018 onwards
    return list(range(2018, current_year + 1))

def get_season_schedule(year: int) -> list:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = []
        for _, event in schedule.iterrows():
            races.append({
                "round": int(event['RoundNumber']),
                "name": event['EventName'],
                "country": event['Country'],
                "location": event['Location'],
            })
        return races
    except Exception as e:
        print(f"Schedule error {year}: {e}")
        return []
def get_latest_race() -> dict:
    """Returns the most recent completed race."""
    try:
        from datetime import datetime
        current_year = datetime.now().year
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        schedule = fastf1.get_event_schedule(current_year, include_testing=False)
        today = pd.Timestamp.now()
        past_races = schedule[schedule['EventDate'] < today]
        if len(past_races) == 0:
            # Try previous year
            schedule = fastf1.get_event_schedule(current_year - 1, include_testing=False)
            past_races = schedule
        latest = past_races.iloc[-1]
        return {
            "year": int(latest['EventDate'].year),
            "round": int(latest['RoundNumber']),
            "session_type": "R"
        }
    except Exception as e:
        print(f"Error getting latest race: {e}")
        return {"year": 2025, "round": 1, "session_type": "R"}        
def get_race_status(year: int, round_number: int, session_type: str = 'R') -> dict:
    """
    Determine prediction mode based on session type and race status.
    
    Pre-race modes:
    - FP1/FP2/FP3 → predict based on practice pace
    - Q/SQ → predict race order based on qualifying times  
    - S/SS → predict sprint result
    - R → predict race result
    
    Post-race modes (Sprint + Race only):
    - Pure pace prediction ignoring incidents/DNFs
    """
    from datetime import datetime, timezone
    import requests

    now = datetime.now(timezone.utc)

    # Practice sessions → always pre-race mode
    if session_type in ['FP1', 'FP2', 'FP3']:
        return {
            "status": "practice",
            "mode": "pre_race",
            "session_label": "Practice Session",
            "description": "Based on practice pace, who looks fastest?"
        }

    # Qualifying → pre-race mode
    if session_type in ['Q', 'SQ']:
        return {
            "status": "qualifying",
            "mode": "pre_race",
            "session_label": "Qualifying",
            "description": "Based on qualifying times, predicted race order"
        }

    try:
        # Check if session has happened
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        event    = schedule[schedule['RoundNumber'] == round_number]

        if len(event) == 0:
            return {"status": "unknown", "mode": "post_race",
                    "session_label": "Race", "description": ""}

        event_date = pd.Timestamp(event.iloc[0]['EventDate'])
        event_date = event_date.tz_localize('UTC') \
                     if event_date.tzinfo is None else event_date
        now_ts = pd.Timestamp.now(tz='UTC')

        if now_ts > event_date + pd.Timedelta(days=1):
            # Race completed
            if session_type in ['S', 'SS']:
                return {
                    "status": "completed",
                    "mode": "post_race",
                    "session_label": "Sprint",
                    "description": "Pure pace prediction — where drivers would finish based on pace alone"
                }
            return {
                "status": "completed",
                "mode": "post_race",
                "session_label": "Race",
                "description": "Pure pace prediction — where drivers would finish based on pace alone, ignoring incidents and DNFs"
            }
        else:
            # Upcoming
            label = "Sprint" if session_type in ['S', 'SS'] else "Race"
            return {
                "status": "upcoming",
                "mode": "pre_race",
                "session_label": label,
                "description": f"Predicted {label} result"
            }

    except Exception as e:
        print(f"Race status error: {e}")
        return {"status": "unknown", "mode": "post_race",
                "session_label": "Race", "description": ""}
# ── Session Loader ────────────────────────────────────────
def load_race_session(year: int, round_number: int, session_type: str = 'R', telemetry: bool = True):
    if not session_type or session_type == 'None':
        session_type = 'R'
    cache_key = f"{year}_{round_number}_{session_type}"
    if cache_key in session_cache:
        cached = session_cache[cache_key]
        # If telemetry is requested but cached session lacks it, reload
        if telemetry and not hasattr(cached, '_car_data'):
            del session_cache[cache_key]
        else:
            return cached
    print(f"🏎️ Loading: {year} Round {round_number} {session_type}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=telemetry, weather=False, messages=True)
    except Exception as e:
        raise SessionNotAvailableError(year, round_number, session_type) from e

    try:
        if session.laps is None or len(session.laps) == 0:
            raise SessionNotAvailableError(year, round_number, session_type)
    except Exception as e:
        if not isinstance(e, SessionNotAvailableError):
            raise SessionNotAvailableError(year, round_number, session_type) from e
        raise

    session_cache[cache_key] = session
    print(f"✅ Loaded: {session.event['EventName']} {year}")
    return session

# ── Dynamic Model Trainer ─────────────────────────────────
def get_race_model(year: int, round_number: int, session_type: str = 'R'):
    cache_key = f"model_{year}_{round_number}_{session_type}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    print(f"🤖 Training model for {year} Round {round_number}...")
    session = load_race_session(year, round_number, session_type)
    laps = session.laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    clean = laps[
        (laps['IsAccurate'] == True) &
        (laps['LapTimeSeconds'] > 60) &
        (laps['LapTimeSeconds'] < 150)
    ].dropna(subset=['LapTimeSeconds', 'TyreLife', 'LapNumber', 'Compound', 'Driver'])

    drivers   = sorted(clean['Driver'].unique().tolist())
    compounds = sorted(clean['Compound'].unique().tolist())
    d_enc = LabelEncoder().fit(drivers)
    c_enc = LabelEncoder().fit(compounds)

    clean = clean.copy()
    clean['CompoundEncoded'] = c_enc.transform(clean['Compound'])
    clean['DriverEncoded']   = d_enc.transform(clean['Driver'])
    clean['TyreLifeSquared'] = clean['TyreLife'] ** 2
    clean['CompoundAge']     = clean['CompoundEncoded'] * clean['TyreLife']

    features = ['TyreLife', 'TyreLifeSquared', 'CompoundEncoded',
                'DriverEncoded', 'CompoundAge', 'LapNumber']

    model = XGBRegressor(
        n_estimators=200, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        random_state=42, verbosity=0
    )
    model.fit(clean[features], clean['LapTimeSeconds'])

    result = {
        'model': model, 'driver_enc': d_enc,
        'compound_enc': c_enc, 'drivers': drivers, 'compounds': compounds
    }
    model_cache[cache_key] = result
    print(f"✅ Model ready: {drivers} | {compounds}")
    return result

# ── Load Default Session at Startup ──────────────────────
print("🏎️ Loading latest race at startup...")
_latest = get_latest_race()
try:
    load_race_session(_latest['year'], _latest['round'], _latest['session_type'], telemetry=False)
    print(f"✅ Latest race loaded: {_latest['year']} Round {_latest['round']}")
except Exception as e:
    print(f"⚠️ Startup preload skipped: {e}")
# ════════════════════════════════════════════════════════
# SCHEDULE API ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/api/seasons", tags=["System"])
async def api_seasons():
    return {"seasons": get_available_seasons()}

@app.get("/api/schedule/{year}", tags=["System"])
async def api_schedule(year: int):
    return {"year": year, "races": get_season_schedule(year)}

@app.get("/api/race-data/{year}/{round_number}/{session_type}", tags=["Analysis"])
async def get_race_data(year: int, round_number: int, session_type: str = 'R'):
    try:
        session = load_race_session(year, round_number, session_type)
        laps = session.laps
        drivers   = sorted(laps['Driver'].unique().tolist())
        compounds = sorted(laps['Compound'].dropna().unique().tolist())
        race_length = int(laps['LapNumber'].max())
        return {
            "status": "success",
            "event_name": session.event['EventName'],
            "year": year,
            "round": round_number,
            "location": session.event['Location'],
            "date": str(session.date.date()),
            "drivers": drivers,
            "compounds": compounds,
            "race_length": race_length,
            "total_laps_recorded": len(laps)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ════════════════════════════════════════════════════════
# PAGE ROUTES
# ════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, year: str = None, round: str = None, session_type: str = None):
    try:
        year  = int(year)  if year  and year  != 'None' else None
        round = int(round) if round and round != 'None' else None
    except:
        year, round = None, None
    if session_type == 'None':
        session_type = None
    if year is None or round is None:
        latest = get_latest_race()
        year = latest['year']
        round = latest['round']
        session_type = session_type or latest['session_type']
    session  = load_race_session(year, round, session_type)
    laps     = session.laps
    seasons  = get_available_seasons()
    schedule = get_season_schedule(year)

    lap_counts = laps.groupby('Driver').size().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=lap_counts.index.tolist(), y=lap_counts.values.tolist(),
        marker_color='#00d2ff', marker_line_color='#0088aa', marker_line_width=1
    ))

    # ── Compound usage per driver ─────────────────────────────
    compound_colors = {
        'SOFT': '#FF3333',
        'MEDIUM': '#FFFF00', 
        'HARD': '#CCCCCC',
        'INTERMEDIATE': '#39B54A',
        'WET': '#0067FF'
    }

    compound_data = laps.groupby(['Driver', 'Compound']).size().reset_index(name='Laps')
    compounds_used = sorted(compound_data['Compound'].unique().tolist())
    drivers_sorted = lap_counts.index.tolist()

    fig2 = go.Figure()
    for compound in compounds_used:
        df = compound_data[compound_data['Compound'] == compound]
        # Align to driver order
        lap_vals = []
        for drv in drivers_sorted:
            row = df[df['Driver'] == drv]
            lap_vals.append(int(row['Laps'].values[0]) if len(row) > 0 else 0)
        
        fig2.add_trace(go.Bar(
            name=compound,
            x=drivers_sorted,
            y=lap_vals,
            marker_color=compound_colors.get(compound, '#888888')
        ))

    fig2.update_layout(
        barmode='stack',
        template='plotly_dark',
        paper_bgcolor='#16213e',
        plot_bgcolor='#16213e',
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title='Driver',
        yaxis_title='Laps',
        legend_title='Compound'
    )
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#16213e', plot_bgcolor='#16213e',
        height=350, margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title='Driver', yaxis_title='Laps Completed'
    )
    # ── Pace Heatmap ─────────────────────────────────────────
    try:
        laps_copy = laps.copy()
        laps_copy['LapTimeSeconds'] = laps_copy['LapTime'].dt.total_seconds()
        heat = laps_copy[
            (laps_copy['LapTimeSeconds'] > 60) &
            (laps_copy['LapTimeSeconds'] < 200) &
            (laps_copy['IsAccurate'] == True)
        ].copy()

        # Pivot: drivers as rows, lap numbers as columns
        pivot = heat.pivot_table(
            index='Driver', columns='LapNumber',
            values='LapTimeSeconds', aggfunc='mean'
        )
        pivot = pivot.reindex(drivers_sorted)

        # Per-driver normalization — show relative pace not absolute
        pivot_norm = pivot.apply(
            lambda row: (row - row.mean()) / max(row.std(), 0.001), axis=1

        )

        fig_heat = go.Figure(go.Heatmap(
            z=pivot_norm.values.tolist(),
            x=pivot_norm.columns.tolist(),
            y=pivot_norm.index.tolist(),
            colorscale=[
                [0.0, '#00ff88'],   # fast = green
                [0.5, '#ffff00'],   # medium = yellow
                [1.0, '#ff3333'],   # slow = red
            ],
            zmin=-2, zmax=2,
            hoverongaps=False,
            hovertemplate='Driver: %{y}<br>Lap: %{x}<br>Z-Score: %{z:.2f}<extra></extra>',
            colorbar=dict(
                title=dict(text='Pace (σ)', side='right'),
                tickvals=[-2, 0, 2],
                ticktext=['Fast', 'Average', 'Slow']
            )   
        ))
        fig_heat.update_layout(
            title=f'Pace Heatmap — {session.event["EventName"]} {year}',
            template='plotly_dark',
            paper_bgcolor='#16213e',
            plot_bgcolor='#16213e',
            height=500,
            margin=dict(l=80, r=20, t=50, b=40),
            xaxis_title='Lap Number',
            yaxis_title='Driver'
        )
        heat_json = fig_heat.to_json()
    except Exception as e:
        print(f"Heatmap error: {e}")
        heat_json = None

    return templates.TemplateResponse(request=request, name="index.html", context={
        "event_name": session.event['EventName'],
        "year": session.event.year,
        "heat_json": heat_json,
        "location": session.event['Location'],
        "date": str(session.date.date()),
        "total_laps": len(laps),
        "total_drivers": laps['Driver'].nunique(),
        "chart_json": fig.to_json(),
        "drivers": sorted(laps['Driver'].unique().tolist()),
        "compounds": sorted(laps['Compound'].dropna().unique().tolist()),
        "seasons": seasons,
        "schedule": schedule,
        "selected_year": year,
        "selected_round": round,
        "chart2_json": fig2.to_json(),
        "selected_session": session_type
    })

@app.get("/telemetry", response_class=HTMLResponse)
async def telemetry(request: Request, driver1: str = None, driver2: str = "None",
                    year: str = None, round: str = None, session_type: str = None):
    try:
        year  = int(year)  if year  and year  != 'None' else None
        round = int(round) if round and round != 'None' else None
    except:
        year, round = None, None
    if session_type == 'None':
        session_type = None
    if year is None or round is None:
        latest = get_latest_race()
        year = latest['year']
        round = latest['round']
        session_type = session_type or latest['session_type']
    
    # Get session first to know available drivers
    session      = load_race_session(year, round, session_type)
    laps         = session.laps
    drivers_list = sorted(laps['Driver'].unique().tolist())
    
    # Auto-select first driver if none selected or not in this race
    if driver1 is None or driver1 not in drivers_list:
        driver1 = drivers_list[0]
    session      = load_race_session(year, round, session_type)
    laps         = session.laps
    drivers_list = sorted(laps['Driver'].unique().tolist())
    seasons      = get_available_seasons()
    schedule     = get_season_schedule(year)

    if driver1 not in drivers_list:
        driver1 = drivers_list[0]

    chart_json = "{}"
    stats = {}
    try:
        lap1 = laps.pick_drivers(driver1).pick_fastest()
        tel1 = lap1.get_car_data().add_distance()
        # Ultimate lap = best individual sector times across all laps
        drv_laps = laps.pick_drivers(driver1).copy()
        sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
        ultimate_time = None
        left_on_table = None
        if all(c in drv_laps.columns for c in sector_cols):
            valid = drv_laps.dropna(subset=sector_cols)
            if len(valid) > 0:
                best_s1 = valid['Sector1Time'].min().total_seconds()
                best_s2 = valid['Sector2Time'].min().total_seconds()
                best_s3 = valid['Sector3Time'].min().total_seconds()
                ultimate_time = best_s1 + best_s2 + best_s3
                actual_best = lap1['LapTime'].total_seconds()
                left_on_table = builtins_round(actual_best - ultimate_time, 3)

        stats = {
            "driver1": driver1,
            "lap_time": str(lap1['LapTime']).split('.')[0][-5:],
            "lap_number": int(lap1['LapNumber']),
            "compound": lap1['Compound'],
            "max_speed": f"{tel1['Speed'].max():.0f}",
            "ultimate_lap": builtins_round(ultimate_time, 3) if ultimate_time else None,
            "left_on_table": left_on_table,
        }
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Speed (km/h)", "Throttle (%)", "Gear"),
                            vertical_spacing=0.08)
        fig.add_trace(go.Scatter(x=tel1['Distance'].tolist(), y=tel1['Speed'].tolist(),
                                 name=driver1, line=dict(color='#00d2ff', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=tel1['Distance'].tolist(), y=tel1['Throttle'].tolist(),
                                 line=dict(color='#00ff88', width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=tel1['Distance'].tolist(), y=tel1['nGear'].tolist(),
                                 line=dict(color='#ff6b6b', width=2), showlegend=False), row=3, col=1)

        if driver2 and driver2 != "None" and driver2 in drivers_list:
            lap2 = laps.pick_drivers(driver2).pick_fastest()
            tel2 = lap2.get_car_data().add_distance()
            delta = (lap1['LapTime'] - lap2['LapTime']).total_seconds()
            stats.update({
                "driver2": driver2,
                "delta": f"{delta:+.3f}",
                "max_speed2": f"{tel2['Speed'].max():.0f}"
            })
            fig.add_trace(go.Scatter(x=tel2['Distance'].tolist(), y=tel2['Speed'].tolist(),
                                     name=driver2, line=dict(color='#FF1E1E', width=2.5, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=tel2['Distance'].tolist(), y=tel2['Throttle'].tolist(),
                                     line=dict(color='#ffaa00', width=2, dash='dash'), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=tel2['Distance'].tolist(), y=tel2['nGear'].tolist(),
                                     line=dict(color='#aa00ff', width=2, dash='dash'), showlegend=False), row=3, col=1)

        fig.update_layout(template='plotly_dark', paper_bgcolor='#16213e', plot_bgcolor='#16213e',
                          height=700, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20))
        fig.update_xaxes(title_text="Distance (m)", row=3, col=1)
        chart_json = fig.to_json()
    except Exception as e:
        import traceback
        print(f"Telemetry error: {e}")
        traceback.print_exc()

    return templates.TemplateResponse(request=request, name="telemetry.html", context={
        "drivers": drivers_list,
        "driver1": driver1, "driver2": driver2,
        "stats": stats, "chart_json": chart_json,
        "seasons": seasons, "schedule": schedule,
        "selected_year": year, "selected_round": round,
        "selected_session": session_type
    })

@app.get("/incidents", response_class=HTMLResponse)
async def incidents(request: Request, year: str = None, round: str = None, session_type: str = None):
    try:
        year  = int(year)  if year  and year  != 'None' else None
        round = int(round) if round and round != 'None' else None
    except:
        year, round = None, None
    if session_type == 'None':
        session_type = None
    if year is None or round is None:
        latest = get_latest_race()
        year = latest['year']
        round = latest['round']
        session_type = session_type or latest['session_type']
    session  = load_race_session(year, round, session_type)
    laps     = session.laps
    seasons  = get_available_seasons()
    schedule = get_season_schedule(year)

    laps_copy = laps.copy()
    laps_copy['LapTimeSeconds'] = laps_copy['LapTime'].dt.total_seconds()
    clean = laps_copy[
        (laps_copy['IsAccurate'] == True) &
        (laps_copy['LapTimeSeconds'] > 60) &
        (laps_copy['LapTimeSeconds'] < 150)
    ].copy()

    clean_iso = clean.dropna(subset=['LapTimeSeconds', 'TyreLife', 'LapNumber']).copy()

    # Per-driver z-score (Option C hybrid — global model + local normalization)
    clean_iso['LapTimeZ'] = clean_iso.groupby('Driver')['LapTimeSeconds'].transform(
        lambda x: (x - x.mean()) / max(x.std(), 0.001)
    )

    # RegEra feature
    clean_iso['RegEra'] = 2 if year >= 2026 else (1 if year >= 2022 else 0)

    clean_iso = clean_iso.dropna(subset=['LapTimeZ', 'TyreLife', 'LapNumber', 'RegEra']).copy()
    feats     = clean_iso[['LapTimeZ', 'TyreLife', 'LapNumber', 'RegEra']]

    if session_type in ['R', 'S', 'SS'] and year < 2026:
        X_scaled = scaler.transform(feats)
        preds    = iso_forest.predict(X_scaled)
    elif session_type in ['R', 'S', 'SS'] and year >= 2026:
        from sklearn.preprocessing import StandardScaler as SS
        from sklearn.ensemble import IsolationForest as IF
        lsc = SS()
        lif = IF(contamination=0.08, n_estimators=100, random_state=42)
        preds = lif.fit_predict(lsc.fit_transform(feats))
    else:
        # Practice/Qualifying — always fit fresh per session
        from sklearn.preprocessing import StandardScaler as SS
        from sklearn.ensemble import IsolationForest as IF
        lsc   = SS()
        lif   = IF(contamination=0.08, n_estimators=100, random_state=42)
        preds = lif.fit_predict(lsc.fit_transform(feats))

    clean_iso['IsAnomaly'] = (preds == -1)

    driver_order   = clean_iso.groupby('Driver')['LapTimeSeconds'].mean().sort_values().index.tolist()
    compound_colors = {'SOFT': '#FF3333', 'MEDIUM': '#FFFF00', 'HARD': '#CCCCCC',
                       'INTERMEDIATE': '#39B54A', 'WET': '#0067FF'}

    fig = go.Figure()
    for drv in driver_order:
        d      = clean_iso[clean_iso['Driver'] == drv]
        normal = d[~d['IsAnomaly']]
        fig.add_trace(go.Scatter(
            x=normal['LapNumber'].tolist(), y=[drv]*len(normal), mode='markers',
            marker=dict(size=6, color=[compound_colors.get(c,'#888') for c in normal['Compound']], opacity=0.5),
            showlegend=False,
            hovertemplate=f'{drv}<br>Lap %{{x}}<br>%{{customdata:.3f}}s<extra></extra>',
            customdata=normal['LapTimeSeconds'].tolist()
        ))
        anom = d[d['IsAnomaly']]
        if len(anom) > 0:
            fig.add_trace(go.Scatter(
                x=anom['LapNumber'].tolist(), y=[drv]*len(anom), mode='markers',
                marker=dict(size=14, color='red', symbol='circle-open', line=dict(width=3, color='red')),
                showlegend=False,
                hovertemplate=f'<b>🚨 INCIDENT</b><br>{drv}<br>Lap %{{x}}<br>%{{customdata:.3f}}s<extra></extra>',
                customdata=anom['LapTimeSeconds'].tolist()
            ))

    # ── SC/VSC overlays ───────────────────────────────────────
    try:
        track_status = session.track_status
        if track_status is not None and len(track_status) > 0:
            # Get lap numbers for SC/VSC periods
            sc_laps  = []
            vsc_laps = []
            for _, row in track_status.iterrows():
                status = str(row.get('Status', ''))
                # Match SC/VSC lap numbers from laps data
                lap_time = row.get('Time', None)
                if lap_time is not None:
                    # Find which lap this corresponds to
                    matching = laps_copy[
                        laps_copy['LapStartTime'] <= lap_time
                    ]
                    if len(matching) > 0:
                        lap_num = int(matching['LapNumber'].max())
                        if status == '4':    # SC
                            sc_laps.append(lap_num)
                        elif status == '6':  # VSC
                            vsc_laps.append(lap_num)

            # Add SC regions
            sc_ranges = []
            if sc_laps:
                sc_laps = sorted(set(sc_laps))
                start   = sc_laps[0]
                for i in range(1, len(sc_laps)):
                    if sc_laps[i] > sc_laps[i-1] + 2:
                        sc_ranges.append((start, sc_laps[i-1]))
                        start = sc_laps[i]
                sc_ranges.append((start, sc_laps[-1]))

            for s, e in sc_ranges:
                fig.add_vrect(
                    x0=s, x1=e,
                    fillcolor='yellow', opacity=0.15,
                    line_width=0,
                    annotation_text='🟡 SC',
                    annotation_position='top left',
                    annotation_font_color='yellow'
                )

            # Add VSC regions
            vsc_ranges = []
            if vsc_laps:
                vsc_laps = sorted(set(vsc_laps))
                start    = vsc_laps[0]
                for i in range(1, len(vsc_laps)):
                    if vsc_laps[i] > vsc_laps[i-1] + 2:
                        vsc_ranges.append((start, vsc_laps[i-1]))
                        start = vsc_laps[i]
                vsc_ranges.append((start, vsc_laps[-1]))

            for s, e in vsc_ranges:
                fig.add_vrect(
                    x0=s, x1=e,
                    fillcolor='orange', opacity=0.15,
                    line_width=0,
                    annotation_text='🟠 VSC',
                    annotation_position='top left',
                    annotation_font_color='orange'
                )
    except Exception as e:
        print(f"SC/VSC overlay error: {e}")

    fig.update_layout(
        title=f'Race-Wide Incident Map — {session.event["EventName"]} {year}',
        template='plotly_dark', paper_bgcolor='#16213e', plot_bgcolor='#16213e',
        height=750, margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(categoryorder='array', categoryarray=driver_order[::-1])
    )

    incidents_df = clean_iso[clean_iso['IsAnomaly']][[
        'Driver', 'LapNumber', 'LapTimeSeconds', 'Compound', 'TyreLife'
    ]].sort_values('LapNumber').reset_index(drop=True)
    incidents_df['LapNumber']      = incidents_df['LapNumber'].astype(int)
    incidents_df['TyreLife']       = incidents_df['TyreLife'].astype(int)
    incidents_df['LapTimeSeconds'] = incidents_df['LapTimeSeconds'].round(3)

    return templates.TemplateResponse(request=request, name="incidents.html", context={
        "total_laps": len(clean_iso),
        "total_incidents": int(clean_iso['IsAnomaly'].sum()),
        "detection_rate": f"{clean_iso['IsAnomaly'].sum()/len(clean_iso)*100:.1f}",
        "drivers_flagged": int(clean_iso[clean_iso['IsAnomaly']]['Driver'].nunique()),
        "chart_json": fig.to_json(),
        "incidents": incidents_df.to_dict(orient='records'),
        "seasons": seasons, "schedule": schedule,
        "selected_year": year, "selected_round": round,
        "selected_session": session_type,
        "event_name": session.event['EventName']
    })

@app.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request, year: str = None,
                        round: str = None, session_type: str = None):
    try:
        year  = int(year)  if year  and year  != 'None' else None
        round = int(round) if round and round != 'None' else None
    except:
        year, round = None, None
    if session_type == 'None':
        session_type = None
    if year is None or round is None:
        latest       = get_latest_race()
        year         = latest['year']
        round        = latest['round']
        session_type = session_type or latest['session_type']

    # Strategy only valid for Race and Sprint sessions
    blocked = ['Q', 'SQ', 'SS', 'FP1', 'FP2', 'FP3',
               'Qualifying', 'Sprint Qualifying', 'Sprint Shootout',
               'Practice 1', 'Practice 2', 'Practice 3']
    if session_type in blocked:
        session_type = 'R'

    session  = load_race_session(year, round, session_type)
    laps     = session.laps
    seasons  = get_available_seasons()
    schedule = get_season_schedule(year)
    drivers  = sorted(laps['Driver'].unique().tolist())
    compounds = sorted(laps['Compound'].dropna().unique().tolist())

    # Detect max stints 
    try:
        max_stints = 1
        for driver in laps['Driver'].unique():
            drv = laps[laps['Driver'] == driver].sort_values('LapNumber')
            compound_changes = (drv['Compound'] != drv['Compound'].shift()).sum()
            driver_stints = max(1, compound_changes)
            max_stints = max(max_stints, driver_stints)
        max_stints = int(max_stints)
    except:
        max_stints = 2
    has_pit_strategy = max_stints > 1

    # Auto pit loss
    try:
        pit_laps = laps[
            laps['PitOutTime'].notna() & laps['PitInTime'].notna()
        ].copy()
        if len(pit_laps) > 0:
            pit_laps['PitLoss'] = (
                pit_laps['PitOutTime'] - pit_laps['PitInTime']
            ).dt.total_seconds()
            pit_loss_avg = round(float(pit_laps['PitLoss'].median()), 1)
            pit_loss_avg = max(15.0, min(35.0, pit_loss_avg))
        else:
            pit_loss_avg = 22.0
    except:
        pit_loss_avg = 22.0

    # Stint summary
    # ── Smart Strategy Suggestions ────────────────────────────
    try:
        stint_summary = []
        for driver in sorted(laps['Driver'].unique()):
            drv_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
            stints, current_compound, stint_start = [], None, 1
            for _, lap in drv_laps.iterrows():
                compound = lap.get('Compound')
                if pd.isna(compound) or compound == 'UNKNOWN':
                    continue
                if compound != current_compound:
                    if current_compound:
                        stints.append({
                            'compound': current_compound,
                            'laps': int(lap['LapNumber']) - stint_start
                        })
                    current_compound = compound
                    stint_start      = int(lap['LapNumber'])
            if current_compound:
                stints.append({
                    'compound': current_compound,
                    'laps': int(drv_laps['LapNumber'].max()) - stint_start + 1
                })
            if stints:
                stint_summary.append({
                    'driver':    driver,
                    'stints':    stints,
                    'num_stops': len(stints) - 1
                })

        # ── Get winner's strategy ─────────────────────────────
        winner_strategy = None
        try:
            results = session.results
            if results is not None and len(results) > 0:
                winner_abbr = results.iloc[0].get('Abbreviation', '')
                winner_data = next(
                    (s for s in stint_summary if s['driver'] == winner_abbr),
                    None
                )
                if winner_data:
                    winner_strategy = winner_data
        except:
            pass

        # ── Analyze stop distribution ─────────────────────────
        stop_counts   = [s['num_stops'] for s in stint_summary]
        avg_stops     = sum(stop_counts) / len(stop_counts) if stop_counts else 1
        one_stoppers  = [s for s in stint_summary if s['num_stops'] == 1]
        two_stoppers  = [s for s in stint_summary if s['num_stops'] == 2]
        zero_stoppers = [s for s in stint_summary if s['num_stops'] == 0]

        pct_two_stop = len(two_stoppers) / len(stint_summary) if stint_summary else 0

        # ── Build strategy suggestions ────────────────────────
        suggestions = []

        # 1. Winner strategy
        if winner_strategy and winner_strategy['stints']:
            s = winner_strategy['stints']
            suggestions.append({
                'label':       '🏆 Winner',
                'description': f"What {winner_strategy['driver']} actually did",
                'stops':       winner_strategy['num_stops'],
                'stint1':      s[0]['compound'] if len(s) > 0 else compounds[0],
                'stint2':      s[1]['compound'] if len(s) > 1 else compounds[-1],
                'stint3':      s[2]['compound'] if len(s) > 2 else None,
                'color':       '#ffd700'
            })

        # 2. Most popular (most common stop count)
        popular_group = one_stoppers if len(one_stoppers) >= len(two_stoppers) \
                        else two_stoppers
        if popular_group:
            # Most common compound combo
            from collections import Counter
            combos = []
            for s in popular_group:
                if len(s['stints']) >= 2:
                    combos.append((
                        s['stints'][0]['compound'],
                        s['stints'][1]['compound'],
                        s['stints'][2]['compound'] if len(s['stints']) > 2 else None
                    ))
            if combos:
                most_common = Counter(combos).most_common(1)[0][0]
                suggestions.append({
                    'label':       '👥 Popular',
                    'description': f"Most common strategy ({len(popular_group)} drivers)",
                    'stops':       len(popular_group[0]['stints']) - 1,
                    'stint1':      most_common[0],
                    'stint2':      most_common[1],
                    'stint3':      most_common[2],
                    'color':       '#00d2ff'
                })

        # 3. Aggressive (most stops)
        if two_stoppers and pct_two_stop > 0.2:
            s = two_stoppers[0]['stints']
            suggestions.append({
                'label':       '⚡ Aggressive',
                'description': f"2-stop ({int(pct_two_stop*100)}% of drivers)",
                'stops':       2,
                'stint1':      s[0]['compound'] if len(s) > 0 else 'SOFT',
                'stint2':      s[1]['compound'] if len(s) > 1 else 'MEDIUM',
                'stint3':      s[2]['compound'] if len(s) > 2 else None,
                'color':       '#ff6b6b'
            })

        # 4. Conservative (fewest stops)
        conservative_group = zero_stoppers if zero_stoppers else one_stoppers
        if conservative_group:
            s = conservative_group[0]['stints']
            suggestions.append({
                'label':       '🛡️ Conservative',
                'description': f"Fewest stops ({conservative_group[0]['num_stops']} stop)",
                'stops':       conservative_group[0]['num_stops'],
                'stint1':      s[0]['compound'] if len(s) > 0 else compounds[-1],
                'stint2':      s[1]['compound'] if len(s) > 1 else compounds[-1],
                'stint3':      None,
                'color':       '#4ade80'
            })

        # ── Default suggested values (from winner or popular) ─
        default = suggestions[0] if suggestions else None
        suggested_s1 = default['stint1'] if default else compounds[0]
        suggested_s2 = default['stint2'] if default else compounds[-1]
        suggested_s3 = default['stint3'] if default else None

        # Auto-enable 3rd stint if >30% did 2 stops
        auto_three_stint = pct_two_stop > 0.3

    except Exception as e:
        print(f"Strategy suggestion error: {e}")
        stint_summary    = []
        suggestions      = []
        suggested_s1     = compounds[0] if compounds else 'SOFT'
        suggested_s2     = compounds[-1] if compounds else 'HARD'
        suggested_s3     = None
        auto_three_stint = False
        pct_two_stop     = 0

    return templates.TemplateResponse(request=request, name="strategy.html", context={
        "drivers":          drivers,
        "compounds":        compounds,
        "race_laps": session.laps['LapNumber'].max() if len(session.laps) > 0 else 57,
        "seasons":          seasons,
        "schedule":         schedule,
        "selected_year":    year,
        "selected_round":   round,
        "selected_session": session_type,
        "event_name":       session.event['EventName'],
        "race_length":      int(laps['LapNumber'].max()),
        "pit_loss_avg":     pit_loss_avg,
        "stint_summary":    stint_summary,
        "suggestions":      suggestions,
        "suggested_s1":     suggested_s1,
        "suggested_s2":     suggested_s2,
        "suggested_s3":     suggested_s3,
        "max_stints":       max_stints,
        "has_pit_strategy": has_pit_strategy,
        "auto_three_stint": auto_three_stint,
        "pct_two_stop":     float(int(pct_two_stop * 1000) / 10)
    })

# ════════════════════════════════════════════════════════
# STRATEGY OPTIMIZER
# ════════════════════════════════════════════════════════

class StrategyRequest(BaseModel):
    driver: str = "HAM"
    stint1: str = "SOFT"
    stint2: str = "HARD"
    stint3: str = "None" 
    race_length: int = 57
    pit_loss: float = 22.0
    pit_min: int = 10
    pit_max: int = 45
    year: int = 2024
    round: int = 1
    session_type: str = "R"

@app.post("/strategy/optimize", tags=["Analysis"])
async def optimize_strategy(req: StrategyRequest):
    is_wet    = req.stint1 in ['INTERMEDIATE', 'WET'] or \
                req.stint2 in ['INTERMEDIATE', 'WET']
    is_sprint = req.session_type in ['S', 'SS', 'Sprint']

    # Validate compounds
    if not is_wet and not is_sprint:
        all_compounds = {req.stint1, req.stint2}
        if req.stint3 and req.stint3 != 'None':
            all_compounds.add(req.stint3)
        if len(all_compounds) < 2:
            return {"error": "F1 rules require at least 2 different compounds in a dry race"}

    race_data = get_race_model(req.year, req.round, req.session_type)
    model     = race_data['model']
    d_enc     = race_data['driver_enc']
    c_enc     = race_data['compound_enc']

    if req.driver not in race_data['drivers']:
        return {"error": f"Driver {req.driver} not in this race"}
    if req.stint1 not in race_data['compounds']:
        return {"error": f"Compound {req.stint1} not available"}
    if req.stint2 not in race_data['compounds']:
        return {"error": f"Compound {req.stint2} not available"}
    if req.stint3 != 'None' and req.stint3 not in race_data['compounds']:
        return {"error": f"Compound {req.stint3} not available"}

    def predict_stint(driver, compound, stint_len, start_lap=1):
        d_e  = d_enc.transform([driver])[0]
        c_e  = c_enc.transform([compound])[0]
        data = [{
            'TyreLife': i, 'TyreLifeSquared': i**2,
            'CompoundEncoded': c_e, 'DriverEncoded': d_e,
            'CompoundAge': c_e*i, 'LapNumber': start_lap+i-1
        } for i in range(1, stint_len+1)]
        return model.predict(pd.DataFrame(data))

    strategies = []
    is_two_stop = req.stint3 and req.stint3 != 'None'

    if is_two_stop:
        # 2-stop: optimize both pit laps
        for pit1 in range(req.pit_min, req.pit_max + 1):
            for pit2 in range(pit1 + 5, req.race_length - 5):
                if pit2 >= req.pit_max + 10:
                    continue
                s1    = predict_stint(req.driver, req.stint1, pit1, 1)
                s2    = predict_stint(req.driver, req.stint2, pit2 - pit1, pit1 + 1)
                s3    = predict_stint(req.driver, req.stint3, req.race_length - pit2, pit2 + 1)
                total = float(sum(s1)) + req.pit_loss + \
                        float(sum(s2)) + req.pit_loss + float(sum(s3))
                strategies.append({
                    "pit_lap":      pit1,
                    "pit_lap2":     pit2,
                    "stint1_time":  round(float(sum(s1)), 3),
                    "stint2_time":  round(float(sum(s2)), 3),
                    "stint3_time":  round(float(sum(s3)), 3),
                    "total":        round(total, 3)
                })
    else:
        # 1-stop
        for pit_lap in range(req.pit_min, req.pit_max + 1):
            s1    = predict_stint(req.driver, req.stint1, pit_lap, 1)
            s2    = predict_stint(req.driver, req.stint2, req.race_length - pit_lap, pit_lap + 1)
            total = float(sum(s1)) + req.pit_loss + float(sum(s2))
            strategies.append({
                "pit_lap":     pit_lap,
                "pit_lap2":    None,
                "stint1_time": round(float(sum(s1)), 3),
                "stint2_time": round(float(sum(s2)), 3),
                "stint3_time": None,
                "total":       round(total, 3)
            })

    strategies.sort(key=lambda x: x['total'])
    optimal    = strategies[0]
    top5       = strategies[:5]
    pit_window = [s['pit_lap'] for s in top5]

    # Chart
    title = f'Pit Window — {req.driver} | {req.stint1} → {req.stint2}'
    if is_two_stop:
        title += f' → {req.stint3}'

    fig = go.Figure()
    if is_two_stop:
        # For 2-stop, plot pit1 vs total
        pit1_vals   = sorted(set(s['pit_lap'] for s in strategies))
        best_by_pit1 = {}
        for s in strategies:
            p = s['pit_lap']
            if p not in best_by_pit1 or s['total'] < best_by_pit1[p]['total']:
                best_by_pit1[p] = s
        x_vals = [best_by_pit1[p]['pit_lap'] for p in pit1_vals]
        y_vals = [best_by_pit1[p]['total']   for p in pit1_vals]
    else:
        x_vals = [s['pit_lap'] for s in strategies]
        y_vals = [s['total']   for s in strategies]

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='lines+markers',
        name='Total Race Time',
        line=dict(color='#00d2ff', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=[optimal['pit_lap']], y=[optimal['total']],
        mode='markers',
        name='🏆 Optimal',
        marker=dict(color='#00ff88', size=22, symbol='star')
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
        xaxis_title='Pit Lap 1' if is_two_stop else 'Pit Lap',
        yaxis_title='Total Race Time (s)',
        title=title
    )

    return {
        "optimal":           optimal,
        "top5":              top5,
        "pit_window_start":  min(pit_window),
        "pit_window_end":    max(pit_window),
        "chart_json":        fig.to_json(),
        "is_two_stop":       is_two_stop
    }
@app.get("/api/model-comparison/{year}/{round_number}/{session_type}", tags=["System"])
async def model_comparison(year: int, round_number: int, session_type: str = 'R'):
    """Compare XGBoost vs GRU performance on the same race."""
    try:
        # XGBoost metrics
        xgb_data = get_race_model(year, round_number, session_type)

        return {
            "status": "success",
            "xgboost": {
                "mae": round(float(
                    mean_absolute_error_for_race(xgb_data, year, round_number, session_type)
                ), 3) if xgb_data else None,
                "model": "XGBoost Regressor"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
# ── LLM Analyst Page ──────────────────────────────────────
@app.get("/analyst", response_class=HTMLResponse)
async def analyst_page(
    request: Request,
    year: str = None,
    round: str = None,
    session_type: str = None
):
    try:
        year  = int(year)  if year  and year  != 'None' else None
        round = int(round) if round and round != 'None' else None
    except:
        year, round = None, None
    if session_type == 'None':
        session_type = None
    if year is None or round is None:
        latest       = get_latest_race()
        year         = latest['year']
        round        = latest['round']
        session_type = session_type or latest['session_type']

    seasons    = get_available_seasons()
    schedule   = get_season_schedule(year)
    event_name = f'Round {round}'
    try:
        session    = load_race_session(year, round, session_type)
        event_name = session.event['EventName']
    except:
        pass

    return templates.TemplateResponse(request=request, name="analyst.html", context={
        "event_name":       event_name,
        "year":             year,
        "seasons":          seasons,
        "schedule":         schedule,
        "selected_year":    year,
        "selected_round":   round,
        "selected_session": session_type,
        "active":           "analyst",
    })


class AnalystRequest(BaseModel):
    question:     str
    year:         int  = 2024
    round:        int  = 1
    session_type: str  = "R"
    mode:         str  = "quick"  # "quick" or "deep"


@app.post("/analyst/ask", tags=["Intelligence"])
async def ask_analyst(req: AnalystRequest):
    """
    Race analyst — two modes:
    - quick: single Claude call with full race context
    - deep: 3 specialist agents in parallel + coordinator synthesis
    """
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {"status": "error", "message": "ANTHROPIC_API_KEY not configured."}

        # Load race session + build context
        try:
            session = load_race_session(req.year, req.round, req.session_type)
            context = build_race_context(session)
            event_name = session.event['EventName']

            # Compact summary for multi-agent (avoid token bloat)
            laps = session.laps
            top_drivers = laps.groupby('Driver')['LapTime'].apply(
                lambda x: x.dropna().dt.total_seconds().mean()
            ).sort_values().head(5)
            driver_summary = ", ".join([f"{d} (avg {t:.2f}s)" for d, t in top_drivers.items()])
            compounds      = laps['Compound'].dropna().unique().tolist()
            total_laps     = int(laps['LapNumber'].max())

            compact_context = f"""
Race: {event_name} {req.year}
Session: {req.session_type}
Total Laps: {total_laps}
Compounds Used: {', '.join(compounds)}
Fastest 5 Drivers (avg pace): {driver_summary}
"""
        except Exception as e:
            context         = f"Race: {req.year} Round {req.round} {req.session_type}"
            compact_context = context
            event_name      = f"Round {req.round}"

        client = anthropic.Anthropic(api_key=api_key)

        # ── QUICK MODE ─────────────────────────────────────
        if req.mode == "quick":
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=(
                    "You are an expert Formula 1 race analyst with deep knowledge "
                    "of F1 strategy, tyre management, and driver performance. "
                    "Use the actual race data provided. Be specific — mention driver names, "
                    "lap times, and strategies. Keep answers concise but insightful. "
                    "Format in clear paragraphs."
                ),
                messages=[{
                    "role":    "user",
                    "content": f"""Race data:
{context}

Question: {req.question}"""
                }]
            )
            return {
                "status": "success",
                "mode":   "quick",
                "answer": message.content[0].text,
                "race":   f"{event_name} {req.year}",
            }

        # ── DEEP MODE (multi-agent) ─────────────────────────
        agents = [
            {
                "name":   "Pace Analyst",
                "emoji":  "⚡",
                "prompt": f"""You are a Formula 1 pace analyst. Analyze ONLY raw speed and lap time aspects.
Race context: {compact_context}
Question: {req.question}
Give a focused 2-3 sentence technical analysis of pace-related factors."""
            },
            {
                "name":   "Strategy Analyst",
                "emoji":  "🔧",
                "prompt": f"""You are a Formula 1 strategy expert. Analyze ONLY tyre strategy, pit stop timing, and compound choices.
Race context: {compact_context}
Question: {req.question}
Give a focused 2-3 sentence analysis of strategy decisions. Be specific about undercut/overcut opportunities."""
            },
            {
                "name":   "Race Craft Analyst",
                "emoji":  "🏎️",
                "prompt": f"""You are a Formula 1 race craft expert. Analyze ONLY overtaking, defending, incidents, and driver decisions.
Race context: {compact_context}
Question: {req.question}
Give a focused 2-3 sentence analysis of on-track battles and driver decisions."""
            },
        ]

        # Run all 3 agents in parallel using threads (anthropic SDK is sync)
        import concurrent.futures

        def run_agent(agent):
            try:
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=300,
                    messages=[{"role": "user", "content": agent["prompt"]}]
                )
                return {
                    "name":     agent["name"],
                    "emoji":    agent["emoji"],
                    "analysis": msg.content[0].text,
                }
            except Exception as e:
                return {
                    "name":     agent["name"],
                    "emoji":    agent["emoji"],
                    "analysis": f"Agent unavailable: {e}",
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            agent_results = list(executor.map(run_agent, agents))

        # Coordinator synthesis
        agent_summaries = "\\n\\n".join([
            f"{r['emoji']} {r['name']}:\\n{r['analysis']}"
            for r in agent_results
        ])

        coordinator_msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": f"""You are a senior F1 analyst coordinating specialist agents.

Agent analyses:
{agent_summaries}

Original question: {req.question}
Race: {compact_context}

Synthesize these insights into a comprehensive 3-4 sentence answer integrating all perspectives.
Lead with the most important insight. Be direct and specific."""
            }]
        )

        return {
            "status":    "success",
            "mode":      "deep",
            "agents":    agent_results,
            "synthesis": coordinator_msg.content[0].text,
            "race":      f"{event_name} {req.year}",
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/predictions", response_class=HTMLResponse)
async def predictions_page(request: Request, year: str = None, round: str = None, session_type: str = None):
    try:
        year_int  = int(year) if year and year != 'None' else None
        round_int = int(round) if round and round != 'None' else None
    except:
        year_int  = None
        round_int = None

    if year_int is None or round_int is None:
        latest    = get_latest_race()
        year_int  = latest['year']
        round_int = latest['round']

    seasons  = get_available_seasons()
    schedule = get_season_schedule(year_int)

    try:
        session    = load_race_session(year_int, round_int, 'R')
        event_name = session.event['EventName']
        event_date = str(session.event['EventDate'])[:10]
    except:
        try:
            import fastf1 as ff1
            ff1.Cache.enable_cache(str(CACHE_DIR))
            ev         = ff1.get_event(year_int, round_int)
            event_name = ev['EventName']
            event_date = str(ev['EventDate'])[:10]
        except:
            event_name = f'Round {round_int}'
            event_date = '—'

    return templates.TemplateResponse(request=request, name="predictions.html", context={
        "seasons":          seasons,
        "schedule":         schedule,
        "selected_year":    year_int,
        "selected_round":   round_int,
        "selected_session": "R",
        "active":           "predictions",
        "event_name":       event_name,
        "event_date":       event_date,
    })

@app.get(
    "/api/predictions/{year}/{round_number}",
    tags=["Intelligence"],
    summary="Auto-detect prediction mode and return race predictions",
    description="Detects available session data (FP/Q/R) and returns the appropriate prediction: practice pace ranking, quali→race prediction (XGBoost+Ridge trained on historical deltas), or post-race pure pace analysis."
)
async def get_predictions(year: int, round_number: int):
    try:
        import fastf1 as ff1
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        ff1.Cache.enable_cache(str(CACHE_DIR))

        # ── Step 1: Fast session detection — metadata only ─
        # Check which sessions exist WITHOUT loading laps
        def check_session_exists(st):
            try:
                s = ff1.get_session(year, round_number, st)
                s.load(laps=True, telemetry=False,
                       weather=False, messages=False)
                if len(s.laps) == 0:
                    return st, None
                return st, s
            except:
                return st, None

        # Run all checks in parallel
        loop     = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=7)
        futures  = [
            loop.run_in_executor(executor, check_session_exists, st)
            for st in ['FP1', 'FP2', 'FP3', 'SQ', 'Q', 'S', 'R']
        ]
        results = await asyncio.gather(*futures)

        # Sessions that exist (metadata loaded, no laps yet)
        exists = {st: s for st, s in results if s is not None}

        if not exists:
            return {
                "status":  "no_data",
                "message": "No session data available yet for this race weekend.",
                "mode":    "no_data"
            }

        is_sprint_weekend = 'SQ' in exists or 'S' in exists

        # ── Step 2: Detect mode from what exists ──────────
        # Check race completion — just check lap count from metadata
        race_complete = False
        if 'R' in exists:
            try:
                r_session = exists['R']
                r_session.load(laps=True, telemetry=False,
                               weather=False, messages=False)
                r_laps = r_session.laps
                if len(r_laps) > 0:
                    max_lap  = int(r_laps['LapNumber'].max())
                    mode_lap = int(r_laps['LapNumber'].mode()[0])
                    race_complete = max_lap >= mode_lap * 0.9
                exists['R'] = r_session  # update with laps loaded
            except:
                pass

        if race_complete:
            mode = 'post_race'
        elif 'Q' in exists:
            mode = 'pre_race_from_quali'
        elif 'S' in exists and is_sprint_weekend:
            mode = 'pre_race_from_sprint'
        elif 'SQ' in exists and is_sprint_weekend:
            mode = 'pre_sprint_from_sq'
        elif any(fp in exists for fp in ['FP1','FP2','FP3']):
            mode = 'pre_quali_from_fp'
        else:
            mode = 'no_data'

        available = exists
        from src.tabnet_model import PreRacePredictor

        # ── Step 3: Run predictions based on mode ─────────

        # POST RACE
        if mode == 'post_race':
            postrace_data     = get_postrace_model(year, round_number, 'R')
            predictor         = postrace_data['predictor']
            session           = available['R']
            results, importance = predictor.predict(session)
            return {
                "status":             "success",
                "mode":               "post_race",
                "mode_label":         "Post-Race Pure Pace Analysis",
                "mode_description":   "Race has happened. Showing predicted finishing order based on pure pace, removing SC/DNF effects.",
                "is_sprint_weekend":  is_sprint_weekend,
                "available_sessions": list(available.keys()),
                "predictions":        results,
                "feature_importance": importance,
                "event":              f"Round {round_number}",
                "year":               year,
                "mae":                predictor.mae,
            }

        # PRE RACE FROM QUALI
        if mode == 'pre_race_from_quali':
            predictor = PreRacePredictor()
            ridge_results, xgb_results, comparison = \
                predictor.predict_from_quali(year, round_number)
            print(f"DEBUG PREDICTION: used_historical={comparison.get('used_historical')} samples={comparison.get('training_samples')} mae={comparison.get('xgb',{}).get('mae')}")
            print(f"DEBUG FIRST 3: {[(r['driver'], r['position'], r['grid_position'], r['position_delta']) for r in xgb_results[:3]]}")

            # Also get FP correlation if available
            fp_corr = {}
            try:
                fp_corr = predictor.get_fp_race_correlation(year, round_number)
            except:
                pass

            return {
                "status":             "success",
                "mode":               "pre_race_from_quali",
                "mode_label":         "Race Prediction from Qualifying",
                "mode_description":   f"Qualifying complete. Predicting race finishing order using quali times + {'historical quali→race model (' + str(comparison.get('training_samples',0)) + ' races)' if comparison.get('used_historical') else 'quali order (insufficient history)'}.",
                "is_sprint_weekend":  is_sprint_weekend,
                "available_sessions": list(available.keys()),
                "predictions":        xgb_results,
                "ridge_predictions":  ridge_results,
                "comparison":         comparison,
                "feature_importance": comparison.get('xgb', {}).get('importance', []),
                "used_historical":    comparison.get('used_historical', False),
                "training_samples":   comparison.get('training_samples', 0),
                "mae":                comparison.get('xgb', {}).get('mae', 0),
                "fp_correlation":     fp_corr,
                "event":              f"Round {round_number}",
                "year":               year,
            }

        # PRE RACE FROM SPRINT (sprint happened, predict race)
        if mode == 'pre_race_from_sprint':
            predictor = PreRacePredictor()
            # Use quali if available, else sprint data
            if 'Q' in available:
                ridge_results, xgb_results, comparison = \
                    predictor.predict_from_quali(year, round_number)
                source = "Qualifying + Sprint + Sprint Qualifying"
            else:
                sq_df = predictor.get_sprint_quali_data(year, round_number)
                ridge_results, xgb_results, comparison = \
                    predictor.predict_from_quali_df(sq_df) if len(sq_df) > 0 \
                    else ([], [], {})
                source = "Sprint Qualifying data"

            return {
                "status":             "success",
                "mode":               "pre_race_from_sprint",
                "mode_label":         "Race Prediction (Sprint Weekend)",
                "mode_description":   f"Sprint complete. Predicting race from {source}.",
                "is_sprint_weekend":  True,
                "available_sessions": list(available.keys()),
                "predictions":        xgb_results,
                "ridge_predictions":  ridge_results,
                "comparison":         comparison,
                "feature_importance": comparison.get('xgb', {}).get('importance', []),
                "mae":                comparison.get('xgb', {}).get('mae', 0),
                "event":              f"Round {round_number}",
                "year":               year,
            }

        # PRE SPRINT FROM SQ
        if mode == 'pre_sprint_from_sq':
            predictor = PreRacePredictor()
            sq_df     = predictor.get_sprint_quali_data(year, round_number)
            if len(sq_df) == 0:
                return {"status": "error", "message": "Sprint qualifying data not available"}
            ridge_results, xgb_results, comparison = \
                predictor.predict_from_quali_df(sq_df)
            return {
                "status":             "success",
                "mode":               "pre_sprint_from_sq",
                "mode_label":         "Sprint Race Prediction",
                "mode_description":   "Sprint Qualifying complete. Predicting sprint race finishing order.",
                "is_sprint_weekend":  True,
                "available_sessions": list(available.keys()),
                "predictions":        xgb_results,
                "ridge_predictions":  ridge_results,
                "comparison":         comparison,
                "feature_importance": comparison.get('xgb', {}).get('importance', []),
                "mae":                comparison.get('xgb', {}).get('mae', 0),
                "event":              f"Round {round_number}",
                "year":               year,
            }

        # PRE QUALI FROM FP
        if mode == 'pre_quali_from_fp':
            predictor = PreRacePredictor()
            results, _, comparison = predictor.predict_from_session(year, round_number, None)
            return {
                "status":             "success",
                "mode":               "pre_quali_from_fp",
                "mode_label":         "Practice Pace Progression",
                "mode_description":   "Qualifying hasn't happened yet. Showing how each driver's pace rank evolved across practice sessions — useful for spotting improving setups vs sandbagging, not a prediction of qualifying order.",
                "is_sprint_weekend":  is_sprint_weekend,
                "available_sessions": list(available.keys()),
                "predictions":        results,
                "sessions_available": comparison.get('sessions_available', []),
                "is_progression":     True,
                "event":              f"Round {round_number}",
                "year":               year,
            }

        return {"status": "no_data", "message": "No prediction available", "mode": "no_data"}

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}


@app.get("/api/race-status/{year}/{round_number}", tags=["Analysis"])
async def race_status(year: int, round_number: int, session_type: str = 'R'):
    """Check if a race has happened, is live, or is upcoming."""
    return get_race_status(year, round_number, session_type)
@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate(request: Request):
    import json
    from pathlib import Path

    MODELS_DIR = Path("models")
    DATA_DIR   = Path("data")

    # Load metadata
    try:
        with open(MODELS_DIR / "model_metadata.json") as f:
            metadata = json.load(f)
    except:
        metadata = {}

    # Per-circuit MAE from parquet
    circuit_stats = []
    try:
        import pandas as pd
        import joblib
        from sklearn.metrics import mean_absolute_error

        df         = pd.read_parquet(DATA_DIR / "all_races_2018_2026.parquet")
        model      = joblib.load(MODELS_DIR / "tire_degradation_model.pkl")
        c_enc      = joblib.load(MODELS_DIR / "compound_encoder.pkl")
        d_enc      = joblib.load(MODELS_DIR / "driver_encoder.pkl")
        circ_enc   = joblib.load(MODELS_DIR / "circuit_encoder.pkl")

        # Normalize compounds
        compound_map = {'ULTRASOFT':'SOFT','SUPERSOFT':'SOFT','HYPERSOFT':'SOFT'}
        df['Compound'] = df['Compound'].replace(compound_map)
        df = df[df['Compound'].isin(c_enc.classes_)].copy()
        df = df[df['Driver'].isin(d_enc.classes_)].copy()
        df = df[df['Circuit'].isin(circ_enc.classes_)].copy()

        df['CompoundEncoded'] = c_enc.transform(df['Compound'])
        df['DriverEncoded']   = d_enc.transform(df['Driver'])
        df['CircuitEncoded']  = circ_enc.transform(df['Circuit'])
        df['TyreLifeSquared'] = df['TyreLife'] ** 2
        df['CompoundAge']     = df['CompoundEncoded'] * df['TyreLife']

        FEATURES = ['TyreLife','TyreLifeSquared','CompoundEncoded',
                    'DriverEncoded','CompoundAge','LapNumber','RegEra','CircuitEncoded']

        df_clean = df.dropna(subset=FEATURES + ['LapTimeSeconds'])
        preds    = model.predict(df_clean[FEATURES])
        df_clean = df_clean.copy()
        df_clean['Pred'] = preds
        df_clean['AbsErr'] = abs(df_clean['LapTimeSeconds'] - df_clean['Pred'])

        # Per-circuit stats
        for circuit, grp in df_clean.groupby('Circuit'):
            circuit_stats.append({
                'circuit': circuit,
                'mae':     round(grp['AbsErr'].mean(), 3),
                'laps':    len(grp),
                'seasons': sorted(grp['Season'].unique().tolist())
            })
        circuit_stats.sort(key=lambda x: x['mae'])

        # Per-season stats
        season_stats = []
        for season, grp in df_clean.groupby('Season'):
            season_stats.append({
                'season': int(season),
                'mae':    round(grp['AbsErr'].mean(), 3),
                'laps':   len(grp),
                'era':    'New Regs' if season >= 2026 else ('Ground Effect' if season >= 2022 else 'Pre-2022')
            })
        season_stats.sort(key=lambda x: x['season'])
        # Per-session-type MAE breakdown
        session_stats = []
        if 'SessionType' in df_clean.columns:
            for stype, grp in df_clean.groupby('SessionType'):
                session_stats.append({
                    'session': stype,
                    'mae':     round(grp['AbsErr'].mean(), 3),
                    'laps':    len(grp)
                })
            session_stats.sort(key=lambda x: x['mae'])
            
    except Exception as e:
        print(f"Eval error: {e}")
        circuit_stats = []
        season_stats  = []
        session_stats = []

    return templates.TemplateResponse(request=request, name="evaluate.html", context={
        "metadata":      metadata,
        "circuit_stats": circuit_stats,
        "season_stats":  season_stats,
        "active":        "evaluate",
        "selected_year": None,
        "selected_round": None,
        "selected_session": None,
    })            
@app.post("/api/whatif", tags=["Intelligence"])
async def whatif(req: dict = Body(...)):
    import pandas as pd
    year         = req.get('year')
    round_number = req.get('round')
    session_type = req.get('session_type', 'R')
    driver       = req.get('driver')
    stint1       = req.get('stint1')
    stint2       = req.get('stint2')
    stint3       = req.get('stint3', 'None')
    pit_lap      = req.get('pit_lap', 25)
    pit_lap2     = req.get('pit_lap2', None)
    race_length  = req.get('race_length', 57)
    pit_loss     = req.get('pit_loss', 22.0)
    sc_lap       = req.get('sc_lap', None)
    sc_duration  = req.get('sc_duration', 5)
    rain_lap     = req.get('rain_lap', None)

    try:
        race_data = get_race_model(year, round_number, session_type)
        model     = race_data['model']
        d_enc     = race_data['driver_enc']
        c_enc     = race_data['compound_enc']

        def predict_stint(compound, stint_len, start_lap):
            d_e  = d_enc.transform([driver])[0]
            c_e  = c_enc.transform([compound])[0]
            # Switch to WET/INTER if rain
            if rain_lap and rain_lap >= start_lap:
                compound = 'INTERMEDIATE' if compound not in ['INTERMEDIATE','WET'] else compound
                if compound in c_enc.classes_:
                    c_e = c_enc.transform([compound])[0]
            rows = []
            for i in range(1, stint_len + 1):
                lap = start_lap + i - 1
                lap_time_mult = 1.3 if (sc_lap and sc_lap <= lap <= sc_lap + sc_duration) else 1.0
                rows.append({
                    'TyreLife': i, 'TyreLifeSquared': i**2,
                    'CompoundEncoded': c_e, 'DriverEncoded': d_e,
                    'CompoundAge': c_e * i, 'LapNumber': lap
                })
            preds = model.predict(pd.DataFrame(rows))
            return [float(p) * lap_time_mult for p in preds]

        is_two_stop = stint3 and stint3 != 'None'
        if is_two_stop:
            s1 = predict_stint(stint1, pit_lap, 1)
            s2 = predict_stint(stint2, pit_lap2 - pit_lap, pit_lap + 1)
            s3 = predict_stint(stint3, race_length - pit_lap2, pit_lap2 + 1)
            total = sum(s1) + pit_loss + sum(s2) + pit_loss + sum(s3)
            stints = [
                {'compound': stint1, 'laps': list(range(1, pit_lap+1)), 'times': s1},
                {'compound': stint2, 'laps': list(range(pit_lap+1, pit_lap2+1)), 'times': s2},
                {'compound': stint3, 'laps': list(range(pit_lap2+1, race_length+1)), 'times': s3},
            ]
        else:
            s1 = predict_stint(stint1, pit_lap, 1)
            s2 = predict_stint(stint2, race_length - pit_lap, pit_lap + 1)
            total = sum(s1) + pit_loss + sum(s2)
            stints = [
                {'compound': stint1, 'laps': list(range(1, pit_lap+1)), 'times': s1},
                {'compound': stint2, 'laps': list(range(pit_lap+1, race_length+1)), 'times': s2},
            ]

        return {
            'total_time': round(total, 2),
            'total_time_fmt': f"{int(total//60)}:{total%60:06.3f}",
            'stints': stints,
            'sc_applied': sc_lap is not None,
            'rain_applied': rain_lap is not None,
        }
    except Exception as e:
        return {'error': str(e)}    
@app.get("/compare", response_class=HTMLResponse)
async def compare(request: Request, year: str = None, round: str = None):
    try:
        year_int  = int(year) if year and year != 'None' else None
        round_int = int(round) if round and round != 'None' else None
    except:
        year_int  = None
        round_int = None

    if year_int is None or round_int is None:
        latest    = get_latest_race()
        year_int  = latest['year']
        round_int = latest['round']

    seasons  = get_available_seasons()
    schedule = get_season_schedule(year_int)
    return templates.TemplateResponse(request=request, name="compare.html", context={
        "seasons": seasons, "schedule": schedule,
        "selected_year": year_int, "selected_round": round_int,
        "selected_session": "R", "active": "compare",
    })

@app.get("/api/compare/{year}/{round_number}", tags=["Deep Dive"])
async def api_compare(year: int, round_number: int):
    try:
        SESSION_TYPES = ['FP1', 'FP2', 'FP3', 'Q', 'SQ', 'S', 'R']
        results = {}
        for st in SESSION_TYPES:
            try:
                session = load_race_session(year, round_number, st)
                laps    = session.laps.copy()
                laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
                clean = laps[
                    (laps['LapTimeSeconds'] > 55) &
                    (laps['LapTimeSeconds'] < 200) &
                    (laps['IsAccurate'] == True)
                ].copy()
                if clean.empty:
                    continue
                best = clean.groupby('Driver')['LapTimeSeconds'].min().reset_index()
                best.columns = ['Driver', 'BestLap']
                best = best.sort_values('BestLap').reset_index(drop=True)
                best['Rank'] = range(1, len(best) + 1)
                best['GapToFastest'] = (best['BestLap'] - best['BestLap'].min()).round(3)
                avg = clean.groupby('Driver')['LapTimeSeconds'].mean().reset_index()
                avg.columns = ['Driver', 'AvgLap']
                merged = best.merge(avg, on='Driver')
                merged['BestLap'] = merged['BestLap'].round(3)
                merged['AvgLap']  = merged['AvgLap'].round(3)
                results[st] = {
                    'session_type': st,
                    'event': session.event['EventName'],
                    'drivers': merged.to_dict(orient='records'),
                    'fastest': float(best['BestLap'].min()),
                    'total_laps': len(clean)
                }
            except:
                continue
        if not results:
            return {"error": "No session data available"}
        return {"status": "success", "sessions": results, "year": year, "round": round_number}
    except Exception as e:
        return {"error": str(e)}

@app.get("/tyres", response_class=HTMLResponse)
async def tyres(request: Request, year: str = None, round: str = None, session_type: str = None):
    try:
        year_int  = int(year) if year and year != 'None' else None
        round_int = int(round) if round and round != 'None' else None
    except:
        year_int  = None
        round_int = None

    if year_int is None or round_int is None:
        latest    = get_latest_race()
        year_int  = latest['year']
        round_int = latest['round']
        session_type = session_type or latest['session_type']

    session_type = session_type or 'R'
    seasons  = get_available_seasons()
    schedule = get_season_schedule(year_int)
    try:
        session = load_race_session(year_int, round_int, session_type)
        laps    = session.laps.copy()
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        clean = laps[
            (laps['LapTimeSeconds'] > 55) &
            (laps['LapTimeSeconds'] < 200) &
            (laps['IsAccurate'] == True)
        ].dropna(subset=['TyreLife', 'Compound', 'LapTimeSeconds']).copy()

        compound_colors = {
            'SOFT': '#FF3333', 'MEDIUM': '#FFFF00',
            'HARD': '#CCCCCC', 'INTERMEDIATE': '#39B54A', 'WET': '#0067FF'
        }
        compounds_used = sorted(clean['Compound'].unique())

        deg_data = {}
        for comp in compounds_used:
            df_c = clean[clean['Compound'] == comp].copy()
            df_c['TyreLife'] = df_c['TyreLife'].astype(float).astype(int)
            curve = df_c.groupby('TyreLife')['LapTimeSeconds'].agg(
                mean='mean', std='std', count='count'
            ).reset_index()
            curve = curve[curve['count'] >= 2]
            deg_data[comp] = {
                'tyre_life': curve['TyreLife'].tolist(),
                'avg_time':  curve['mean'].round(3).tolist(),
                'std':       curve['std'].fillna(0).round(3).tolist(),
                'color':     compound_colors.get(comp, '#888')
            }

        comp_summary = []
        for comp in compounds_used:
            df_c = clean[clean['Compound'] == comp]
            comp_summary.append({
                'compound':   comp,
                'avg_lap':    builtins_round(df_c['LapTimeSeconds'].mean(), 3),
                'best_lap':   builtins_round(df_c['LapTimeSeconds'].min(), 3),
                'total_laps': len(df_c),
                'drivers':    df_c['Driver'].nunique(),
                'color':      compound_colors.get(comp, '#888')
            })
        comp_summary.sort(key=lambda x: x['avg_lap'])

        import json
        return templates.TemplateResponse(request=request, name="tyres.html", context={
            "seasons": seasons, "schedule": schedule,
            "selected_year": year_int, "selected_round": round_int,
            "selected_session": session_type, "active": "tyres",
            "event_name": session.event['EventName'],
            "deg_data": json.dumps(deg_data),
            "comp_summary": comp_summary,
            "compounds_used": compounds_used,
        })
    except Exception as e:
        print(f"TYRES ERROR: {e}")
        import traceback; traceback.print_exc()
        return templates.TemplateResponse(request=request, name="tyres.html", context={
            "seasons": seasons, "schedule": schedule,
            "selected_year": year_int, "selected_round": round_int,
            "selected_session": session_type, "active": "tyres",
            "event_name": "—", "deg_data": "{}", "comp_summary": [], "compounds_used": [],
            "error": str(e)
        })    
@app.get("/simulate", response_class=HTMLResponse)
async def simulate(request: Request, year: str = None, round: str = None, session_type: str = None):
    try:
        year_int  = int(year) if year and year != 'None' else None
        round_int = int(round) if round and round != 'None' else None
    except:
        year_int  = None
        round_int = None

    if year_int is None or round_int is None:
        latest    = get_latest_race()
        year_int  = latest['year']
        round_int = latest['round']

    session_type = 'R'  # Always race
    seasons      = get_available_seasons()
    schedule     = get_season_schedule(year_int)

    CIRCUIT_LAPS = {
        'Melbourne': 58, 'Sakhir': 57, 'Shanghai': 56,
        'Suzuka': 53, 'Miami': 57, 'Monaco': 78,
        'Montreal': 70, 'Barcelona': 66, 'Spielberg': 71,
        'Silverstone': 52, 'Budapest': 70, 'Spa': 44,
        'Zandvoort': 72, 'Monza': 53, 'Baku': 51,
        'Singapore': 62, 'Austin': 56, 'Mexico City': 71,
        'São Paulo': 71, 'Las Vegas': 50, 'Lusail': 57,
        'Yas Marina': 58, 'Miami Gardens': 57, 'Jeddah': 50,
        'Portimão': 66, 'Mugello': 59, 'Istanbul': 58,
        'Imola': 63, 'Nürburgring': 60
    }

    # Initialize with defaults
    race_happened = True
    fallback_year  = None
    fallback_event = None
    drivers        = []
    compounds      = ['SOFT', 'MEDIUM', 'HARD']
    race_laps      = 57
    event_name     = '—'
    circuit        = '—'
    driver_noise   = {}

    try:
        session    = load_race_session(year_int, round_int, 'R')
        event_name = session.event['EventName']
        circuit    = session.event['Location']
        laps_data  = session.laps.copy()
        laps_data['LapTimeSeconds'] = laps_data['LapTime'].dt.total_seconds()
        clean      = laps_data[laps_data['LapTimeSeconds'] > 55].copy()
        drivers    = sorted(session.laps['Driver'].unique().tolist())
        compounds  = sorted(clean['Compound'].dropna().unique().tolist())
        race_laps  = int(session.laps['LapNumber'].max()) if len(session.laps) > 0 else CIRCUIT_LAPS.get(circuit, 57)

        # Driver consistency (std of lap times per driver)
        driver_noise = {}
        for drv in drivers:
            drv_laps = clean[clean['Driver'] == drv]['LapTimeSeconds']
            driver_noise[drv] = builtins_round(float(drv_laps.std()), 3) if len(drv_laps) > 3 else 0.5
    except:
        race_happened = False

        try:
            event_info = fastf1.get_event(year_int, round_int)
        except Exception as e:
            pass

        try:
            fastf1.Cache.enable_cache(str(CACHE_DIR))
            event_info  = fastf1.get_event(year_int, round_int)
            event_name  = str(event_info.get('EventName', f'Round {round_int}'))
            circuit     = str(event_info.get('Location', ''))
            race_laps   = CIRCUIT_LAPS.get(circuit, 57)
            # Find same event last year — match by circuit Location (stable across name changes)
            hist_sched = fastf1.get_event_schedule(year_int - 1, include_testing=False)
            hist_match = hist_sched[hist_sched['Location'] == circuit]
            if len(hist_match) == 0:
                # Fallback: try name-based match
                hist_match = hist_sched[
                    hist_sched['EventName'].str.contains(
                        event_name.replace('Grand Prix', '').strip(), na=False
                    )
                ]
            if len(hist_match) > 0:
                hist_round   = int(hist_match.iloc[0]['RoundNumber'])
                session      = load_race_session(year_int - 1, hist_round, 'R')
                fallback_year  = year_int - 1
                fallback_event = f"{event_name} {year_int - 1}"
            else:
                latest  = get_latest_race()
                session = load_race_session(latest['year'], latest['round'], 'R')
                fallback_year  = latest['year']
                fallback_event = f"{session.event['EventName']} {latest['year']}"
            laps_data  = session.laps.copy()
            laps_data['LapTimeSeconds'] = laps_data['LapTime'].dt.total_seconds()
            clean      = laps_data[laps_data['LapTimeSeconds'] > 55].copy()
            drivers    = sorted(session.laps['Driver'].unique().tolist())
            compounds  = sorted(clean['Compound'].dropna().unique().tolist())
            driver_noise = {}
            for drv in drivers:
                drv_laps = clean[clean['Driver'] == drv]['LapTimeSeconds']
                driver_noise[drv] = builtins_round(float(drv_laps.std()), 3) if len(drv_laps) > 3 else 0.5
        except Exception as e2:
            import traceback; traceback.print_exc()

    import json
    return templates.TemplateResponse(request=request, name="simulate.html", context={
        "seasons": seasons, "schedule": schedule,
        "selected_year": year_int, "selected_round": round_int,
        "selected_session": 'R', "active": "simulate",
        "drivers": drivers, "compounds": compounds,
        "race_laps": race_laps, "event_name": event_name,
        "race_happened": race_happened,
        "fallback_year": fallback_year,
        "fallback_event": fallback_event,
        "driver_noise": json.dumps(driver_noise),
    })

@app.post("/api/montecarlo", tags=["System"])
async def montecarlo(req: dict = Body(...)):
    try:
        import numpy as np
        year           = req.get('year')
        round_number   = req.get('round')
        session_type   = 'R'
        driver         = req.get('driver')
        stint1         = req.get('stint1', 'HARD')
        stint2         = req.get('stint2', 'MEDIUM')
        stint3         = req.get('stint3', 'None')
        race_length    = req.get('race_length', 57)
        pit_loss       = req.get('pit_loss', 22.0)
        n_sims         = int(req.get('n_sims', 500))
        sc_prob        = req.get('sc_prob', 0.3)
        rain_intensity = req.get('rain_intensity', 0)   # 0=none,1=light,2=heavy
        rain_start_lap = req.get('rain_start_lap', 0)   # 0=no rain
        driver_noise   = req.get('driver_noise', 0.3)   # std of lap time noise
        optimize_pit   = req.get('optimize_pit', False)  # pit window optimizer
        pit_min        = req.get('pit_min', 10)
        pit_max        = req.get('pit_max', 50)

        race_data = get_race_model(year, round_number, session_type)
        model     = race_data['model']
        d_enc     = race_data['driver_enc']
        c_enc     = race_data['compound_enc']

        if driver not in race_data['drivers']:
            return {"error": f"Driver {driver} not in race data"}

        RAIN_MULT = {0: 1.0, 1: 1.15, 2: 1.35}
        RAIN_COMP = {1: 'INTERMEDIATE', 2: 'WET'}

        def predict_stint(compound, stint_len, start_lap, noise_std,
                         sc_laps=None, rain_int=0, rain_from=0, drying=True):
            # Switch compound for rain
            comp = compound
            if rain_int > 0 and rain_from > 0 and start_lap >= rain_from:
                rain_comp = RAIN_COMP.get(rain_int, 'INTERMEDIATE')
                if rain_comp in c_enc.classes_:
                    comp = rain_comp

            if comp not in c_enc.classes_:
                comp = compound
            if comp not in c_enc.classes_:
                return [90.0] * stint_len

            d_e = d_enc.transform([driver])[0]
            c_e = c_enc.transform([comp])[0]

            rows = [{'TyreLife': i, 'TyreLifeSquared': i**2,
                     'CompoundEncoded': c_e, 'DriverEncoded': d_e,
                     'CompoundAge': c_e * i, 'LapNumber': start_lap + i - 1}
                    for i in range(1, stint_len + 1)]
            preds = model.predict(pd.DataFrame(rows))

            result = []
            for j, p in enumerate(preds):
                lap       = start_lap + j
                sc_mult   = 1.3 if sc_laps and lap in sc_laps else 1.0
                # Rain effect — drying track reduces multiplier over time
                if rain_int > 0 and rain_from > 0 and lap >= rain_from:
                    laps_since_rain = lap - rain_from
                    if drying:
                        rain_mult = max(1.0, RAIN_MULT[rain_int] - laps_since_rain * 0.01)
                    else:
                        rain_mult = RAIN_MULT[rain_int]
                else:
                    rain_mult = 1.0
                noise = np.random.normal(0, noise_std)
                result.append(float(p) * sc_mult * rain_mult + noise)
            return result

        def run_sim(pit_lap, pit_lap2=None):
            sc_happens = np.random.random() < sc_prob
            sc_laps    = None
            sc_start   = None
            if sc_happens:
                sc_start    = np.random.randint(1, race_length - 5)
                sc_duration = np.random.randint(3, 8)
                sc_laps     = set(range(sc_start, sc_start + sc_duration))

            pl        = pit_loss + np.random.normal(0, 1.5)
            rain_from = rain_start_lap if rain_intensity > 0 else 0

            is_two_stop = stint3 and stint3 != 'None' and pit_lap2
            if is_two_stop:
                s1 = predict_stint(stint1, pit_lap, 1, driver_noise, sc_laps, rain_intensity, rain_from)
                s2 = predict_stint(stint2, pit_lap2 - pit_lap, pit_lap + 1, driver_noise, sc_laps, rain_intensity, rain_from)
                s3 = predict_stint(stint3, race_length - pit_lap2, pit_lap2 + 1, driver_noise, sc_laps, rain_intensity, rain_from)
                total = sum(s1) + pl + sum(s2) + pl + sum(s3)
            else:
                s1 = predict_stint(stint1, pit_lap, 1, driver_noise, sc_laps, rain_intensity, rain_from)
                s2 = predict_stint(stint2, race_length - pit_lap, pit_lap + 1, driver_noise, sc_laps, rain_intensity, rain_from)
                total = sum(s1) + pl + sum(s2)

            return total, sc_happens, sc_start

        pit_lap  = req.get('pit_lap', 25)
        pit_lap2 = req.get('pit_lap2', 40)

        # ── Pit Window Optimizer ──────────────────────────
        pit_optimizer_results = []
        if optimize_pit:
            for test_pit in range(int(pit_min), int(pit_max) + 1, 2):
                test_totals = [run_sim(test_pit)[0] for _ in range(100)]
                pit_optimizer_results.append({
                    'pit_lap': test_pit,
                    'p50':     round(float(np.percentile(test_totals, 50)), 2),
                    'p10':     round(float(np.percentile(test_totals, 10)), 2),
                    'p90':     round(float(np.percentile(test_totals, 90)), 2),
                })
            best_pit = min(pit_optimizer_results, key=lambda x: x['p50'])
        else:
            best_pit = None

        # ── Main simulation ───────────────────────────────
        totals, sc_totals, nosc_totals = [], [], []
        early_sc, late_sc = [], []

        for _ in range(n_sims):
            total, sc_happened, sc_start_lap = run_sim(pit_lap, pit_lap2 if stint3 and stint3 != 'None' else None)
            totals.append(total)
            if sc_happened:
                sc_totals.append(total)
                if sc_start_lap and sc_start_lap < race_length // 2:
                    early_sc.append(total)
                else:
                    late_sc.append(total)
            else:
                nosc_totals.append(total)

        totals = sorted(totals)
        p10 = float(np.percentile(totals, 10))
        p50 = float(np.percentile(totals, 50))
        p90 = float(np.percentile(totals, 90))

        def fmt(t):
            return f"{int(t//60)}:{t%60:06.3f}"

        return {
            "status":     "success",
            "n_sims":     n_sims,
            "p10":        round(p10, 2), "p10_fmt": fmt(p10),
            "p50":        round(p50, 2), "p50_fmt": fmt(p50),
            "p90":        round(p90, 2), "p90_fmt": fmt(p90),
            "range":      round(p90 - p10, 2),
            "risk_score": round((p90 - p10) / p50 * 100, 1),
            "sc_avg":     round(float(np.mean(sc_totals)), 2) if sc_totals else None,
            "nosc_avg":   round(float(np.mean(nosc_totals)), 2) if nosc_totals else None,
            "early_sc_avg": round(float(np.mean(early_sc)), 2) if early_sc else None,
            "late_sc_avg":  round(float(np.mean(late_sc)), 2) if late_sc else None,
            "sc_pct":     round(len(sc_totals) / n_sims * 100, 1),
            "rain_applied": rain_intensity > 0 and rain_start_lap > 0,
            "distribution": totals[::max(1, len(totals)//100)],
            "pit_optimizer": pit_optimizer_results,
            "best_pit":      best_pit,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

@app.post("/api/scenario", tags=["System"])
async def scenario(req: dict = Body(...)):
    try:
        import numpy as np
        year         = req.get('year')
        round_number = req.get('round')
        driver       = req.get('driver')
        race_length  = req.get('race_length', 57)
        pit_loss     = req.get('pit_loss', 22.0)
        sc_prob      = req.get('sc_prob', 0.3)
        n_sims       = int(req.get('n_sims', 300))
        scenarios    = req.get('scenarios', [])
        driver_noise = req.get('driver_noise', 0.3)
        risk_tolerance = req.get('risk_tolerance', 50)  # 0=conservative, 100=aggressive

        race_data = get_race_model(year, round_number, 'R')
        model     = race_data['model']
        d_enc     = race_data['driver_enc']
        c_enc     = race_data['compound_enc']

        if driver not in race_data['drivers']:
            return {"error": f"Driver {driver} not in race data"}

        results = []
        all_p50s = []

        for sc in scenarios:
            name    = sc.get('name', 'Strategy')
            stint1  = sc.get('stint1', 'HARD')
            stint2  = sc.get('stint2', 'MEDIUM')
            pit_lap = int(sc.get('pit_lap', 25))

            totals = []
            for _ in range(n_sims):
                sc_happens = np.random.random() < sc_prob
                sc_laps = None
                if sc_happens:
                    sc_start = np.random.randint(1, race_length - 5)
                    sc_laps  = set(range(sc_start, sc_start + np.random.randint(3, 8)))

                pl   = pit_loss + np.random.normal(0, 1.5)
                d_e  = d_enc.transform([driver])[0]
                c1_e = c_enc.transform([stint1])[0] if stint1 in c_enc.classes_ else 0
                c2_e = c_enc.transform([stint2])[0] if stint2 in c_enc.classes_ else 0

                def sim_stint(c_e, length, start):
                    rows = [{'TyreLife': i, 'TyreLifeSquared': i**2,
                             'CompoundEncoded': c_e, 'DriverEncoded': d_e,
                             'CompoundAge': c_e*i, 'LapNumber': start+i-1}
                            for i in range(1, length+1)]
                    preds = model.predict(pd.DataFrame(rows))
                    return sum(
                        float(p) * (1.3 if sc_laps and (start+j) in sc_laps else 1.0)
                        + np.random.normal(0, driver_noise)
                        for j, p in enumerate(preds)
                    )

                s1 = sim_stint(c1_e, pit_lap, 1)
                s2 = sim_stint(c2_e, race_length - pit_lap, pit_lap + 1)
                totals.append(s1 + pl + s2)

            totals_sorted = sorted(totals)
            p10 = float(np.percentile(totals_sorted, 10))
            p50 = float(np.percentile(totals_sorted, 50))
            p90 = float(np.percentile(totals_sorted, 90))
            all_p50s.append(p50)

            results.append({
                'name':       name,
                'stint1':     stint1,
                'stint2':     stint2,
                'pit_lap':    pit_lap,
                'p10':        round(p10, 2),
                'p50':        round(p50, 2),
                'p90':        round(p90, 2),
                'best':       round(float(min(totals)), 2),
                'worst':      round(float(max(totals)), 2),
                'risk_score': round((p90 - p10) / p50 * 100, 1),
                'distribution': totals_sorted[::max(1, len(totals_sorted)//50)],
                'raw_totals': totals,
            })

        # ── Win probability ───────────────────────────────
        # For each simulation index, find which strategy had lowest time
        win_counts = {r['name']: 0 for r in results}
        for sim_i in range(n_sims):
            sim_times = {r['name']: r['raw_totals'][sim_i] for r in results}
            winner    = min(sim_times, key=sim_times.get)
            win_counts[winner] += 1

        for r in results:
            r['win_pct'] = round(win_counts[r['name']] / n_sims * 100, 1)
            del r['raw_totals']  # remove before sending

        # ── Recommended strategy ──────────────────────────
        # risk_tolerance 0 = minimize P90 (conservative)
        # risk_tolerance 100 = minimize P10 (aggressive)
        for r in results:
            weight     = risk_tolerance / 100
            r['score'] = (1 - weight) * r['p90'] + weight * r['p10']

        results.sort(key=lambda x: x['score'])
        recommended = results[0]['name']

        for i, r in enumerate(results):
            r['rank'] = i + 1
            r['recommended'] = r['name'] == recommended
            del r['score']

        return {
            "status":      "success",
            "scenarios":   results,
            "n_sims":      n_sims,
            "recommended": recommended,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}                    
@app.get("/drivers", response_class=HTMLResponse)
async def drivers_page(request: Request, year: str = None):
    try:
        year_int = int(year) if year and year != 'None' else None
    except:
        year_int = None
    if year_int is None:
        year_int = get_latest_race()['year']

    seasons  = get_available_seasons()
    schedule = get_season_schedule(year_int)

    return templates.TemplateResponse(request=request, name="drivers.html", context={
        "seasons":          seasons,
        "schedule":         schedule,
        "selected_year":    year_int,
        "selected_round":   None,
        "selected_session": "R",
        "active":           "drivers",
    })

@app.get("/api/driver-alltime-similar/{driver}", tags=["Deep Dive"])
async def driver_alltime_similar(driver: str, current_year: int = 2025):
    """Find most similar drivers across all seasons 2018-2026."""
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity

        DATA_DIR = Path("data")
        df = pd.read_parquet(DATA_DIR / "all_races_2018_2026.parquet")
        df = df[df['SessionType'] == 'R'].copy()

        STYLE_FEATURES = [
            'LapTimeZ', 'TyreLife', 'LapNumber'
        ]

        # Per-driver aggregation across all seasons
        driver_stats = {}
        for drv, grp in df.groupby('Driver'):
            if len(grp) < 50:
                continue
            stats = {
                'consistency':    float(grp['LapTimeZ'].std()),
                'avg_tyre_life':  float(grp['TyreLife'].mean()),
                'pace_z':         float(grp['LapTimeZ'].mean()),
                'lap_count':      len(grp),
                'seasons':        sorted(grp['Season'].unique().tolist()),
            }
            driver_stats[drv] = stats

        if driver not in driver_stats:
            return {"error": f"Driver {driver} not found in historical data"}

        # Build feature matrix
        drivers  = list(driver_stats.keys())
        feat_mat = np.array([[
            driver_stats[d]['consistency'],
            driver_stats[d]['avg_tyre_life'],
            driver_stats[d]['pace_z'],
        ] for d in drivers])

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(feat_mat)
        sim_mat  = cosine_similarity(X_scaled)

        drv_idx  = drivers.index(driver)
        sims     = [(drivers[j], round(float(sim_mat[drv_idx][j]), 3))
                    for j in range(len(drivers)) if j != drv_idx]
        sims.sort(key=lambda x: x[1], reverse=True)

        results = []
        for sim_drv, score in sims[:10]:
            results.append({
                'driver':  sim_drv,
                'score':   score,
                'pct':     round(score * 100),
                'seasons': driver_stats[sim_drv]['seasons'],
                'laps':    driver_stats[sim_drv]['lap_count'],
            })

        return {
            "status":  "success",
            "driver":  driver,
            "similar": results
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/driver-embeddings/{year}", tags=["Deep Dive"])
async def driver_embeddings(year: int, rounds: str = "all", mode: str = "season"):
    """
    mode='season' → compute fresh for selected year (current behavior)
    mode='alltime' → serve from pre-computed cache, project new drivers in
    """
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import cosine_similarity

        DATA_DIR   = Path("data")
        MODELS_DIR = Path("models")

        STYLE_FEATURES = [
            'brake_aggression', 'throttle_smoothness', 'throttle_attack',
            'top_speed', 'coasting_ratio', 'gear_aggression',
            'consistency', 'tyre_management'
        ]

        TEAM_COLORS = {
            'Red Bull Racing': '#3671C6', 'Ferrari': '#E8002D',
            'Mercedes': '#27F4D2', 'McLaren': '#FF8000',
            'Aston Martin': '#229971', 'Alpine': '#FF87BC',
            'Williams': '#64C4FF', 'RB': '#6692FF',
            'Racing Bulls': '#6692FF', 'Kick Sauber': '#52E252',
            'Haas F1 Team': '#B6BABD', 'AlphaTauri': '#5E8FAA',
            'Alfa Romeo': '#C92D4B', 'Renault': '#FFF500',
        }

        # ── All-time mode: serve from cache ───────────────
        if mode == 'alltime':
            import json
            cache_path = DATA_DIR / "driver_embeddings_alltime.json"
            if not cache_path.exists():
                return {"error": "All-time embeddings not computed yet. Run notebook 05_driver_embeddings.ipynb first."}

            with open(cache_path) as f:
                db = json.load(f)

            # Get current team for each driver from latest season
            try:
                fastf1.Cache.enable_cache(str(CACHE_DIR))
                latest_session = load_race_session(year, 1, 'R')
                team_map = {}
                if hasattr(latest_session, 'results') and latest_session.results is not None:
                    for _, row in latest_session.results.iterrows():
                        team_map[row.get('Abbreviation','')] = row.get('TeamName','')
            except:
                team_map = {}

            # Assign colors based on most recent team
            for d in db['drivers']:
                drv = d['driver']
                team = team_map.get(drv, '')
                d['color'] = TEAM_COLORS.get(team, '#888888')
                d['team']  = team or f"{d['seasons'][-1]} driver"

            return {
                "status":       "success",
                "year":         "2018-2026",
                "mode":         "alltime",
                "drivers":      db['drivers'],
                "features":     db['features'],
                "pca_variance": [round(v*100, 1) for v in db['pca_variance']],
            }

        # ── Season mode: compute fresh (existing behavior) ─
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        schedule  = fastf1.get_event_schedule(year, include_testing=False)
        completed = schedule[
            pd.to_datetime(schedule['EventDate']) < pd.Timestamp.now()
        ]

        if rounds == "last5":
            completed = completed.tail(5)
        elif rounds == "last10":
            completed = completed.tail(10)

        if len(completed) == 0:
            return {"error": "No completed races found"}

        driver_race_features = {}

        for _, event in completed.iterrows():
            round_num  = int(event['RoundNumber'])
            event_name = event['EventName']
            try:
                session = load_race_session(year, round_num, 'R')
                laps    = session.laps.copy()
                laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
                clean = laps[
                    (laps['LapTimeSeconds'] > 60) &
                    (laps['LapTimeSeconds'] < 200) &
                    (laps['IsAccurate'] == True)
                ].copy()
                if clean.empty:
                    continue

                team_map = {}
                if hasattr(session, 'results') and session.results is not None:
                    for _, row in session.results.iterrows():
                        team_map[row.get('Abbreviation','')] = row.get('TeamName','')

                for drv in clean['Driver'].unique():
                    drv_laps = clean[clean['Driver'] == drv].copy()
                    if len(drv_laps) < 5:
                        continue

                    feats = {}
                    try:
                        fastest_lap = session.laps.pick_drivers(drv).pick_fastest()
                        tel = fastest_lap.get_car_data().add_distance()
                        if len(tel) > 10:
                            braking = tel[tel['Brake'] == True]
                            feats['brake_aggression']    = float(abs(braking['Speed'].diff().dropna().mean())) if len(braking) > 5 else 0.0
                            feats['throttle_smoothness'] = float(tel['Throttle'].std())
                            accel = tel[tel['Speed'].diff() > 0]
                            feats['throttle_attack']     = float(accel['Throttle'].mean()) if len(accel) > 5 else 50.0
                            feats['top_speed']           = float(tel['Speed'].quantile(0.95))
                            coasting = tel[(tel['Throttle'] < 5) & (tel['Brake'] == False)]
                            feats['coasting_ratio']      = float(len(coasting) / len(tel) * 100)
                            mid_speed = tel[(tel['Speed'] > 100) & (tel['Speed'] < 200)]
                            feats['gear_aggression']     = float(mid_speed['nGear'].mean()) if len(mid_speed) > 5 else 5.0
                        else:
                            for f in ['brake_aggression','throttle_smoothness','throttle_attack','top_speed','coasting_ratio','gear_aggression']:
                                feats[f] = np.nan
                    except:
                        for f in ['brake_aggression','throttle_smoothness','throttle_attack','top_speed','coasting_ratio','gear_aggression']:
                            feats[f] = np.nan

                    feats['consistency']     = float(drv_laps['LapTimeSeconds'].std())
                    early = drv_laps[drv_laps['TyreLife'] <= 5]['LapTimeSeconds'].mean()
                    late  = drv_laps[drv_laps['TyreLife'] >= 15]['LapTimeSeconds'].mean()
                    feats['tyre_management'] = float(late - early) if not np.isnan(early) and not np.isnan(late) else np.nan
                    feats['team']            = team_map.get(drv, 'Unknown')

                    if drv not in driver_race_features:
                        driver_race_features[drv] = []
                    driver_race_features[drv].append(feats)

            except Exception as e:
                print(f"Skipping {event_name}: {e}")
                continue

        if not driver_race_features:
            return {"error": "Could not extract features from any race"}

        driver_profiles = []
        for drv, race_list in driver_race_features.items():
            profile = {'driver': drv, 'team': race_list[0].get('team','Unknown'), 'races': len(race_list)}
            for feat in STYLE_FEATURES:
                vals = [r[feat] for r in race_list if not np.isnan(r.get(feat, np.nan))]
                profile[feat] = float(np.mean(vals)) if vals else 0.0
            driver_profiles.append(profile)

        if len(driver_profiles) < 3:
            return {"error": "Not enough driver data for embeddings"}

        feat_matrix = np.array([[p[f] for f in STYLE_FEATURES] for p in driver_profiles])
        col_means   = np.nanmean(feat_matrix, axis=0)
        for i in range(feat_matrix.shape[1]):
            feat_matrix[np.isnan(feat_matrix[:, i]), i] = col_means[i]

        # ── Check if all-time PCA exists → project into it ──
        pca_path    = MODELS_DIR / "driver_embedding_pca.pkl"
        scaler_path = MODELS_DIR / "driver_embedding_scaler.pkl"

        if pca_path.exists() and scaler_path.exists():
            # Project current season drivers into historical embedding space
            saved_scaler = joblib.load(scaler_path)
            saved_pca    = joblib.load(pca_path)
            X_scaled     = saved_scaler.transform(feat_matrix)
            coords       = saved_pca.transform(X_scaled)
        else:
            # Fallback: fit fresh
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(feat_matrix)
            pca      = PCA(n_components=2)
            coords   = pca.fit_transform(X_scaled)
            pca_variance = pca.explained_variance_ratio_.tolist()

        pca_variance = saved_pca.explained_variance_ratio_.tolist() if pca_path.exists() else pca.explained_variance_ratio_.tolist()

        sim_matrix   = cosine_similarity(X_scaled)
        driver_names = [p['driver'] for p in driver_profiles]

        results = []
        for i, p in enumerate(driver_profiles):
            sims = [(driver_names[j], round(float(sim_matrix[i][j]), 3))
                    for j in range(len(driver_names)) if j != i]
            sims.sort(key=lambda x: x[1], reverse=True)
            results.append({
                'driver':   p['driver'],
                'team':     p['team'],
                'color':    TEAM_COLORS.get(p['team'], '#888888'),
                'races':    p['races'],
                'x':        round(float(coords[i][0]), 4),
                'y':        round(float(coords[i][1]), 4),
                'similar':  sims[:3],
                'features': {f: round(p[f], 3) for f in STYLE_FEATURES},
            })

        return {
            "status":       "success",
            "year":         year,
            "mode":         "season",
            "rounds":       rounds,
            "drivers":      results,
            "features":     STYLE_FEATURES,
            "pca_variance": [round(v*100, 1) for v in pca_variance],
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}
@app.post("/api/counterfactual", tags=["Intelligence"])
async def counterfactual(req: dict = Body(...)):
    """
    What-if analysis: change grid position, tyre, pit timing
    and predict how finishing position would change.
    """
    try:
        import numpy as np
        year         = req.get('year')
        round_number = req.get('round')
        session_type = req.get('session_type', 'R')
        driver       = req.get('driver')

        # Counterfactual changes
        cf_grid     = req.get('cf_grid', None)      # new grid position
        cf_compound = req.get('cf_compound', None)   # new tyre choice
        cf_pit_lap  = req.get('cf_pit_lap', None)    # new first pit lap
        cf_stops    = req.get('cf_stops', None)      # new number of stops

        cache_key = f"postrace_{year}_{round_number}_{session_type}"
        if cache_key not in postrace_cache:
            return {"error": "Run post-race prediction first — load the race on Predictions page"}

        predictor = postrace_cache[cache_key]['predictor']
        session   = load_race_session(year, round_number, session_type)

        # Get actual features for this driver
        df = predictor.prepare_features(session)
        if driver not in df['Driver'].values:
            return {"error": f"Driver {driver} not found in race data"}

        df['CompoundEnc'] = predictor.compound_enc.transform(df['PrimaryCompound'])
        df['DriverEnc']   = predictor.driver_enc.transform(df['Driver'])

        # ── Baseline prediction (actual race) ────────────
        X_base   = df[predictor.feature_names].fillna(0).values.astype(np.float32)
        X_scaled = predictor.scaler.transform(X_base)
        base_preds = predictor.model.predict(X_scaled)

        df['BasePred'] = base_preds
        df_sorted      = df.sort_values('BasePred').reset_index(drop=True)
        df_sorted['BasePosition'] = range(1, len(df_sorted) + 1)

        drv_row     = df[df['Driver'] == driver].iloc[0].copy()
        base_rank   = int(df_sorted[df_sorted['Driver'] == driver]['BasePosition'].values[0])
        actual_pos  = int(drv_row.get('ActualPosition', base_rank))

        # ── Counterfactual prediction ─────────────────────
        df_cf = df.copy()
        drv_idx = df_cf[df_cf['Driver'] == driver].index[0]

        changes = {}
        if cf_grid is not None:
            old_grid = float(df_cf.at[drv_idx, 'GridPosition'])
            df_cf.at[drv_idx, 'GridPosition']     = float(cf_grid)
            df_cf.at[drv_idx, 'PositionsGained']  = float(cf_grid) - float(df_cf.at[drv_idx, 'FinalPositionLap'])
            changes['Grid Position'] = {'from': int(old_grid), 'to': int(cf_grid)}

        if cf_compound is not None and cf_compound in predictor.compound_enc.classes_:
            old_comp = df_cf.at[drv_idx, 'PrimaryCompound']
            df_cf.at[drv_idx, 'PrimaryCompound'] = cf_compound
            df_cf.at[drv_idx, 'CompoundEnc']     = float(predictor.compound_enc.transform([cf_compound])[0])
            changes['Tyre Compound'] = {'from': old_comp, 'to': cf_compound}

        if cf_pit_lap is not None:
            old_pit = float(df_cf.at[drv_idx, 'FirstPitLap'])
            df_cf.at[drv_idx, 'FirstPitLap'] = float(cf_pit_lap)
            changes['First Pit Lap'] = {'from': int(old_pit), 'to': int(cf_pit_lap)}

        if cf_stops is not None:
            old_stops = float(df_cf.at[drv_idx, 'NumStops'])
            df_cf.at[drv_idx, 'NumStops'] = float(cf_stops)
            changes['Pit Stops'] = {'from': int(old_stops), 'to': int(cf_stops)}

        X_cf      = df_cf[predictor.feature_names].fillna(0).values.astype(np.float32)
        X_cf_sc   = predictor.scaler.transform(X_cf)
        cf_preds  = predictor.model.predict(X_cf_sc)

        df_cf['CfPred'] = cf_preds
        df_cf_sorted    = df_cf.sort_values('CfPred').reset_index(drop=True)
        df_cf_sorted['CfPosition'] = range(1, len(df_cf_sorted) + 1)
        cf_rank = int(df_cf_sorted[df_cf_sorted['Driver'] == driver]['CfPosition'].values[0])

        # ── Feature impact analysis ───────────────────────
        base_score = float(df[df['Driver'] == driver]['BasePred'].values[0])
        cf_score   = float(df_cf[df_cf['Driver'] == driver]['CfPred'].values[0])

        # Full grid results
        grid_results = []
        for _, row in df_cf_sorted.iterrows():
            base_r = int(df_sorted[df_sorted['Driver'] == row['Driver']]['BasePosition'].values[0]) if row['Driver'] in df_sorted['Driver'].values else None
            grid_results.append({
                'driver':       row['Driver'],
                'cf_position':  int(row['CfPosition']),
                'base_position': base_r,
                'delta':        (base_r - int(row['CfPosition'])) if base_r else 0,
                'is_subject':   row['Driver'] == driver
            })

        return {
            "status":         "success",
            "driver":         driver,
            "actual_position": actual_pos,
            "base_position":  base_rank,
            "cf_position":    cf_rank,
            "position_delta": base_rank - cf_rank,
            "base_score":     round(base_score, 3),
            "cf_score":       round(cf_score, 3),
            "changes":        changes,
            "grid_results":   grid_results,
            "compounds":      list(predictor.compound_enc.classes_),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

@app.get("/counterfactual", response_class=HTMLResponse)
async def counterfactual_page(request: Request, year: str = None, round: str = None, session_type: str = None):
    try:
        year_int  = int(year) if year and year != 'None' else None
        round_int = int(round) if round and round != 'None' else None
    except:
        year_int  = None
        round_int = None

    if year_int is None or round_int is None:
        latest    = get_latest_race()
        year_int  = latest['year']
        round_int = latest['round']

    session_type = session_type or 'R'
    seasons      = get_available_seasons()
    schedule     = get_season_schedule(year_int)

    try:
        session   = load_race_session(year_int, round_int, session_type)
        drivers   = sorted(session.laps['Driver'].unique().tolist())
        laps_data = session.laps.copy()
        laps_data['LapTimeSeconds'] = laps_data['LapTime'].dt.total_seconds()
        clean     = laps_data[laps_data['LapTimeSeconds'] > 55].copy()
        compounds = sorted(clean['Compound'].dropna().unique().tolist())
        race_laps = int(session.laps['LapNumber'].max()) if len(session.laps) > 0 else 57
        event_name = session.event['EventName']

        # Get grid positions
        grid_map = {}
        if hasattr(session, 'results') and session.results is not None:
            for _, row in session.results.iterrows():
                abbr = row.get('Abbreviation', '')
                grid = row.get('GridPosition', None)
                if abbr and grid:
                    try: grid_map[abbr] = int(grid)
                    except: pass
    except SessionNotAvailableError:
        # Race hasn't happened — counterfactual needs post-race data
        return templates.TemplateResponse(request=request, name="no_data.html", status_code=200, context={
            "seasons":          seasons,
            "schedule":         schedule,
            "selected_year":    year_int,
            "selected_round":   round_int,
            "selected_session": session_type,
            "active":           "none",
            "year":             year_int,
            "round_number":     round_int,
            "session_type":     session_type,
            "error_detail":     "Counterfactual analysis requires the race to have happened — load it on the Predictions page first.",
        })
    except Exception as e:
        drivers    = []
        compounds  = ['SOFT', 'MEDIUM', 'HARD']
        race_laps  = 57
        event_name = '—'
        grid_map   = {}

    import json
    return templates.TemplateResponse(request=request, name="counterfactual.html", context={
        "seasons": seasons, "schedule": schedule,
        "selected_year": year_int, "selected_round": round_int,
        "selected_session": session_type, "active": "counterfactual",
        "drivers": drivers, "compounds": compounds,
        "race_laps": race_laps, "event_name": event_name,
        "grid_map": json.dumps(grid_map),
    })

@app.get("/circuit", response_class=HTMLResponse)
async def circuit_page(request: Request, year: str = None, round: str = None, session_type: str = None):
    try:
        year_int  = int(year) if year and year != 'None' else None
        round_int = int(round) if round and round != 'None' else None
    except:
        year_int  = None
        round_int = None

    if year_int is None or round_int is None:
        latest    = get_latest_race()
        year_int  = latest['year']
        round_int = latest['round']

    session_type = session_type or 'R'
    seasons      = get_available_seasons()
    schedule     = get_season_schedule(year_int)

    circuit    = '—'
    event_name = f'Round {round_int}'
    try:
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        ev         = fastf1.get_event(year_int, round_int)
        circuit    = str(ev.get('Location', '—'))
        event_name = str(ev.get('EventName', event_name))
    except Exception as e:
        print(f"Circuit metadata error: {e}")

    return templates.TemplateResponse(request=request, name="circuit.html", context={
        "seasons":          seasons,
        "schedule":         schedule,
        "selected_year":    year_int,
        "selected_round":   round_int,
        "selected_session": session_type,
        "active":           "circuit",
        "circuit":          circuit,
        "event_name":       event_name,
    })

@app.get("/api/circuit-intelligence/{circuit_name}", tags=["Deep Dive"])
async def circuit_intelligence(circuit_name: str):
    try:
        import numpy as np
        DATA_DIR = Path("data")
        df = pd.read_parquet(DATA_DIR / "all_races_2018_2026.parquet")
        df = df[df['SessionType'] == 'R'].copy()

        # Normalize circuit name
        circuit_name = circuit_name.replace('%20', ' ').replace('+', ' ')
        circ_df = df[df['Circuit'] == circuit_name].copy()

        if circ_df.empty:
            # Try partial match
            matches = df[df['Circuit'].str.contains(circuit_name, case=False, na=False)]
            if matches.empty:
                return {"error": f"No data for circuit: {circuit_name}"}
            circ_df = matches.copy()
            circuit_name = circ_df['Circuit'].iloc[0]

        # ── Basic stats ───────────────────────────────────
        total_laps    = len(circ_df)
        seasons_avail = sorted(circ_df['Season'].unique().tolist())
        avg_lap_time  = round(float(circ_df['LapTimeSeconds'].mean()), 3)
        best_lap_time = round(float(circ_df['LapTimeSeconds'].min()), 3)

        # ── Tyre degradation per compound ─────────────────
        compound_deg = {}
        for comp, grp in circ_df.groupby('Compound'):
            if len(grp) < 20:
                continue
            curve = grp.groupby('TyreLife')['LapTimeSeconds'].mean()
            if len(curve) > 3:
                # Degradation rate = slope of lap time vs tyre life
                x = curve.index.values
                y = curve.values
                slope = float(np.polyfit(x, y, 1)[0])
                compound_deg[comp] = {
                    'deg_rate':   round(slope, 4),
                    'avg_lap':    round(float(grp['LapTimeSeconds'].mean()), 3),
                    'best_lap':   round(float(grp['LapTimeSeconds'].min()), 3),
                    'total_laps': len(grp),
                    'tyre_life':  curve.index.tolist()[:20],
                    'avg_times':  curve.values.tolist()[:20],
                }

        # ── Team dominance ────────────────────────────────
        team_stats = []
        if 'Team' in circ_df.columns:
            for team, grp in circ_df.groupby('Team'):
                if len(grp) < 10:
                    continue
                team_stats.append({
                    'team':     team,
                    'avg_lap':  round(float(grp['LapTimeSeconds'].mean()), 3),
                    'best_lap': round(float(grp['LapTimeSeconds'].min()), 3),
                    'laps':     len(grp),
                })
            team_stats.sort(key=lambda x: x['avg_lap'])
            team_stats = team_stats[:10]

        # ── Driver dominance ──────────────────────────────
        driver_stats = []
        for drv, grp in circ_df.groupby('Driver'):
            if len(grp) < 10:
                continue
            driver_stats.append({
                'driver':   drv,
                'avg_lap':  round(float(grp['LapTimeSeconds'].mean()), 3),
                'best_lap': round(float(grp['LapTimeSeconds'].min()), 3),
                'seasons':  sorted(grp['Season'].unique().tolist()),
                'laps':     len(grp),
            })
        driver_stats.sort(key=lambda x: x['best_lap'])
        driver_stats = driver_stats[:10]

        # ── Lap time evolution across seasons ─────────────
        season_evolution = []
        for season, grp in circ_df.groupby('Season'):
            season_evolution.append({
                'season':   int(season),
                'avg_lap':  round(float(grp['LapTimeSeconds'].mean()), 3),
                'best_lap': round(float(grp['LapTimeSeconds'].min()), 3),
                'laps':     len(grp),
            })
        season_evolution.sort(key=lambda x: x['season'])

        # ── Overtaking difficulty proxy ───────────────────
        # High std of lap times = more variance = more overtaking/incidents
        lap_std = float(circ_df['LapTimeSeconds'].std())
        overtaking_index = round(min(100, lap_std * 10), 1)

        # ── SC frequency ─────────────────────────────────
        sc_laps  = len(circ_df[circ_df['UnderSC'] == True]) if 'UnderSC' in circ_df.columns else 0
        sc_ratio = round(sc_laps / total_laps * 100, 1) if total_laps > 0 else 0

        # ── Compound usage distribution ───────────────────
        compound_usage = circ_df['Compound'].value_counts().to_dict()

        return {
            "status":           "success",
            "circuit":          circuit_name,
            "total_laps":       total_laps,
            "seasons":          seasons_avail,
            "avg_lap_time":     avg_lap_time,
            "best_lap_time":    best_lap_time,
            "overtaking_index": overtaking_index,
            "sc_ratio":         sc_ratio,
            "compound_deg":     compound_deg,
            "compound_usage":   compound_usage,
            "team_stats":       team_stats,
            "driver_stats":     driver_stats,
            "season_evolution": season_evolution,
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

@app.get("/constructor", response_class=HTMLResponse)
async def constructor_page(request: Request, year: str = None, round: str = None, session_type: str = None):
    try:
        year_int  = int(year) if year and year != 'None' else None
        round_int = int(round) if round and round != 'None' else None
    except:
        year_int  = None
        round_int = None

    if year_int is None or round_int is None:
        latest    = get_latest_race()
        year_int  = latest['year']
        round_int = latest['round']

    session_type = session_type or 'R'
    seasons      = get_available_seasons()
    schedule     = get_season_schedule(year_int)

    # Get available teams
    try:
        DATA_DIR = Path("data")
        df = pd.read_parquet(DATA_DIR / "all_races_2018_2026.parquet")
        df = df[df['SessionType'] == 'R']
        teams = sorted(df['Team'].dropna().unique().tolist())
    except:
        teams = []

    return templates.TemplateResponse(request=request, name="constructor.html", context={
        "seasons":          seasons,
        "schedule":         schedule,
        "selected_year":    year_int,
        "selected_round":   round_int,
        "selected_session": session_type,
        "active":           "constructor",
        "teams":            teams,
    })

@app.get("/api/constructor-intelligence/{team_name}", tags=["Deep Dive"])
async def constructor_intelligence(team_name: str):
    try:
        import numpy as np
        DATA_DIR = Path("data")
        df = pd.read_parquet(DATA_DIR / "all_races_2018_2026.parquet")
        df = df[df['SessionType'] == 'R'].copy()

        team_name = team_name.replace('%20', ' ').replace('+', ' ')
        team_df   = df[df['Team'] == team_name].copy()

        if team_df.empty:
            matches = df[df['Team'].str.contains(team_name, case=False, na=False)]
            if matches.empty:
                return {"error": f"No data for team: {team_name}"}
            team_df   = matches.copy()
            team_name = team_df['Team'].iloc[0]

        # ── Basic stats ───────────────────────────────────
        seasons_active = sorted(team_df['Season'].unique().tolist())
        total_laps     = len(team_df)
        drivers_used   = sorted(team_df['Driver'].unique().tolist())

        # ── Season-by-season performance ──────────────────
        season_perf = []
        for season, grp in team_df.groupby('Season'):
            season_perf.append({
                'season':   int(season),
                'avg_lap':  round(float(grp['LapTimeSeconds'].mean()), 3),
                'best_lap': round(float(grp['LapTimeSeconds'].min()), 3),
                'laps':     len(grp),
                'drivers':  sorted(grp['Driver'].unique().tolist()),
            })
        season_perf.sort(key=lambda x: x['season'])

        # ── vs field performance (normalized) ─────────────
        season_relative = []
        for season, grp in team_df.groupby('Season'):
            all_season = df[df['Season'] == season]
            field_avg  = float(all_season['LapTimeSeconds'].mean())
            team_avg   = float(grp['LapTimeSeconds'].mean())
            season_relative.append({
                'season': int(season),
                'delta':  round(team_avg - field_avg, 3),  # + = slower than field
            })
        season_relative.sort(key=lambda x: x['season'])

        # ── Teammate comparisons ──────────────────────────
        teammate_comparisons = []
        for season, grp in team_df.groupby('Season'):
            season_drivers = grp['Driver'].unique()
            if len(season_drivers) >= 2:
                pairs = []
                for i in range(len(season_drivers)):
                    for j in range(i+1, len(season_drivers)):
                        d1    = season_drivers[i]
                        d2    = season_drivers[j]
                        d1avg = float(grp[grp['Driver']==d1]['LapTimeSeconds'].mean())
                        d2avg = float(grp[grp['Driver']==d2]['LapTimeSeconds'].mean())
                        pairs.append({
                            'season':  int(season),
                            'driver1': d1,
                            'driver2': d2,
                            'avg1':    round(d1avg, 3),
                            'avg2':    round(d2avg, 3),
                            'delta':   round(d1avg - d2avg, 3),
                            'winner':  d1 if d1avg < d2avg else d2
                        })
                teammate_comparisons.extend(pairs)

        # ── Best circuits ──────────────────────────────────
        circuit_perf = []
        for circuit, grp in team_df.groupby('Circuit'):
            all_circ  = df[df['Circuit'] == circuit]
            field_avg = float(all_circ['LapTimeSeconds'].mean())
            team_avg  = float(grp['LapTimeSeconds'].mean())
            circuit_perf.append({
                'circuit': circuit,
                'delta':   round(team_avg - field_avg, 3),
                'laps':    len(grp),
            })
        circuit_perf.sort(key=lambda x: x['delta'])
        best_circuits  = circuit_perf[:5]   # most negative = fastest vs field
        worst_circuits = circuit_perf[-5:][::-1]

        # ── Development trajectory ────────────────────────
        # Linear trend of performance vs field over seasons
        if len(season_relative) >= 3:
            x     = np.array([s['season'] for s in season_relative])
            y     = np.array([s['delta'] for s in season_relative])
            slope = float(np.polyfit(x, y, 1)[0])
            trend = 'improving' if slope < -0.01 else 'declining' if slope > 0.01 else 'stable'
        else:
            slope = 0.0
            trend = 'insufficient data'

        # ── Compound preferences ──────────────────────────
        compound_dist = team_df['Compound'].value_counts().to_dict()

        return {
            "status":                "success",
            "team":                  team_name,
            "seasons_active":        seasons_active,
            "total_laps":            total_laps,
            "drivers_used":          drivers_used,
            "season_perf":           season_perf,
            "season_relative":       season_relative,
            "teammate_comparisons":  teammate_comparisons,
            "best_circuits":         best_circuits,
            "worst_circuits":        worst_circuits,
            "trend":                 trend,
            "trend_slope":           round(slope, 4),
            "compound_dist":         compound_dist,
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    seasons  = get_available_seasons()
    latest   = get_latest_race()
    schedule = get_season_schedule(latest['year'])

    return templates.TemplateResponse(request=request, name="pipeline.html", context={
        "seasons":          seasons,
        "schedule":         schedule,
        "selected_year":    latest['year'],
        "selected_round":   latest['round'],
        "selected_session": "R",
        "active":           "pipeline",
    })

@app.get("/api/pipeline-status", tags=["System"])
async def pipeline_status():
    """Health check + data freshness for all pipeline components."""
    try:
        import os
        import json
        from datetime import datetime

        DATA_DIR   = Path("data")
        MODELS_DIR = Path("models")

        def file_info(path):
            if not path.exists():
                return None
            stat = path.stat()
            return {
                "name":      path.name,
                "size_kb":   round(stat.st_size / 1024, 1),
                "modified":  datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "age_days":  round((datetime.now().timestamp() - stat.st_mtime) / 86400, 1),
            }

        # ── Core dataset ───────────────────────────────────
        parquet_path = DATA_DIR / "all_races_2018_2026.parquet"
        dataset_info = file_info(parquet_path)
        dataset_rows = None
        dataset_seasons = []
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                dataset_rows = len(df)
                if 'Season' in df.columns:
                    dataset_seasons = sorted(df['Season'].unique().tolist())
            except Exception as e:
                pass

        # ── Models ─────────────────────────────────────────
        model_files = [
            "tire_degradation_model.pkl",
            "incident_isolation_forest.pkl",
            "incident_isolation_forest_practice.pkl",
            "incident_scaler.pkl",
            "incident_scaler_practice.pkl",
            "driver_encoder.pkl",
            "compound_encoder.pkl",
            "circuit_encoder.pkl",
            "driver_embedding_pca.pkl",
            "driver_embedding_scaler.pkl",
        ]
        models = []
        for mf in model_files:
            info = file_info(MODELS_DIR / mf)
            if info:
                models.append(info)

        # ── Model metadata ─────────────────────────────────
        metadata = {}
        meta_path = MODELS_DIR / "model_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except:
                pass

        # ── Driver embeddings ──────────────────────────────
        emb_path = DATA_DIR / "driver_embeddings_alltime.json"
        emb_info = file_info(emb_path)
        emb_drivers = 0
        if emb_path.exists():
            try:
                with open(emb_path) as f:
                    emb_data = json.load(f)
                emb_drivers = len(emb_data.get('drivers', []))
            except Exception as e:
                print(f"PIPELINE DEBUG ERROR: {e}")

        # ── FastF1 cache ───────────────────────────────────
        cache_size_mb = 0
        cache_seasons = []
        if CACHE_DIR.exists():
            try:
                total_size = 0
                for f in CACHE_DIR.rglob('*'):
                    if f.is_file():
                        total_size += f.stat().st_size
                cache_size_mb = round(total_size / (1024*1024), 1)
                cache_seasons = sorted([
                    d.name for d in CACHE_DIR.iterdir()
                    if d.is_dir() and d.name.isdigit()
                ])
            except:
                pass

        # ── Live health checks ─────────────────────────────
        health_checks = []

        # Check 1: Can we load model files?
        try:
            test_model = joblib.load(MODELS_DIR / "tire_degradation_model.pkl")
            health_checks.append({"name": "Tyre Degradation Model", "status": "healthy", "detail": "Loads successfully"})
        except Exception as e:
            health_checks.append({"name": "Tyre Degradation Model", "status": "error", "detail": str(e)})

        # Check 2: Can we read the dataset?
        if dataset_rows:
            health_checks.append({"name": "Training Dataset", "status": "healthy", "detail": f"{dataset_rows:,} rows readable"})
        else:
            health_checks.append({"name": "Training Dataset", "status": "error", "detail": "Could not read parquet file"})

        # Check 3: FastF1 cache accessible
        if cache_size_mb > 0:
            health_checks.append({"name": "FastF1 Cache", "status": "healthy", "detail": f"{cache_size_mb} MB across {len(cache_seasons)} seasons"})
        else:
            health_checks.append({"name": "FastF1 Cache", "status": "warning", "detail": "Cache empty or inaccessible"})

        # Check 4: Driver embeddings
        if emb_drivers > 0:
            health_checks.append({"name": "Driver Embeddings", "status": "healthy", "detail": f"{emb_drivers} drivers indexed"})
        else:
            health_checks.append({"name": "Driver Embeddings", "status": "warning", "detail": "Not computed yet"})

        # Check 5: Ergast API status — live ping with short timeout
        try:
            import requests
            resp = requests.get("https://api.jolpi.ca/ergast/f1/2024/1/results.json", timeout=3)
            if resp.status_code == 200:
                health_checks.append({"name": "Ergast API (jolpi.ca)", "status": "healthy", "detail": "Responding normally"})
            else:
                health_checks.append({"name": "Ergast API (jolpi.ca)", "status": "degraded", "detail": f"HTTP {resp.status_code} — using cached fallback (expected)"})
        except Exception:
            health_checks.append({"name": "Ergast API (jolpi.ca)", "status": "degraded", "detail": "Unreachable/timeout — using cached fallback (expected, deprecated upstream)"})

        # ── Overall status ─────────────────────────────────
        statuses = [h['status'] for h in health_checks]
        if 'error' in statuses:
            overall = 'error'
        elif 'warning' in statuses:
            overall = 'warning'
        else:
            overall = 'healthy'

        return {
            "status":          "success",
            "overall":         overall,
            "dataset":         dataset_info,
            "dataset_rows":    dataset_rows,
            "dataset_seasons": dataset_seasons,
            "models":          models,
            "model_metadata":  metadata,
            "driver_embeddings": emb_info,
            "driver_embeddings_count": emb_drivers,
            "cache_size_mb":   cache_size_mb,
            "cache_seasons":   cache_seasons,
            "health_checks":   health_checks,
            "checked_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

# ── Health ────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "cached_sessions": list(session_cache.keys()),
        "cached_models": list(model_cache.keys())
    }
@app.get("/api/sessions/{year}/{round_number}", tags=["Analysis"])
async def api_sessions(year: int, round_number: int):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        event = fastf1.get_event(year, round_number)
        from datetime import datetime, timezone

        # Name → code mapping
        name_to_code = {
            'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3',
            'Sprint Qualifying': 'SQ', 'Sprint Shootout': 'SQ',
            'Sprint': 'S', 'Qualifying': 'Q', 'Race': 'R'
        }

        sessions = []
        for i in range(1, 6):
            name = event.get(f'Session{i}')
            date = event.get(f'Session{i}Date')
            if not name or str(name) in ['None', '', 'nan']:
                continue
            # Only include sessions that have already happened
            if date is not None:
                try:
                    if hasattr(date, 'tzinfo') and date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    if date > now:
                        continue  # Skip future sessions
                except:
                    pass
            code = name_to_code.get(str(name), str(name))
            sessions.append({
                "name": str(name),
                "code": code
            })

        return {
            "sessions": sessions if sessions else [{"name": "Race", "code": "R"}],
            "round":    round_number,
            "year":     year
        }
    except Exception as e:
        print(f"Session fetch error: {e}")
        return {"sessions": [{"name": "Race", "code": "R"}]}  
@app.get("/api/shap/{year}/{round_number}/{session_type}", tags=["System"])
async def get_shap_explanation(year: int, round_number: int, session_type: str = 'R'):
    """Get SHAP explanations for post-race predictions."""
    try:
        cache_key = f"postrace_{year}_{round_number}_{session_type}"

        if cache_key not in postrace_cache:
            return {"status": "error", "message": "Run post-race prediction first"}

        predictor = postrace_cache[cache_key]['predictor']
        session   = load_race_session(year, round_number, session_type)
        shap_data = predictor.get_shap_values(session)

        return {
            "status": "success",
            "explanations": shap_data,
            "feature_names": predictor.feature_names
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}        
@app.get("/api/confidence/{year}/{round_number}/{session_type}", tags=["System"])
async def get_confidence_intervals(year: int, round_number: int, session_type: str = 'R'):
    """Get P10/P50/P90 confidence intervals for predictions."""
    try:
        cache_key = f"postrace_{year}_{round_number}_{session_type}"

        if cache_key not in postrace_cache:
            return {
                "status": "error",
                "message": "Run post-race prediction first"
            }

        predictor  = postrace_cache[cache_key]['predictor']
        session    = load_race_session(year, round_number, session_type)
        intervals  = predictor.get_confidence_intervals(session)

        return {
            "status":    "success",
            "intervals": intervals
        }

    except Exception as e:
        print(f"Confidence interval error: {e}")
        return {"status": "error", "message": str(e)}