"""
F1 Race Intelligence API
REST endpoints for tire prediction, incident detection, and strategy optimization.

Run with: uvicorn src.api:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ── App Initialization ───────────────────────────────────
app = FastAPI(
    title="F1 Race Intelligence API",
    description="ML-powered Formula 1 race analytics: tire degradation, incident detection, strategy optimization",
    version="1.0.0",
    contact={
        "name": "Sanjay Chowdary",
        "url": "https://github.com/sanjay-099/f1-race-intelligence",
    }
)

# CORS — allow dashboard to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

tire_model = joblib.load(MODELS_DIR / "tire_degradation_model.pkl")
compound_encoder = joblib.load(MODELS_DIR / "compound_encoder.pkl")
driver_encoder = joblib.load(MODELS_DIR / "driver_encoder.pkl")
iso_forest = joblib.load(MODELS_DIR / "incident_isolation_forest.pkl")
scaler = joblib.load(MODELS_DIR / "incident_scaler.pkl")

# ── Health Endpoint ──────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Welcome endpoint with API info."""
    return {
        "message": "🏎️ F1 Race Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "tire_prediction": "/predict/tire-stint",
            "incident_detection": "/detect/incidents",
            "strategy_optimization": "/optimize/strategy",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Check if API + models are loaded properly."""
    return {
        "status": "healthy",
        "models_loaded": {
            "tire_model": tire_model is not None,
            "isolation_forest": iso_forest is not None,
            "encoders": True
        },
        "compounds_available": list(compound_encoder.classes_),
        "drivers_available": list(driver_encoder.classes_),
        "total_drivers": len(driver_encoder.classes_)
    }
# ════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS (Pydantic schemas for validation)
# ════════════════════════════════════════════════════════════

class TireStintRequest(BaseModel):
    driver: str = Field(..., description="Driver code (e.g. HAM, VER)", example="HAM")
    compound: str = Field(..., description="Tire compound", example="SOFT")
    stint_length: int = Field(..., gt=0, le=60, description="Number of laps", example=20)
    start_lap: int = Field(default=1, ge=1, le=80, description="Race lap when stint starts")

class TireStintResponse(BaseModel):
    driver: str
    compound: str
    stint_length: int
    predicted_lap_times: List[float]
    cumulative_time: float
    avg_lap_time: float
    fastest_lap_time: float

class StrategyRequest(BaseModel):
    driver: str = Field(..., example="HAM")
    stint1_compound: str = Field(..., example="SOFT")
    stint2_compound: str = Field(..., example="HARD")
    race_length: int = Field(default=57, gt=20, le=80, example=57)
    pit_stop_loss: float = Field(default=22.0, gt=15.0, le=30.0)
    pit_lap_min: int = Field(default=10, gt=0)
    pit_lap_max: int = Field(default=45, gt=0)

class StrategyResponse(BaseModel):
    driver: str
    strategy: str
    optimal_pit_lap: int
    stint1_laps: int
    stint2_laps: int
    total_race_time: float
    pit_window_start: int
    pit_window_end: int
    top_5_strategies: List[dict]

class IncidentRequest(BaseModel):
    lap_time_seconds: float = Field(..., gt=60.0, lt=150.0, example=98.5)
    tyre_life: int = Field(..., ge=1, le=60, example=12)
    lap_number: int = Field(..., ge=1, le=80, example=15)

class IncidentResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    severity: str
    interpretation: str


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def predict_stint_internal(driver: str, compound: str, stint_length: int, start_lap: int = 1):
    """Predicts lap times for a stint using the tire degradation model."""
    if driver not in driver_encoder.classes_:
        raise HTTPException(status_code=400, detail=f"Driver '{driver}' not in dataset. Available: {list(driver_encoder.classes_)}")
    if compound not in compound_encoder.classes_:
        raise HTTPException(status_code=400, detail=f"Compound '{compound}' not supported. Available: {list(compound_encoder.classes_)}")
    
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
    predictions = tire_model.predict(stint_df)
    return predictions.tolist()


# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.post("/predict/tire-stint", response_model=TireStintResponse, tags=["ML Predictions"])
def predict_tire_stint(request: TireStintRequest):
    """
    Predict lap times for an entire stint using the trained XGBoost tire degradation model.
    
    Returns per-lap predictions plus aggregate statistics.
    """
    predictions = predict_stint_internal(
        driver=request.driver,
        compound=request.compound,
        stint_length=request.stint_length,
        start_lap=request.start_lap
    )
    
    return TireStintResponse(
        driver=request.driver,
        compound=request.compound,
        stint_length=request.stint_length,
        predicted_lap_times=[round(p, 3) for p in predictions],
        cumulative_time=round(sum(predictions), 3),
        avg_lap_time=round(sum(predictions) / len(predictions), 3),
        fastest_lap_time=round(min(predictions), 3)
    )


@app.post("/detect/incidents", response_model=IncidentResponse, tags=["ML Predictions"])
def detect_incident(request: IncidentRequest):
    """
    Detect if a given lap is anomalous using the Isolation Forest model.
    
    Returns anomaly flag, score, severity, and a human-readable interpretation.
    """
    X = np.array([[request.lap_time_seconds, request.tyre_life, request.lap_number]])
    X_scaled = scaler.transform(X)
    
    prediction = iso_forest.predict(X_scaled)[0]
    score = iso_forest.score_samples(X_scaled)[0]
    
    is_anomaly = bool(prediction == -1)
    
    if not is_anomaly:
        severity = "Normal"
        interpretation = "✅ This lap looks normal for the given conditions."
    elif score < -0.6:
        severity = "Critical"
        interpretation = "🚨 Severe anomaly — likely mechanical issue, damage, or major incident."
    elif score < -0.55:
        severity = "High"
        interpretation = "⚠️ Significant anomaly — investigate for tire issues or strategy mistake."
    else:
        severity = "Medium"
        interpretation = "🟡 Mild anomaly — could be pit stop, traffic, or fresh tyre push lap."
    
    return IncidentResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(float(score), 4),
        severity=severity,
        interpretation=interpretation
    )


@app.post("/optimize/strategy", response_model=StrategyResponse, tags=["ML Predictions"])
def optimize_strategy(request: StrategyRequest):
    """
    Find the optimal pit lap for a 1-stop race strategy by simulating all possible pit windows.
    
    Returns the best pit lap, top 5 strategies, and pit window for safety-car-resilient decisions.
    """
    if request.stint1_compound == request.stint2_compound:
        raise HTTPException(
            status_code=400,
            detail="Stint compounds must differ — F1 rules require 2 different compounds."
        )
    
    strategies = []
    for pit_lap in range(request.pit_lap_min, request.pit_lap_max + 1):
        stint1 = predict_stint_internal(request.driver, request.stint1_compound, stint_length=pit_lap, start_lap=1)
        stint2 = predict_stint_internal(request.driver, request.stint2_compound, stint_length=request.race_length - pit_lap, start_lap=pit_lap + 1)
        
        total = sum(stint1) + request.pit_stop_loss + sum(stint2)
        
        strategies.append({
            "pit_lap": pit_lap,
            "stint1_time": round(sum(stint1), 3),
            "stint2_time": round(sum(stint2), 3),
            "total_race_time": round(total, 3)
        })
    
    strategies.sort(key=lambda x: x['total_race_time'])
    optimal = strategies[0]
    top5 = strategies[:5]
    pit_laps_in_window = sorted([s['pit_lap'] for s in top5])
    
    return StrategyResponse(
        driver=request.driver,
        strategy=f"{request.stint1_compound} → {request.stint2_compound}",
        optimal_pit_lap=optimal['pit_lap'],
        stint1_laps=optimal['pit_lap'],
        stint2_laps=request.race_length - optimal['pit_lap'],
        total_race_time=optimal['total_race_time'],
        pit_window_start=pit_laps_in_window[0],
        pit_window_end=pit_laps_in_window[-1],
        top_5_strategies=top5
    )