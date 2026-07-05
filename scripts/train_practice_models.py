import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

CACHE_DIR = Path("data/cache")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

print("Collecting practice/qualifying laps...")
all_laps = []

for year in [2024, 2025]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        rnd = int(event['RoundNumber'])
        for session_type in ['FP1', 'FP2', 'FP3', 'Q']:
            try:
                s = fastf1.get_session(year, rnd, session_type)
                s.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = s.laps.copy()
                if laps.empty:
                    continue
                laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
                laps = laps[laps['LapTimeSeconds'] > 60]
                laps = laps.dropna(subset=['LapTimeSeconds'])
                all_laps.append(laps[['LapTimeSeconds','TyreLife','LapNumber']])
                print(f"  OK {year} R{rnd} {session_type}: {len(laps)} laps")
            except Exception as e:
                print(f"  SKIP {year} R{rnd} {session_type}: {e}")
                continue

df = pd.concat(all_laps, ignore_index=True).dropna()
print(f"Total laps: {len(df)}")

X = df[['LapTimeSeconds','TyreLife','LapNumber']].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_scaled)

joblib.dump(iso, MODELS_DIR / "incident_isolation_forest_practice.pkl")
joblib.dump(scaler, MODELS_DIR / "incident_scaler_practice.pkl")
print("DONE: saved incident_isolation_forest_practice.pkl + incident_scaler_practice.pkl")
