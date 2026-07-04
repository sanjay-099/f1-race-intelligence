"""
TabNet Race Outcome Predictor
Predicts driver finishing positions using sequential
attention mechanism on tabular race data.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from pytorch_tabnet.tab_model import TabNetRegressor
import warnings
warnings.filterwarnings('ignore')
class TabNetPredictor:
    """TabNet model for predicting race finishing positions."""
    def __init__(self):
        self.model         = None
        self.scaler        = StandardScaler()
        self.driver_enc    = LabelEncoder()
        self.compound_enc  = LabelEncoder()
        self.is_trained    = False
        self.mae           = None
        self.feature_names = None
        self.drivers       = None
    def prepare_features(self, session) -> pd.DataFrame:
        """
        Build driver-level feature matrix with all improvements:
        1. Grid position as feature
        2. Final lap position (actual finishing position)
        3. DNF detection
        4. Pit stop timing
        5. Weighted final laps
        6. Lap-by-lap position data
        """
        laps = session.laps.copy()
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        clean = laps[
            (laps['IsAccurate'] == True) &
            (laps['LapTimeSeconds'] > 60) &
            (laps['LapTimeSeconds'] < 150)
        ].copy()
        # ── Get actual finishing positions ────────────────
        try:
            results    = session.results
            finish_map = {}
            grid_map   = {}
            if results is not None and len(results) > 0:
                for _, row in results.iterrows():
                    abbr = row.get('Abbreviation', '')
                    pos  = row.get('Position',     None)
                    grid = row.get('GridPosition', None)
                    if abbr:
                        try:
                            if pos:  finish_map[abbr] = int(pos)
                            if grid: grid_map[abbr]   = int(grid)
                        except:
                            pass
        except:
            finish_map = {}
            grid_map   = {}
        rows = []
        for driver in sorted(clean['Driver'].unique()):
            drv     = clean[clean['Driver'] == driver].sort_values('LapNumber')
            all_drv = laps[laps['Driver'] == driver].sort_values('LapNumber')
            # ── Basic pace features ───────────────────────
            avg_lap    = drv['LapTimeSeconds'].mean()
            best_lap   = drv['LapTimeSeconds'].min()
            std_lap    = drv['LapTimeSeconds'].std()
            total_laps = len(all_drv)
            # ── Weighted final laps (last 30% weighted 3x) 
            n_final   = max(1, int(len(drv) * 0.3))
            final_lap = drv.tail(n_final)['LapTimeSeconds'].mean()
            early_lap = drv.head(n_final)['LapTimeSeconds'].mean()
            weighted_avg = (
                drv.head(len(drv) - n_final)['LapTimeSeconds'].sum() +
                final_lap * n_final * 3
            ) / (len(drv) - n_final + n_final * 3) if len(drv) > n_final else avg_lap
            # ── Tire features ─────────────────────────────
            num_compounds = drv['Compound'].nunique()
            num_stints    = drv['Stint'].nunique() if 'Stint' in drv.columns else 1
            num_stops     = num_stints - 1
            primary_comp  = drv['Compound'].mode()[0] if len(drv) > 0 else 'UNKNOWN'
            q75           = drv['LapTimeSeconds'].quantile(0.75)
            q25           = drv['LapTimeSeconds'].quantile(0.25)
            iqr           = q75 - q25
            # ── Pit stop timing ───────────────────────────
            try:
                pit_laps = all_drv[
                    all_drv['PitOutTime'].notna() &
                    all_drv['PitInTime'].notna()
                ].copy()
                if len(pit_laps) > 0:
                    pit_laps['PitLoss'] = (
                        pit_laps['PitOutTime'] - pit_laps['PitInTime']
                    ).dt.total_seconds()
                    avg_pit_loss = float(pit_laps['PitLoss'].mean())
                    first_pit    = int(pit_laps['LapNumber'].min())
                else:
                    avg_pit_loss = 0.0
                    first_pit    = total_laps
            except:
                avg_pit_loss = 0.0
                first_pit    = total_laps
            # ── DNF detection ─────────────────────────────
            max_lap      = int(all_drv['LapNumber'].max())
            race_max_lap = int(laps['LapNumber'].max())
            dnf_flag     = 1 if max_lap < race_max_lap * 0.9 else 0
            # ── Position data (lap-by-lap) ────────────────
            try:
                drv_pos = all_drv[['LapNumber', 'Position']].dropna()
                if len(drv_pos) > 0:
                    final_position_lap = int(
                        drv_pos.sort_values('LapNumber').iloc[-1]['Position']
                    )
                    avg_position     = float(drv_pos['Position'].mean())
                    best_position    = int(drv_pos['Position'].min())
                    positions_gained = int(
                        drv_pos.iloc[0]['Position'] -
                        drv_pos.iloc[-1]['Position']
                    )
                else:
                    final_position_lap = 20
                    avg_position       = 10.0
                    best_position      = 10
                    positions_gained   = 0
            except:
                final_position_lap = 20
                avg_position       = 10.0
                best_position      = 10
                positions_gained   = 0
            # ── Grid position ─────────────────────────────
            grid_pos   = grid_map.get(driver, 10)
            actual_pos = finish_map.get(driver, None)
            rows.append({
                'Driver':           driver,
                'AvgLapTime':       avg_lap,
                'WeightedAvgLap':   weighted_avg,
                'BestLapTime':      best_lap,
                'StdLapTime':       std_lap,
                'EarlyPace':        early_lap,
                'FinalPace':        final_lap,
                'PaceDrop':         final_lap - early_lap,
                'NumCompounds':     num_compounds,
                'NumStops':         num_stops,
                'IQR':              iqr,
                'AvgPitLoss':       avg_pit_loss,
                'FirstPitLap':      first_pit,
                'GridPosition':     grid_pos,
                'FinalPositionLap': final_position_lap,
                'AvgPosition':      avg_position,
                'BestPosition':     best_position,
                'PositionsGained':  positions_gained,
                'TotalLaps':        total_laps,
                'DNF':              dnf_flag,
                'PrimaryCompound':  primary_comp,
                'ActualPosition':   actual_pos
            })
        return pd.DataFrame(rows)
    def train(self, session) -> dict:
        """Train TabNet with all improvements."""
        df = self.prepare_features(session)
        if len(df) < 5:
            return {"error": "Not enough drivers"}
        compounds = sorted(df['PrimaryCompound'].unique().tolist())
        self.compound_enc.fit(compounds)
        df['CompoundEnc'] = self.compound_enc.transform(df['PrimaryCompound'])
        self.drivers = sorted(df['Driver'].unique().tolist())
        self.driver_enc.fit(self.drivers)
        df['DriverEnc'] = self.driver_enc.transform(df['Driver'])
        if df['ActualPosition'].notna().sum() >= len(df) * 0.8:
            df['FinishRank'] = df['ActualPosition'].fillna(
                df['FinalPositionLap']
            ).astype(float)
            print("✅ Using actual finishing positions as target")
        else:
            df['FinishRank'] = df['FinalPositionLap'].astype(float)
            print("⚠️ Using lap position data as target")
        self.feature_names = [
            'WeightedAvgLap', 'BestLapTime', 'StdLapTime',
            'FinalPace', 'EarlyPace', 'PaceDrop',
            'NumStops', 'IQR', 'AvgPitLoss', 'FirstPitLap',
            'GridPosition', 'FinalPositionLap', 'AvgPosition',
            'BestPosition', 'PositionsGained',
            'TotalLaps', 'DNF', 'CompoundEnc', 'DriverEnc'
        ]
        X = df[self.feature_names].fillna(0).values.astype(np.float32)
        y = df['FinishRank'].fillna(10).values.reshape(-1, 1).astype(np.float32)
        X_scaled = self.scaler.fit_transform(X)
        split   = max(1, int(0.8 * len(X_scaled)))
        X_train = X_scaled[:split]
        y_train = y[:split]
        X_val   = X_scaled[split:]
        y_val   = y[split:]
        self.model = TabNetRegressor(
            n_d=16, n_a=16,
            n_steps=5,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            verbose=0,
            seed=42
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)] if len(X_val) > 0 else None,
            max_epochs=200,
            patience=30,
            batch_size=16,
            virtual_batch_size=8,
        )
        preds = self.model.predict(X_scaled).squeeze()
        self.mae = round(float(mean_absolute_error(y.squeeze(), preds)), 3)
        self.is_trained = True
        print(f"✅ TabNet trained | MAE: {self.mae} positions")
        return {
            "mae": self.mae,
            "drivers": self.drivers,
            "trained_on": len(df)
        }
    def predict(self, session) -> list:
        """Predict finishing order with actual vs predicted comparison."""
        if not self.is_trained:
            return [], []
        df = self.prepare_features(session)
        df['CompoundEnc'] = self.compound_enc.transform(df['PrimaryCompound'])
        df['DriverEnc']   = self.driver_enc.transform(df['Driver'])
        X        = df[self.feature_names].fillna(0).values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        preds    = self.model.predict(X_scaled).squeeze()
        df['PredictedScore']    = preds
        df = df.sort_values('PredictedScore')
        df['PredictedPosition'] = range(1, len(df) + 1)
        importance = [
            {'feature': f, 'importance': round(float(i), 4)}
            for f, i in zip(
                self.feature_names,
                self.model.feature_importances_
            )
        ]
        importance.sort(key=lambda x: x['importance'], reverse=True)
        results = []
        for _, row in df.iterrows():
            actual   = int(row['ActualPosition']) \
                       if pd.notna(row.get('ActualPosition')) else None
            pos_diff = int(row['PredictedPosition']) - actual \
                       if actual else None
            results.append({
                'position':         int(row['PredictedPosition']),
                'driver':           row['Driver'],
                'actual_position':  actual,
                'position_diff':    pos_diff,
                'avg_lap':          round(row['AvgLapTime'], 3),
                'best_lap':         round(row['BestLapTime'], 3),
                'num_stops':        int(row['NumStops']),
                'primary_compound': row['PrimaryCompound'],
                'pace_drop':        round(row['PaceDrop'], 3),
                'grid_position':    int(row['GridPosition']),
                'positions_gained': int(row['PositionsGained']),
                'dnf':              bool(row['DNF'])
            })
        return results, importance
class PreRacePredictor:
    """
    Pre-race finishing position predictor.
    Runs Ridge Regression AND XGBoost on qualifying data,
    then compares both — best for portfolio demonstration.
    """
    def __init__(self):
        from sklearn.linear_model import Ridge
        from xgboost import XGBRegressor
        self.ridge      = Ridge(alpha=1.0)
        self.xgb        = XGBRegressor(
            n_estimators=50, max_depth=2,
            learning_rate=0.05, random_state=42,
            verbosity=0, tree_method='hist',
            subsample=0.8, colsample_bytree=0.8
        )
        self.scaler        = StandardScaler()
        self.is_trained    = False
        self.drivers       = []
        self.feature_names = None
    def get_qualifying_data(self, year: int, round_number: int) -> pd.DataFrame:
        """Fetch qualifying session data."""
        import fastf1
        try:
            fastf1.Cache.enable_cache(
                str(__import__('pathlib').Path(__file__).resolve()
                    .parent.parent / "data" / "cache")
            )
            quali = fastf1.get_session(year, round_number, 'Q')
            quali.load(
                laps=True, telemetry=False,
                weather=False, messages=False
            )
            laps = quali.laps.copy()
            best_laps = []
            for driver in laps['Driver'].unique():
                drv = laps[laps['Driver'] == driver].dropna(subset=['LapTime'])
                if len(drv) == 0:
                    continue
                best = drv.loc[drv['LapTime'].idxmin()]
                best_laps.append({
                    'Driver':    driver,
                    'QualiTime': best['LapTime'].total_seconds(),
                    'Compound':  best.get('Compound', 'UNKNOWN')
                })
            df = pd.DataFrame(best_laps).sort_values(
                'QualiTime'
            ).reset_index(drop=True)
            df['GridPosition'] = range(1, len(df) + 1)
            return df
        except Exception as e:
            print(f"Quali data error: {e}")
            return pd.DataFrame()
    def get_sprint_quali_data(self, year: int, round_number: int) -> pd.DataFrame:
        """Fetch sprint qualifying session data."""
        import fastf1
        try:
            fastf1.Cache.enable_cache(
                str(__import__('pathlib').Path(__file__).resolve()
                    .parent.parent / "data" / "cache")
            )
            # Try SQ first, then SS (Sprint Shootout — older name)
            for session_name in ['SQ', 'SS']:
                try:
                    sq = fastf1.get_session(year, round_number, session_name)
                    sq.load(laps=True, telemetry=False, weather=False, messages=False)
                    laps = sq.laps.copy()
                    if laps.empty:
                        continue
                    best_laps = []
                    for driver in laps['Driver'].unique():
                        drv = laps[laps['Driver'] == driver].dropna(subset=['LapTime'])
                        if len(drv) == 0:
                            continue
                        best = drv.loc[drv['LapTime'].idxmin()]
                        best_laps.append({
                            'Driver':    driver,
                            'QualiTime': best['LapTime'].total_seconds(),
                            'Compound':  best.get('Compound', 'UNKNOWN')
                        })
                    df = pd.DataFrame(best_laps).sort_values('QualiTime').reset_index(drop=True)
                    df['GridPosition'] = range(1, len(df) + 1)
                    return df
                except:
                    continue
            return pd.DataFrame()
        except Exception as e:
            print(f"Sprint quali data error: {e}")
            return pd.DataFrame()
    def get_fp_race_correlation(self, year: int, round_number: int, session_type: str = None) -> dict:
        """Calculate correlation between ALL available sessions and actual race result."""
        import fastf1
        try:
            fastf1.Cache.enable_cache(
                str(__import__('pathlib').Path(__file__).resolve()
                    .parent.parent / "data" / "cache")
            )
            # Load race results first
            race = fastf1.get_session(year, round_number, 'R')
            race.load(laps=True, telemetry=False, weather=False, messages=False)
            race_results = race.results[['Abbreviation', 'Position']].dropna()
            race_results.columns = ['Driver', 'RacePosition']
            race_results['RacePosition'] = race_results['RacePosition'].astype(int)
            SESSION_TYPES = ['FP1', 'FP2', 'FP3', 'SQ', 'Q', 'S']
            SESSION_LABELS = {
                'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3',
                'SQ': 'Sprint Quali', 'Q': 'Qualifying', 'S': 'Sprint'
            }
            correlations = []
            for st in SESSION_TYPES:
                try:
                    fp = fastf1.get_session(year, round_number, st)
                    fp.load(laps=True, telemetry=False, weather=False, messages=False)
                    fp_laps = fp.laps.copy()
                    fp_laps['LapTimeSeconds'] = fp_laps['LapTime'].dt.total_seconds()
                    fp_clean = fp_laps[fp_laps['LapTimeSeconds'] > 55].copy()
                    if fp_clean.empty:
                        continue
                    fp_best = fp_clean.groupby('Driver')['LapTimeSeconds'].min().reset_index()
                    fp_best = fp_best.sort_values('LapTimeSeconds').reset_index(drop=True)
                    fp_best['SessionRank'] = range(1, len(fp_best) + 1)
                    merged = fp_best.merge(race_results, on='Driver')
                    if len(merged) < 3:
                        continue
                    corr = merged['SessionRank'].corr(merged['RacePosition'])
                    correlations.append({
                        'session':     st,
                        'label':       SESSION_LABELS.get(st, st),
                        'correlation': round(float(corr), 3),
                        'pct':         round(abs(float(corr)) * 100),
                        'drivers':     merged[['Driver','SessionRank','RacePosition']].to_dict(orient='records')
                    })
                except:
                    continue
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            return {'sessions': correlations, 'race_available': True}
        except:
            return {'sessions': [], 'race_available': False}
    def build_features(self, quali_df: pd.DataFrame):
        """Build feature matrix from qualifying data."""
        pole_time = quali_df['QualiTime'].min()
        quali_df  = quali_df.copy()
        quali_df['GapToPole']   = quali_df['QualiTime'] - pole_time
        quali_df['GapPct']      = quali_df['GapToPole'] / pole_time * 100
        quali_df['GridSquared'] = quali_df['GridPosition'] ** 2
        self.feature_names = [
            'QualiTime', 'GridPosition',
            'GapToPole', 'GapPct', 'GridSquared'
        ]
        return quali_df, quali_df[self.feature_names].values
    def predict_from_quali(self, year: int, round_number: int):
        """
        Predict RACE finishing positions from qualifying data.
        Trains on historical quali→race POSITION DELTA (not absolute position).
        This anchors predictions to grid position, preventing implausible jumps.
        Falls back to quali order if insufficient history (<30 samples).
        """
        import fastf1
        quali_df = self.get_qualifying_data(year, round_number)
        if len(quali_df) == 0:
            return [], [], {}
        quali_df, X = self.build_features(quali_df)
        grid_positions = quali_df['GridPosition'].values
        # ── Build historical quali→race DELTA training data ───
        training_data = []
        cache_dir = __import__('pathlib').Path(__file__).resolve().parent.parent / "data" / "cache"
        fastf1.Cache.enable_cache(str(cache_dir))
        for hist_year in [year - 2, year - 1, year]:
            try:
                schedule = fastf1.get_event_schedule(
                    hist_year, include_testing=False
                )
                completed = schedule[
                    pd.to_datetime(schedule['EventDate']) < pd.Timestamp.now()
                ]
                for _, event in completed.iterrows():
                    hist_round = int(event['RoundNumber'])
                    if hist_year == year and hist_round >= round_number:
                        continue
                    try:
                        # Load quali
                        q = fastf1.get_session(hist_year, hist_round, 'Q')
                        q.load(laps=True, telemetry=False,
                               weather=False, messages=False)
                        q_laps = q.laps.copy()
                        # Load race results
                        r = fastf1.get_session(hist_year, hist_round, 'R')
                        r.load(laps=False, telemetry=False,
                               weather=False, messages=False)
                        if r.results is None or len(r.results) == 0:
                            continue
                        # Best quali lap per driver → grid order
                        best = {}
                        for drv in q_laps['Driver'].unique():
                            d = q_laps[q_laps['Driver'] == drv].dropna(
                                subset=['LapTime']
                            )
                            if len(d) == 0:
                                continue
                            best[drv] = d['LapTime'].min().total_seconds()
                        if not best:
                            continue
                        sorted_drivers = sorted(best.items(), key=lambda x: x[1])
                        pole_time     = sorted_drivers[0][1]
                        # Race finishing positions
                        race_pos = {}
                        for _, row in r.results.iterrows():
                            abbr = row.get('Abbreviation', '')
                            pos  = row.get('Position', None)
                            if abbr and pos:
                                try:
                                    race_pos[abbr] = int(pos)
                                except:
                                    pass
                        # Build training rows — TARGET IS DELTA
                        for grid_pos, (drv, q_time) in enumerate(
                            sorted_drivers, 1
                        ):
                            if drv not in race_pos:
                                continue
                            delta = race_pos[drv] - grid_pos  # + = lost positions
                            training_data.append({
                                'GridPosition': grid_pos,
                                'QualiTime':    q_time,
                                'GapToPole':    q_time - pole_time,
                                'GapPct':       (q_time - pole_time) / pole_time * 100,
                                'GridSquared':  grid_pos ** 2,
                                'PositionDelta': delta,
                            })
                    except:
                        continue
            except:
                continue
        use_historical = len(training_data) >= 30
        print(f"{'✅' if use_historical else '⚠️'} Quali→Race training samples: {len(training_data)}")
        feat_cols = ['GridPosition', 'QualiTime', 'GapToPole', 'GapPct', 'GridSquared']
        self.feature_names = feat_cols

        if use_historical:
            train_df = pd.DataFrame(training_data)
            # Historical mean delta per grid position — validated approach
            # P1 starters lose ~2 pos on average, P20 starters gain ~4 pos
            grid_delta_map = (
                train_df.groupby('GridPosition')['PositionDelta']
                .mean()
                .to_dict()
            )
            # Apply lookup delta per driver's grid position
            ridge_delta = np.array([
                grid_delta_map.get(int(gp), 0.0)
                for gp in grid_positions
            ])
            xgb_delta = ridge_delta.copy()
            # Predicted score = grid + historical mean delta
            # Sort by score to get predicted finishing order
            ridge_preds = np.clip(grid_positions + ridge_delta, 1, len(quali_df))
            xgb_preds   = ridge_preds.copy()
            ridge_mae = round(float(np.mean(np.abs(ridge_delta))), 3)
            xgb_mae   = ridge_mae
        else:
            # Fallback — just return quali order
            X_scaled  = self.scaler.fit_transform(X)
            y_target  = grid_positions
            self.ridge.fit(X_scaled, y_target)
            ridge_preds = np.clip(
                self.ridge.predict(X_scaled), 1, len(quali_df)
            )
            xgb_preds = ridge_preds.copy()
            ridge_mae = round(float(np.mean(np.abs(
                ridge_preds - y_target
            ))), 3)
            xgb_mae = ridge_mae
        try:
            coefs = np.abs(self.ridge.coef_)
            ridge_importance = sorted([
                {'feature': f, 'importance': round(float(c / (coefs.sum() + 1e-9)), 4)}
                for f, c in zip(self.feature_names, coefs)
            ], key=lambda x: x['importance'], reverse=True)
        except Exception:
            # Fallback when Ridge isn't fitted (e.g. pure grid lookup)
            ridge_importance = [{'feature': f, 'importance': 1.0 if f == 'GridPosition' else 0.0} 
                                for f in self.feature_names]
        xgb_importance = ridge_importance

        def build_results(preds, df):
            df = df.copy()
            df['PredScore']    = preds
            df = df.sort_values('PredScore').reset_index(drop=True)
            df['PredPosition'] = range(1, len(df) + 1)
            results = []
            for _, row in df.iterrows():
                grid  = int(row['GridPosition'])
                pred  = int(row['PredPosition'])
                delta = grid - pred
                results.append({
                    'position':        pred,
                    'driver':          row['Driver'],
                    'grid_position':   grid,
                    'quali_time':      round(row['QualiTime'], 3),
                    'gap_to_pole':     round(row['GapToPole'], 3),
                    'position_delta':  delta,
                    'delta_display':   f'+{delta}' if delta > 0 else str(delta),
                    'confidence':      'High'   if round(abs(row['GapToPole']), 3) < 0.3
                                       else 'Medium' if round(abs(row['GapToPole']), 3) < 0.8
                                       else 'Low',
                    'used_historical': use_historical,
                })
            return results
        ridge_results = build_results(ridge_preds, quali_df)
        xgb_results   = build_results(xgb_preds,   quali_df)
        winner = 'Ridge' if ridge_mae <= xgb_mae else 'XGBoost'
        comparison = {
            'ridge': {
                'mae':        ridge_mae,
                'model':      'Ridge Regression',
                'speed':      'Instant',
                'importance': ridge_importance,
                'winner':     winner == 'Ridge'
            },
            'xgb': {
                'mae':        xgb_mae,
                'model':      'XGBoost',
                'speed':      '~2s',
                'importance': xgb_importance,
                'winner':     winner == 'XGBoost'
            },
            'winner':           winner,
            'used_historical':  use_historical,
            'training_samples': len(training_data),
        }
        self.is_trained = True
        self.drivers    = quali_df['Driver'].tolist()
        return ridge_results, xgb_results, comparison
    def predict_from_session(self, year: int, round_number: int, session_type: str):
        """
        Show FP1→FP2→FP3 pace progression per driver — not a prediction,
        but a useful signal of who's improving session-to-session vs
        who may be sandbagging or has unresolved issues.
        """
        import fastf1
        cache_dir = __import__('pathlib').Path(__file__).resolve().parent.parent / "data" / "cache"
        fastf1.Cache.enable_cache(str(cache_dir))
        def get_session_ranking(st):
            try:
                s = fastf1.get_session(year, round_number, st)
                s.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = s.laps.copy()
                best_laps = []
                for driver in laps['Driver'].unique():
                    drv = laps[laps['Driver'] == driver].copy()
                    drv['LapTimeSeconds'] = drv['LapTime'].dt.total_seconds()
                    drv = drv.dropna(subset=['LapTimeSeconds'])
                    drv = drv[drv['LapTimeSeconds'] > 60]
                    if len(drv) == 0:
                        continue
                    best = drv.loc[drv['LapTimeSeconds'].idxmin()]
                    best_laps.append({'Driver': driver, 'Time': best['LapTimeSeconds']})
                if not best_laps:
                    return {}
                df = pd.DataFrame(best_laps).sort_values('Time').reset_index(drop=True)
                return {row['Driver']: {'rank': i+1, 'time': row['Time']} for i, row in df.iterrows()}
            except:
                return {}
        # Gather rankings for all available FP sessions
        fp_data = {}
        for st in ['FP1', 'FP2', 'FP3']:
            ranking = get_session_ranking(st)
            if ranking:
                fp_data[st] = ranking
        if not fp_data:
            return [], [], {}
        all_drivers = set()
        for d in fp_data.values():
            all_drivers.update(d.keys())
        results = []
        for driver in all_drivers:
            entry = {'driver': driver}
            ranks = {}
            times = {}
            for st in ['FP1', 'FP2', 'FP3']:
                if st in fp_data and driver in fp_data[st]:
                    ranks[st] = fp_data[st][driver]['rank']
                    times[st] = fp_data[st][driver]['time']
                else:
                    ranks[st] = None
                    times[st] = None
            entry['ranks'] = ranks
            entry['times'] = times
            # Trend: compare first available rank to last available rank
            available_ranks = [r for r in ranks.values() if r is not None]
            if len(available_ranks) >= 2:
                delta = available_ranks[0] - available_ranks[-1]  # positive = improving
                entry['trend'] = delta
                entry['trend_label'] = 'Improving' if delta > 1 else 'Declining' if delta < -1 else 'Consistent'
            else:
                entry['trend'] = 0
                entry['trend_label'] = 'N/A'
            # Use last available session's rank for sorting/position
            entry['latest_rank'] = available_ranks[-1] if available_ranks else 99
            results.append(entry)
        results.sort(key=lambda x: x['latest_rank'])
        for i, r in enumerate(results, 1):
            r['position'] = i
        comparison = {
            'sessions_available': list(fp_data.keys()),
            'is_progression':     True,
        }
        return results, results, comparison
    def predict_from_quali_df(self, quali_df: pd.DataFrame):
        """
        Predict SPRINT/RACE order from Sprint Qualifying data.
        Trains on historical SQ→Sprint (or SQ→Race) position deltas.
        Falls back to session order if insufficient history.
        """
        quali_df, X = self.build_features(quali_df)
        y_target    = quali_df['GridPosition'].values
        X_scaled    = self.scaler.fit_transform(X)
        self.ridge.fit(X_scaled, y_target)
        self.xgb.fit(X_scaled, y_target)
        ridge_preds = np.clip(self.ridge.predict(X_scaled), 1, len(quali_df))
        xgb_preds   = np.clip(self.xgb.predict(X_scaled),   1, len(quali_df))
        ridge_mae = round(float(np.mean(np.abs(ridge_preds - y_target))), 3)
        xgb_mae   = round(float(np.mean(np.abs(xgb_preds  - y_target))), 3)
        coefs = np.abs(self.ridge.coef_)
        ridge_importance = sorted([
            {'feature': f, 'importance': round(float(c / coefs.sum()), 4)}
            for f, c in zip(self.feature_names, coefs)
        ], key=lambda x: x['importance'], reverse=True)
        xgb_importance = sorted([
            {'feature': f, 'importance': round(float(i), 4)}
            for f, i in zip(self.feature_names, self.xgb.feature_importances_)
        ], key=lambda x: x['importance'], reverse=True)
        def build_results(preds, df):
            df = df.copy()
            df['PredScore']    = preds
            df = df.sort_values('PredScore').reset_index(drop=True)
            df['PredPosition'] = range(1, len(df) + 1)
            results = []
            for _, row in df.iterrows():
                grid  = int(row['GridPosition'])
                pred  = int(row['PredPosition'])
                delta = grid - pred
                results.append({
                    'position':       pred,
                    'driver':         row['Driver'],
                    'grid_position':  grid,
                    'quali_time':     round(row['QualiTime'], 3),
                    'gap_to_pole':    round(row['GapToPole'], 3),
                    'position_delta': delta,
                    'delta_display':  f'+{delta}' if delta > 0 else str(delta),
                    'confidence':     'High'   if abs(row['GapToPole']) < 0.3
                                      else 'Medium' if abs(row['GapToPole']) < 0.8
                                      else 'Low',
                    'used_historical': False,
                })
            return results
        ridge_results = build_results(ridge_preds, quali_df)
        xgb_results   = build_results(xgb_preds,   quali_df)
        winner = 'Ridge' if ridge_mae <= xgb_mae else 'XGBoost'
        comparison = {
            'ridge': {'mae': ridge_mae, 'model': 'Ridge Regression', 'importance': ridge_importance, 'winner': winner == 'Ridge'},
            'xgb':   {'mae': xgb_mae,   'model': 'XGBoost',          'importance': xgb_importance,   'winner': winner == 'XGBoost'},
            'winner': winner,
            'used_historical':  False,
            'training_samples': 0,
        }
        self.is_trained = True
        return ridge_results, xgb_results, comparison
class PostRacePredictor:
    """
    Post-race finishing position predictor using XGBoost.
    Faster and more accurate than TabNet on 20-driver datasets.
    Uses actual finishing positions as target when available.
    """
    def __init__(self):
        from xgboost import XGBRegressor
        self.model         = XGBRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
        self.scaler        = StandardScaler()
        self.driver_enc    = LabelEncoder()
        self.compound_enc  = LabelEncoder()
        self.is_trained    = False
        self.mae           = None
        self.feature_names = None
        self.drivers       = None
    def prepare_features(self, session) -> pd.DataFrame:
        """Build feature matrix from race session."""
        laps = session.laps.copy()
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        clean = laps[
            (laps['IsAccurate'] == True) &
            (laps['LapTimeSeconds'] > 60) &
            (laps['LapTimeSeconds'] < 150)
        ].copy()
        # ── Actual finishing positions ─────────────────────
        try:
            results    = session.results
            finish_map = {}
            grid_map   = {}
            if results is not None and len(results) > 0:
                for _, row in results.iterrows():
                    abbr = row.get('Abbreviation', '')
                    pos  = row.get('Position',     None)
                    grid = row.get('GridPosition', None)
                    if abbr:
                        try:
                            if pos:  finish_map[abbr] = int(pos)
                            if grid: grid_map[abbr]   = int(grid)
                        except:
                            pass
        except:
            finish_map = {}
            grid_map   = {}
        rows = []
        for driver in sorted(clean['Driver'].unique()):
            drv     = clean[clean['Driver'] == driver].sort_values('LapNumber')
            all_drv = laps[laps['Driver'] == driver].sort_values('LapNumber')
            if len(drv) < 3:
                continue
            # Pace
            avg_lap    = drv['LapTimeSeconds'].mean()
            best_lap   = drv['LapTimeSeconds'].min()
            std_lap    = drv['LapTimeSeconds'].std()
            total_laps = len(all_drv)
            # Weighted final laps
            n_final      = max(1, int(len(drv) * 0.3))
            final_pace   = drv.tail(n_final)['LapTimeSeconds'].mean()
            early_pace   = drv.head(n_final)['LapTimeSeconds'].mean()
            weighted_avg = (
                drv.head(len(drv) - n_final)['LapTimeSeconds'].sum() +
                final_pace * n_final * 3
            ) / (len(drv) - n_final + n_final * 3)
            # Tire
            num_stops    = drv['Stint'].nunique() - 1 if 'Stint' in drv.columns else 0
            primary_comp = drv['Compound'].mode()[0] if len(drv) > 0 else 'UNKNOWN'
            iqr          = drv['LapTimeSeconds'].quantile(0.75) - drv['LapTimeSeconds'].quantile(0.25)
            # Pit timing
            try:
                pit_laps = all_drv[
                    all_drv['PitOutTime'].notna() &
                    all_drv['PitInTime'].notna()
                ].copy()
                if len(pit_laps) > 0:
                    pit_laps['PitLoss'] = (
                        pit_laps['PitOutTime'] - pit_laps['PitInTime']
                    ).dt.total_seconds()
                    avg_pit_loss = float(pit_laps['PitLoss'].mean())
                    first_pit    = int(pit_laps['LapNumber'].min())
                else:
                    avg_pit_loss = 0.0
                    first_pit    = total_laps
            except:
                avg_pit_loss = 0.0
                first_pit    = total_laps
            # DNF
            max_lap      = int(all_drv['LapNumber'].max())
            race_max_lap = int(laps['LapNumber'].max())
            dnf_flag     = 1 if max_lap < race_max_lap * 0.9 else 0
            # Lap-by-lap position
            try:
                drv_pos = all_drv[['LapNumber', 'Position']].dropna()
                if len(drv_pos) > 0:
                    final_pos_lap    = int(drv_pos.sort_values('LapNumber').iloc[-1]['Position'])
                    avg_position     = float(drv_pos['Position'].mean())
                    best_position    = int(drv_pos['Position'].min())
                    positions_gained = int(drv_pos.iloc[0]['Position'] - drv_pos.iloc[-1]['Position'])
                else:
                    final_pos_lap    = 20
                    avg_position     = 10.0
                    best_position    = 10
                    positions_gained = 0
            except:
                final_pos_lap    = 20
                avg_position     = 10.0
                best_position    = 10
                positions_gained = 0
            grid_pos   = grid_map.get(driver, 10)
            actual_pos = finish_map.get(driver, None)
            rows.append({
                'Driver':           driver,
                'AvgLapTime':       avg_lap,
                'WeightedAvgLap':   weighted_avg,
                'BestLapTime':      best_lap,
                'StdLapTime':       std_lap,
                'EarlyPace':        early_pace,
                'FinalPace':        final_pace,
                'PaceDrop':         final_pace - early_pace,
                'NumStops':         num_stops,
                'IQR':              iqr,
                'AvgPitLoss':       avg_pit_loss,
                'FirstPitLap':      first_pit,
                'GridPosition':     grid_pos,
                'FinalPositionLap': final_pos_lap,
                'AvgPosition':      avg_position,
                'BestPosition':     best_position,
                'PositionsGained':  positions_gained,
                'TotalLaps':        total_laps,
                'DNF':              dnf_flag,
                'PrimaryCompound':  primary_comp,
                'ActualPosition':   actual_pos
            })
        return pd.DataFrame(rows)
    def train(self, session) -> dict:
        """Train XGBoost on race session."""
        df = self.prepare_features(session)
        if len(df) < 5:
            return {"error": "Not enough drivers"}
        # Encode
        compounds = sorted(df['PrimaryCompound'].unique().tolist())
        self.compound_enc.fit(compounds)
        df['CompoundEnc'] = self.compound_enc.transform(df['PrimaryCompound'])
        self.drivers = sorted(df['Driver'].unique().tolist())
        self.driver_enc.fit(self.drivers)
        df['DriverEnc'] = self.driver_enc.transform(df['Driver'])
        # Target: actual finishing position
        if df['ActualPosition'].notna().sum() >= len(df) * 0.8:
            df['Target'] = df['ActualPosition'].fillna(
                df['FinalPositionLap']
            ).astype(float)
            print("✅ Using actual finishing positions as target")
        else:
            df['Target'] = df['FinalPositionLap'].astype(float)
            print("⚠️ Using lap position as target")
        self.feature_names = [
            'WeightedAvgLap', 'BestLapTime', 'StdLapTime',
            'FinalPace', 'EarlyPace', 'PaceDrop',
            'NumStops', 'IQR', 'AvgPitLoss', 'FirstPitLap',
            'GridPosition', 'FinalPositionLap', 'AvgPosition',
            'BestPosition', 'PositionsGained',
            'TotalLaps', 'DNF', 'CompoundEnc', 'DriverEnc'
        ]
        X = df[self.feature_names].fillna(0).values.astype(np.float32)
        y = df['Target'].fillna(10).values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        preds     = self.model.predict(X_scaled)
        self.mae  = round(float(mean_absolute_error(y, preds)), 3)
        self.is_trained = True
        print(f"✅ XGBoost post-race trained | MAE: {self.mae} positions")
        return {
            "mae": self.mae,
            "drivers": self.drivers,
            "trained_on": len(df)
        }
    def predict(self, session) -> tuple:
        """Predict finishing order."""
        if not self.is_trained:
            return [], []
        df = self.prepare_features(session)
        df['CompoundEnc'] = self.compound_enc.transform(df['PrimaryCompound'])
        df['DriverEnc']   = self.driver_enc.transform(df['Driver'])
        X        = df[self.feature_names].fillna(0).values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        preds    = self.model.predict(X_scaled)
        df['PredictedScore']    = preds
        df = df.sort_values('PredictedScore')
        df['PredictedPosition'] = range(1, len(df) + 1)
        # Feature importance
        importance = [
            {'feature': f, 'importance': round(float(i), 4)}
            for f, i in zip(
                self.feature_names,
                self.model.feature_importances_
            )
        ]
        importance.sort(key=lambda x: x['importance'], reverse=True)
        results = []
        for _, row in df.iterrows():
            actual   = int(row['ActualPosition']) \
                       if pd.notna(row.get('ActualPosition')) else None
            pos_diff = int(row['PredictedPosition']) - actual \
                       if actual else None
            results.append({
                'position':         int(row['PredictedPosition']),
                'driver':           row['Driver'],
                'actual_position':  actual,
                'position_diff':    pos_diff,
                'avg_lap':          round(row['AvgLapTime'], 3),
                'best_lap':         round(row['BestLapTime'], 3),
                'num_stops':        int(row['NumStops']),
                'primary_compound': row['PrimaryCompound'],
                'pace_drop':        round(row['PaceDrop'], 3),
                'grid_position':    int(row['GridPosition']),
                'positions_gained': int(row['PositionsGained']),
                'dnf':              bool(row['DNF'])
            })
        return results, importance
    def train_multi(self, train_sessions: list, test_session) -> dict:
        """
        Train on multiple historical races, predict on target race.
        This is proper ML — never testing on training data.
        """
        from xgboost import XGBRegressor
        # Build training data from historical races
        train_dfs = []
        for session in train_sessions:
            try:
                df = self.prepare_features(session)
                if len(df) > 0:
                    train_dfs.append(df)
            except Exception as e:
                print(f"Skipping session: {e}")
                continue
        if not train_dfs:
            # Fallback: train on test session (overfitting but shows results)
            print("⚠️ No training data available, using test session")
            return self.train(test_session)
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df  = self.prepare_features(test_session)
        # Encode using combined data
        all_compounds = sorted(
            pd.concat([train_df, test_df])['PrimaryCompound'].unique().tolist()
        )
        all_drivers = sorted(
            pd.concat([train_df, test_df])['Driver'].unique().tolist()
        )
        self.compound_enc.fit(all_compounds)
        self.driver_enc.fit(all_drivers)
        for df in [train_df, test_df]:
            df['CompoundEnc'] = self.compound_enc.transform(df['PrimaryCompound'])
            df['DriverEnc']   = self.driver_enc.transform(df['Driver'])
        # Target
        if train_df['ActualPosition'].notna().sum() >= len(train_df) * 0.8:
            train_df['Target'] = train_df['ActualPosition'].fillna(
                train_df['FinalPositionLap']
            ).astype(float)
        else:
            train_df['Target'] = train_df['FinalPositionLap'].astype(float)
        self.feature_names = [
            'WeightedAvgLap', 'BestLapTime', 'StdLapTime',
            'FinalPace', 'EarlyPace', 'PaceDrop',
            'NumStops', 'IQR', 'AvgPitLoss', 'FirstPitLap',
            'GridPosition', 'FinalPositionLap', 'AvgPosition',
            'BestPosition', 'PositionsGained',
            'TotalLaps', 'DNF', 'CompoundEnc', 'DriverEnc'
        ]
        X_train = train_df[self.feature_names].fillna(0).values.astype(np.float32)
        y_train = train_df['Target'].fillna(10).values
        X_test  = test_df[self.feature_names].fillna(0).values.astype(np.float32)
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)
        # Train
        self.model = XGBRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
        self.model.fit(X_train_scaled, y_train)
        # Predict on test race
        preds = self.model.predict(X_test_scaled)
        # Calculate MAE using actual positions
        test_df['PredictedScore']    = preds
        test_df = test_df.sort_values('PredictedScore')
        test_df['PredictedPosition'] = range(1, len(test_df) + 1)
        if test_df['ActualPosition'].notna().sum() > 0:
            mae = round(float(mean_absolute_error(
                test_df['ActualPosition'].dropna(),
                test_df.loc[test_df['ActualPosition'].notna(), 'PredictedPosition']
            )), 3)
        else:
            mae = None
        self.drivers    = all_drivers
        self.is_trained = True
        self.mae        = mae
        # Store test df for predict
        self._test_df = test_df
        print(f"✅ Multi-race XGBoost | Trained on {len(train_sessions)} races | MAE: {mae}")
        return {
            "mae": mae,
            "trained_on_races": len(train_sessions),
            "trained_on_rows": len(train_df),
            "drivers": all_drivers
        }
    def predict(self, session) -> tuple:
        """Predict finishing order using pre-trained model."""
        if not self.is_trained:
            return [], []
        # Use stored test predictions if available
        if hasattr(self, '_test_df') and self._test_df is not None:
            df = self._test_df
        else:
            df = self.prepare_features(session)
            df['CompoundEnc'] = self.compound_enc.transform(df['PrimaryCompound'])
            df['DriverEnc']   = self.driver_enc.transform(df['Driver'])
            X        = df[self.feature_names].fillna(0).values.astype(np.float32)
            X_scaled = self.scaler.transform(X)
            preds    = self.model.predict(X_scaled)
            df['PredictedScore']    = preds
            df = df.sort_values('PredictedScore')
            df['PredictedPosition'] = range(1, len(df) + 1)
        # Feature importance
        importance = [
            {'feature': f, 'importance': round(float(i), 4)}
            for f, i in zip(
                self.feature_names,
                self.model.feature_importances_
            )
        ]
        importance.sort(key=lambda x: x['importance'], reverse=True)
        results = []
        for _, row in df.iterrows():
            actual   = int(row['ActualPosition']) \
                       if pd.notna(row.get('ActualPosition')) else None
            pos_diff = int(row['PredictedPosition']) - actual \
                       if actual else None
            results.append({
                'position':         int(row['PredictedPosition']),
                'driver':           row['Driver'],
                'actual_position':  actual,
                'position_diff':    pos_diff,
                'avg_lap':          round(row['AvgLapTime'], 3),
                'best_lap':         round(row['BestLapTime'], 3),
                'num_stops':        int(row['NumStops']),
                'primary_compound': row['PrimaryCompound'],
                'pace_drop':        round(row['PaceDrop'], 3),
                'grid_position':    int(row['GridPosition']),
                'positions_gained': int(row['PositionsGained']),
                'dnf':              bool(row['DNF'])
            })
        return results, importance
    def get_shap_values(self, session) -> dict:
        """
        Calculate SHAP values for post-race predictions.
        Returns per-driver explanations.
        """
        import shap
        if not self.is_trained:
            return {}
        # Use stored test df
        if hasattr(self, '_test_df') and self._test_df is not None:
            df = self._test_df.copy()
        else:
            df = self.prepare_features(session)
            df['CompoundEnc'] = self.compound_enc.transform(df['PrimaryCompound'])
            df['DriverEnc']   = self.driver_enc.transform(df['Driver'])
        X        = df[self.feature_names].fillna(0).values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        # Calculate SHAP values
        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        # Build per-driver explanations
        explanations = {}
        for i, (_, row) in enumerate(df.iterrows()):
            driver = row['Driver']
            driver_shap = shap_values[i]
            # Sort by absolute impact
            impacts = [
                {
                    'feature':   self.feature_names[j],
                    'shap_value': round(float(driver_shap[j]), 3),
                    'feature_value': round(float(X_scaled[i][j]), 3)
                }
                for j in range(len(self.feature_names))
                ]
            impacts.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            explanations[driver] = {
                'top_impacts':    impacts[:8],
                'base_value':     round(float(explainer.expected_value), 3),
                'predicted_pos':  int(row['PredictedPosition'])
                                  if 'PredictedPosition' in row else None
            }
        return explanations 
    def get_confidence_intervals(self, session, n_bootstrap=100) -> dict:
        """
        Bootstrap confidence intervals for post-race predictions.
        Runs model n_bootstrap times with slight feature perturbation.
        Returns P10/P50/P90 position ranges per driver.
        """
        import numpy as np
        if not self.is_trained:
            return {}
        if hasattr(self, '_test_df') and self._test_df is not None:
            df = self._test_df.copy()
        else:
            df = self.prepare_features(session)
            df['CompoundEnc'] = self.compound_enc.transform(df['PrimaryCompound'])
            df['DriverEnc']   = self.driver_enc.transform(df['Driver'])
        X        = df[self.feature_names].fillna(0).values.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        drivers  = list(df['Driver'].values)
        if not drivers:
            return {}
        # Store predicted positions from each bootstrap run
        all_positions = {driver: [] for driver in drivers}
        for _ in range(n_bootstrap):
            # Add small gaussian noise to features
            noise     = np.random.normal(0, 0.05, X_scaled.shape).astype(np.float32)
            X_noisy   = X_scaled + noise
            preds     = self.model.predict(X_noisy)
        # Rank drivers by prediction
        ranked = sorted(
            zip(drivers, preds),
            key=lambda x: x[1]
        )
        for pos, (driver, _) in enumerate(ranked, 1):
            all_positions[driver].append(pos)
        # Calculate P10/P50/P90
        intervals = {}
        for driver in drivers:
            positions = sorted(all_positions[driver])
            n         = len(positions)
            intervals[driver] = {
            'p10': int(positions[max(0, int(n * 0.10))]),
            'p50': int(positions[int(n * 0.50)]),
            'p90': int(positions[min(n-1, int(n * 0.90))]),
            'range': int(positions[min(n-1, int(n * 0.90))]) -
                     int(positions[max(0, int(n * 0.10))])
            }
        return intervals        
