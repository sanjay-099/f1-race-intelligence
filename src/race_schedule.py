"""
F1 Race Schedule Helpers
Utility functions for fetching and formatting the F1 season schedule.
"""

import fastf1
import pandas as pd
from pathlib import Path


def get_current_year() -> int:
    from datetime import datetime
    return datetime.now().year


def get_season_schedule(year: int, cache_dir: str = None) -> list:
    """
    Returns the full season schedule as a list of dicts.
    Each dict has: round, name, country, location, date
    """
    try:
        if cache_dir:
            fastf1.Cache.enable_cache(cache_dir)
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = []
        for _, event in schedule.iterrows():
            races.append({
                "round":    int(event['RoundNumber']),
                "name":     str(event['EventName']),
                "country":  str(event['Country']),
                "location": str(event['Location']),
                "date":     str(event['EventDate'])[:10],
            })
        return races
    except Exception as e:
        print(f"Schedule error {year}: {e}")
        return []


def get_latest_completed_race(cache_dir: str = None) -> dict:
    """
    Returns the most recently completed race as a dict with year, round, session_type.
    Falls back to previous year if no completed races found in current year.
    """
    try:
        if cache_dir:
            fastf1.Cache.enable_cache(cache_dir)
        from datetime import datetime
        current_year = datetime.now().year
        schedule     = fastf1.get_event_schedule(current_year, include_testing=False)
        today        = pd.Timestamp.now()
        past_races   = schedule[schedule['EventDate'] < today]

        if len(past_races) == 0:
            schedule   = fastf1.get_event_schedule(current_year - 1, include_testing=False)
            past_races = schedule

        latest = past_races.iloc[-1]
        return {
            "year":         int(latest['EventDate'].year),
            "round":        int(latest['RoundNumber']),
            "session_type": "R",
        }
    except Exception as e:
        print(f"Latest race error: {e}")
        return {"year": 2026, "round": 1, "session_type": "R"}


def get_available_seasons(start_year: int = 2018) -> list:
    """Returns list of available seasons from start_year to current year."""
    from datetime import datetime
    current_year = datetime.now().year
    return list(range(start_year, current_year + 1))


def is_sprint_weekend(year: int, round_number: int, cache_dir: str = None) -> bool:
    """Returns True if the given race weekend includes a sprint session."""
    try:
        if cache_dir:
            fastf1.Cache.enable_cache(cache_dir)
        event = fastf1.get_event(year, round_number)
        for i in range(1, 6):
            session_name = str(event.get(f'Session{i}', ''))
            if 'Sprint' in session_name:
                return True
        return False
    except Exception:
        return False