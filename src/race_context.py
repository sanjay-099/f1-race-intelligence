"""
F1 Race Context Builder
Builds a structured text context from a FastF1 session
for use as LLM analyst input.
"""

import pandas as pd
import numpy as np


def build_race_context(session) -> str:
    """
    Build a compact, structured race context string from a FastF1 session.
    Used as context for the Multi-Agent LLM Analyst.
    """
    try:
        lines = []
        event = session.event

        # ── Header ──────────────────────────────────────────
        lines.append(f"RACE: {event['EventName']} {event['EventDate'].year}")
        lines.append(f"CIRCUIT: {event['Location']}, {event['Country']}")
        lines.append(f"SESSION: {session.name}")
        lines.append("")

        laps = session.laps.copy()
        if laps.empty:
            return "\n".join(lines) + "\nNo lap data available."

        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        valid = laps[
            (laps['LapTimeSeconds'] > 60) &
            (laps['LapTimeSeconds'] < 200) &
            (laps['IsAccurate'] == True)
        ].copy()

        # ── Race summary ─────────────────────────────────────
        total_laps = int(laps['LapNumber'].max()) if len(laps) > 0 else 0
        drivers    = laps['Driver'].unique().tolist()
        compounds  = laps['Compound'].dropna().unique().tolist()

        lines.append(f"TOTAL LAPS: {total_laps}")
        lines.append(f"DRIVERS: {len(drivers)}")
        lines.append(f"COMPOUNDS USED: {', '.join(compounds)}")
        lines.append("")

        # ── Pace ranking ─────────────────────────────────────
        if len(valid) > 0:
            lines.append("PACE RANKING (avg lap time, clean laps only):")
            pace = (
                valid.groupby('Driver')['LapTimeSeconds']
                .agg(['mean', 'min', 'count'])
                .sort_values('mean')
                .reset_index()
            )
            pace.columns = ['Driver', 'AvgLap', 'BestLap', 'Laps']
            pole_avg = pace['AvgLap'].iloc[0]
            for i, row in pace.head(10).iterrows():
                gap = row['AvgLap'] - pole_avg
                gap_str = f"+{gap:.3f}s" if gap > 0 else "LEADER"
                lines.append(
                    f"  P{i+1:2d} {row['Driver']:4s}  avg={row['AvgLap']:.3f}s  "
                    f"best={row['BestLap']:.3f}s  gap={gap_str}  laps={int(row['Laps'])}"
                )
            lines.append("")

        # ── Tyre strategy ────────────────────────────────────
        if 'Compound' in laps.columns:
            lines.append("TYRE STRATEGY:")
            for driver in sorted(drivers):
                drv_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
                stints   = []
                current_compound = None
                stint_start      = None
                for _, lap in drv_laps.iterrows():
                    compound = lap.get('Compound', 'UNKNOWN')
                    lap_num  = lap.get('LapNumber', 0)
                    if compound != current_compound:
                        if current_compound is not None:
                            stints.append(f"{current_compound}({int(lap_num - stint_start)})")
                        current_compound = compound
                        stint_start      = lap_num
                if current_compound and stint_start is not None:
                    last_lap = drv_laps['LapNumber'].max()
                    stints.append(f"{current_compound}({int(last_lap - stint_start + 1)})")
                if stints:
                    lines.append(f"  {driver:4s}: {' → '.join(stints)}")
            lines.append("")

        # ── Pit stops ────────────────────────────────────────
        if 'PitOutTime' in laps.columns:
            lines.append("PIT STOPS:")
            for driver in sorted(drivers):
                drv_laps = laps[laps['Driver'] == driver]
                pit_laps = drv_laps[drv_laps['PitOutTime'].notna()]['LapNumber'].tolist()
                if pit_laps:
                    lines.append(f"  {driver:4s}: laps {pit_laps}")
            lines.append("")

        # ── Fastest laps ─────────────────────────────────────
        if len(valid) > 0:
            fastest = (
                valid.groupby('Driver')['LapTimeSeconds']
                .min()
                .sort_values()
                .head(5)
            )
            lines.append("FASTEST LAPS:")
            for driver, lap_time in fastest.items():
                mins = int(lap_time // 60)
                secs = lap_time % 60
                lines.append(f"  {driver:4s}: {mins}:{secs:06.3f}")
            lines.append("")

        # ── Race results ─────────────────────────────────────
        try:
            if session.results is not None and len(session.results) > 0:
                lines.append("RACE RESULTS:")
                for _, row in session.results.sort_values('Position').head(10).iterrows():
                    pos    = row.get('Position', '?')
                    driver = row.get('Abbreviation', '?')
                    status = row.get('Status', '')
                    points = row.get('Points', 0)
                    lines.append(f"  P{int(pos):2d} {driver:4s}  status={status}  pts={points}")
                lines.append("")
        except Exception:
            pass

        return "\n".join(lines)

    except Exception as e:
        return f"Race context error: {e}"