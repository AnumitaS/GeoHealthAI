import pandas as pd
from pathlib import Path

CRUDE_BIRTH_RATE = 16.9
CRUDE_DEATH_RATE = 6.5
NET_GROWTH_RATE = (CRUDE_BIRTH_RATE - CRUDE_DEATH_RATE) / 1000
BASE_YEAR = 2011
TARGET_YEAR = 2026
YEARS_DELTA = TARGET_YEAR - BASE_YEAR

def calculate_state_population(source: str | pd.DataFrame, state_name: str) -> dict:
    if source is None:
        return {"total_2011": 0, "projected_target": 0, "formatted_str": "Pop Data N/A", "district_pop_map": {}}

    if isinstance(source, pd.DataFrame):
        df = source
    else:
        if not Path(source).exists():
            return {"total_2011": 0, "projected_target": 0, "formatted_str": "Pop Data N/A", "district_pop_map": {}}
        df = pd.read_excel(source)
        df.columns = [str(c).strip().lower() for c in df.columns]

    state_col = [c for c in df.columns if "state" in c or "st_name" in c][0]
    pop_col = [c for c in df.columns if "tot_p" in c or "population" in c or "persons" in c][0]

    target_clean = state_name.strip().lower().replace(" ", "")
    state_df = df[df[state_col].astype(str).str.strip().str.lower().str.replace(" ", "") == target_clean]

    if state_df.empty:
        return {"total_2011": 0, "projected_target": 0, "formatted_str": "Pop Data N/A", "district_pop_map": {}}

    total_2011 = pd.to_numeric(state_df[pop_col], errors="coerce").sum()
    projected_target = total_2011 * ((1 + NET_GROWTH_RATE) ** YEARS_DELTA)

    return {
        "total_2011": int(total_2011),
        "projected_target": int(projected_target),
        "formatted_str": f"{TARGET_YEAR} Projected Pop: {projected_target/1e6:.2f}M (2011: {total_2011/1e6:.2f}M)"
    }