"""Collect historical Barttorvik ratings for training.

Barttorvik (barttorvik.com) provides freely accessible adjusted efficiency
metrics, Barthag, and SOS. This data is already used at runtime from the
current-season snapshot but is NOT in the historical training pipeline.

This collector scrapes archived Barttorvik data by season.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .common import PROCESSED_ROOT, RAW_ROOT, ensure_dirs, fetch_html, load_season_config

BARTTORVIK_URL = "https://barttorvik.com/trank.php"


def collect_barttorvik_season(season: int) -> pd.DataFrame | None:
    """Scrape Barttorvik team ratings for a given season.

    Returns a DataFrame with columns:
      season, source_team_name, kenpom_badj_em, barthag, adjoe, adjde, adjt
    or None if the page cannot be parsed.
    """
    url = f"{BARTTORVIK_URL}?year={season}&conlimit=All"
    try:
        html = fetch_html(url)
        tables = pd.read_html(html)
        if not tables:
            return None
    except Exception:
        return None

    df = tables[0]
    if df.empty:
        return None

    # Barttorvik table columns vary by year; find the right ones
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in ("team", "team name"):
            col_map["source_team_name"] = col
        elif col_lower in ("barthag",):
            col_map["barthag"] = col
        elif col_lower in ("adjoe", "adj oe"):
            col_map["adjoe"] = col
        elif col_lower in ("adjde", "adj de"):
            col_map["adjde"] = col
        elif col_lower in ("adjt", "adj t", "adj tempo"):
            col_map["adjt"] = col
        elif col_lower in ("adjem", "adj em"):
            col_map["kenpom_badj_em"] = col

    if "source_team_name" not in col_map:
        return None

    result = pd.DataFrame({"season": season}, index=df.index)
    result["source_team_name"] = df[col_map["source_team_name"]].astype(str)

    for target, source_col in col_map.items():
        if target == "source_team_name":
            continue
        result[target] = pd.to_numeric(df[source_col], errors="coerce")

    if "kenpom_badj_em" not in result.columns and "adjoe" in result.columns and "adjde" in result.columns:
        result["kenpom_badj_em"] = result["adjoe"] - result["adjde"]

    return result


def collect_all_seasons() -> pd.DataFrame:
    """Collect Barttorvik ratings for all configured training seasons."""
    ensure_dirs()
    config = load_season_config()
    frames = []

    for season in config.training_seasons:
        print(f"  Barttorvik {season}...", end=" ", flush=True)
        df = collect_barttorvik_season(season)
        if df is not None:
            frames.append(df)
            print(f"{len(df)} teams")
        else:
            print("skipped")
        time.sleep(2)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    output_path = RAW_ROOT / "barttorvik_ratings.parquet"
    combined.to_parquet(output_path, index=False)
    print(f"Wrote {len(combined)} rows to {output_path}")
    return combined


if __name__ == "__main__":
    collect_all_seasons()
