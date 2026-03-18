"""Collect historical Sagarin ratings for training.

Sagarin ratings are published on usatoday.com and are one of the oldest
computer rating systems for college basketball. The ratings provide an
independent quality signal from a different methodology than KenPom/Torvik.

This module provides a collector interface. Sagarin data is available
historically via web archives and can be imported from local CSV files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import EXTERNAL_ROOT, PROCESSED_ROOT, ensure_dirs


def load_sagarin_from_csv(csv_path: Path | None = None) -> pd.DataFrame | None:
    """Load Sagarin ratings from a local CSV file.

    Expected columns: season, team_name, sagarin_rating, sagarin_rank
    """
    if csv_path is None:
        csv_path = EXTERNAL_ROOT / "sagarin_ratings.csv"

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    required = {"season", "team_name"}
    if not required.issubset(set(df.columns)):
        return None

    return df


def merge_sagarin_into_snapshots(
    snapshots: pd.DataFrame,
    sagarin: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Sagarin ratings into the season snapshots DataFrame.

    Adds sagarin_rating and sagarin_rank columns when available.
    """
    if sagarin is None or sagarin.empty:
        snapshots["sagarin_rating"] = None
        snapshots["sagarin_rank"] = None
        return snapshots

    merged = snapshots.merge(
        sagarin[["season", "team_name", "sagarin_rating", "sagarin_rank"]].rename(
            columns={"team_name": "source_team_name"}
        ),
        on=["season", "source_team_name"],
        how="left",
    )
    return merged
