"""Collect historical Sagarin ratings for training.

Sagarin ratings are one of the oldest computer rating systems for college
basketball. They provide an independent quality signal from a different
methodology than KenPom/Torvik.

This module extracts Sagarin rankings from the Kaggle Massey ordinals
dataset (system name "SAG"), which has coverage from 2003-2023. It can
also load from a local CSV file as a fallback.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import EXTERNAL_ROOT, PROCESSED_ROOT, ensure_dirs, resolve_kaggle_root

SAGARIN_SYSTEM_NAME = "SAG"


def extract_sagarin_from_kaggle(kaggle_root: Path | None = None) -> pd.DataFrame | None:
    """Extract end-of-regular-season Sagarin rankings from Kaggle Massey ordinals.

    For each season, takes the ranking from the latest available day
    (highest RankingDayNum), which approximates the pre-tournament snapshot.

    Returns DataFrame with columns: season, team_id, sagarin_rank.
    Returns None if Kaggle data is not available.
    """
    root = kaggle_root or resolve_kaggle_root()
    if root is None:
        return None

    ordinals_candidates = list(root.rglob("MMasseyOrdinals.csv"))
    if not ordinals_candidates:
        return None

    ordinals = pd.read_csv(ordinals_candidates[0])
    sag = ordinals[ordinals["SystemName"] == SAGARIN_SYSTEM_NAME].copy()
    if sag.empty:
        return None

    # Take the latest ranking day per season for each team (pre-tournament snapshot)
    sag = sag.sort_values("RankingDayNum")
    latest = sag.groupby(["Season", "TeamID"]).last().reset_index()

    result = pd.DataFrame({
        "season": latest["Season"].astype(int),
        "team_id": latest["TeamID"].astype(int),
        "sagarin_rank": latest["OrdinalRank"].astype(int),
    })

    return result


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


def merge_sagarin_into_training(
    features: pd.DataFrame,
    sagarin_df: pd.DataFrame,
    kaggle_root: Path | None = None,
) -> pd.DataFrame:
    """Merge Sagarin rankings into the training features.

    Uses the same Kaggle TeamID → canonical slug mapping as the Elo module.
    Adds: sagarin_rank_percentile_diff, missing_sagarin.
    """
    from .elo import _build_kaggle_to_canonical_map

    FIXED_FIELD_SIZE = 362

    root = kaggle_root or resolve_kaggle_root()
    if root is None:
        features["sagarin_rank_percentile_diff"] = 0.0
        features["missing_sagarin"] = 1
        return features

    canonical_ids = set(
        features["canonical_team_id"].dropna().unique()
    ) | set(
        features["opponent_team_id"].dropna().unique()
    )

    id_map = _build_kaggle_to_canonical_map(root, canonical_ids)
    if not id_map:
        features["sagarin_rank_percentile_diff"] = 0.0
        features["missing_sagarin"] = 1
        return features

    # Map Kaggle team_id → canonical slug
    sag = sagarin_df.copy()
    sag["canonical_id"] = sag["team_id"].map(id_map)
    sag = sag.dropna(subset=["canonical_id"])

    # Convert rank to percentile (lower rank = higher percentile)
    sag["sagarin_rank_percentile"] = 1.0 - (
        (sag["sagarin_rank"] - 1) / (FIXED_FIELD_SIZE - 1)
    )

    # Build lookup
    sag_lookup = sag.set_index(["season", "canonical_id"])["sagarin_rank_percentile"]

    team_keys = list(
        zip(features["season"].astype(int), features["canonical_team_id"])
    )
    opp_keys = list(
        zip(features["season"].astype(int), features["opponent_team_id"])
    )

    team_pct = pd.Series(
        [sag_lookup.get(k, np.nan) for k in team_keys],
        index=features.index,
    )
    opp_pct = pd.Series(
        [sag_lookup.get(k, np.nan) for k in opp_keys],
        index=features.index,
    )

    has_both = team_pct.notna() & opp_pct.notna()
    features["sagarin_rank_percentile_diff"] = np.where(
        has_both, team_pct - opp_pct, 0.0
    )
    features["missing_sagarin"] = (~has_both).astype(int)

    return features
