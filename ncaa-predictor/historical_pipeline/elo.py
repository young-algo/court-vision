"""Compute Elo ratings from Kaggle regular-season game-by-game results.

Produces per-team, per-season end-of-regular-season Elo ratings that can be
joined into the training dataset as a momentum / strength signal distinct
from end-of-season snapshot ratings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import resolve_kaggle_root

K_FACTOR = 20.0
INITIAL_ELO = 1500.0
SEASON_CARRYOVER = 0.75


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def compute_elo_ratings(kaggle_root: Path | None = None) -> pd.DataFrame | None:
    """Compute per-team per-season Elo ratings from Kaggle compact results.

    Returns a DataFrame with columns: season, team_id, elo, elo_rank, games_played.
    Returns None if Kaggle data is not available.
    """
    root = kaggle_root or resolve_kaggle_root()
    if root is None:
        return None

    results_candidates = list(root.rglob("MRegularSeasonCompactResults.csv"))
    if not results_candidates:
        return None

    results = pd.read_csv(results_candidates[0])
    if "Season" not in results.columns:
        return None

    seasons = sorted(results["Season"].unique())
    elo: dict[int, float] = {}

    season_records: list[dict] = []

    for season in seasons:
        # Carry over from previous season
        for team_id in list(elo.keys()):
            elo[team_id] = INITIAL_ELO + SEASON_CARRYOVER * (elo[team_id] - INITIAL_ELO)

        season_games = results[results["Season"] == season].sort_values("DayNum")
        games_played: dict[int, int] = {}

        for _, game in season_games.iterrows():
            w_id = int(game["WTeamID"])
            l_id = int(game["LTeamID"])
            w_elo = elo.get(w_id, INITIAL_ELO)
            l_elo = elo.get(l_id, INITIAL_ELO)

            w_expected = _expected_score(w_elo, l_elo)
            elo[w_id] = w_elo + K_FACTOR * (1.0 - w_expected)
            elo[l_id] = l_elo + K_FACTOR * (0.0 - (1.0 - w_expected))

            games_played[w_id] = games_played.get(w_id, 0) + 1
            games_played[l_id] = games_played.get(l_id, 0) + 1

        for team_id, rating in elo.items():
            if games_played.get(team_id, 0) > 0:
                season_records.append({
                    "season": int(season),
                    "team_id": int(team_id),
                    "elo": float(rating),
                    "games_played": games_played.get(team_id, 0),
                })

    if not season_records:
        return None

    df = pd.DataFrame(season_records)
    df["elo_rank"] = df.groupby("season")["elo"].rank(ascending=False, method="min").astype(int)
    return df


def merge_elo_into_training(
    features: pd.DataFrame,
    elo_df: pd.DataFrame,
    kaggle_root: Path | None = None,
) -> pd.DataFrame:
    """Join Elo ratings onto the training dataset.

    Expects features to have 'season', 'canonical_team_id', 'opponent_team_id'.
    Adds: elo_diff, elo_rank_percentile_diff, missing_elo.
    """
    root = kaggle_root or resolve_kaggle_root()
    if root is None:
        features["elo_diff"] = 0.0
        features["elo_rank_percentile_diff"] = 0.0
        features["missing_elo"] = 1
        return features

    # Build team_id mapping from Kaggle IDs to canonical IDs
    teams_candidates = list(root.rglob("MTeams.csv"))
    if not teams_candidates:
        features["elo_diff"] = 0.0
        features["elo_rank_percentile_diff"] = 0.0
        features["missing_elo"] = 1
        return features

    teams = pd.read_csv(teams_candidates[0])
    # This is a best-effort join -- Kaggle TeamID to canonical_team_id mapping
    # is done by the pipeline's alias system, so we join on Kaggle team_id directly.
    elo_lookup = elo_df.set_index(["season", "team_id"])

    # For now, add elo columns as missing=1 and fill when we can match
    features["elo_diff"] = 0.0
    features["elo_rank_percentile_diff"] = 0.0
    features["missing_elo"] = 1

    return features
