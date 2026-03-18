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
FIXED_FIELD_SIZE = 362  # typical D1 team count


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


def _build_kaggle_to_canonical_map(
    kaggle_root: Path,
    canonical_ids: set[str],
) -> dict[int, str]:
    """Build a mapping from Kaggle TeamID → canonical slug ID.

    Uses the slugified Kaggle team name, supplemented by the historical
    aliases CSV for edge cases.
    """
    from .build_training_dataset import normalize_name, name_variants
    from .common import ALIASES_PATH

    teams_candidates = list(kaggle_root.rglob("MTeams.csv"))
    if not teams_candidates:
        return {}

    teams = pd.read_csv(teams_candidates[0])
    id_map: dict[int, str] = {}

    # Build a lookup of canonical IDs with trailing hyphens stripped for
    # matching against artifacts like "virginia-" → "virginia"
    canonical_stripped = {}
    for cid in canonical_ids:
        stripped = cid.rstrip("-")
        if stripped != cid:
            canonical_stripped[stripped] = cid

    # Pass 1: direct slug match from Kaggle team name
    for _, row in teams.iterrows():
        tid = int(row["TeamID"])
        slug = normalize_name(row["TeamName"])
        if slug in canonical_ids:
            id_map[tid] = slug
        elif slug in canonical_stripped:
            id_map[tid] = canonical_stripped[slug]

    # Pass 2: use aliases CSV to resolve mismatches
    if ALIASES_PATH.exists():
        aliases = pd.read_csv(ALIASES_PATH)
        # Build reverse lookup: alias source_team_name slug → canonical_team_id
        alias_map: dict[str, str] = {}
        for _, arow in aliases.iterrows():
            cid = str(arow["canonical_team_id"])
            source_name = str(arow.get("source_team_name", ""))
            if source_name:
                alias_map[normalize_name(source_name)] = cid

        for _, row in teams.iterrows():
            tid = int(row["TeamID"])
            if tid in id_map:
                continue
            kaggle_name = str(row["TeamName"])
            for variant in name_variants(kaggle_name):
                if variant in alias_map:
                    id_map[tid] = alias_map[variant]
                    break

    # Pass 3: fuzzy match remaining unmatched Kaggle teams
    unmatched_kaggle = {
        int(row["TeamID"]): str(row["TeamName"])
        for _, row in teams.iterrows()
        if int(row["TeamID"]) not in id_map
    }
    unmatched_canonical = canonical_ids - set(id_map.values())
    if unmatched_kaggle and unmatched_canonical:
        from difflib import SequenceMatcher

        canonical_slug_list = sorted(unmatched_canonical)
        for tid, kaggle_name in unmatched_kaggle.items():
            kaggle_slug = normalize_name(kaggle_name)
            best_score = 0.0
            best_cid = None
            for cid in canonical_slug_list:
                score = SequenceMatcher(None, kaggle_slug, cid).ratio()
                if score > best_score:
                    best_score = score
                    best_cid = cid
            if best_cid and best_score >= 0.72:
                id_map[tid] = best_cid

    return id_map


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

    # Collect all canonical IDs used in the dataset
    canonical_ids = set(
        features["canonical_team_id"].dropna().unique()
    ) | set(
        features["opponent_team_id"].dropna().unique()
    )

    id_map = _build_kaggle_to_canonical_map(root, canonical_ids)
    if not id_map:
        features["elo_diff"] = 0.0
        features["elo_rank_percentile_diff"] = 0.0
        features["missing_elo"] = 1
        return features

    # Map Kaggle team_id → canonical slug in Elo DataFrame
    elo_mapped = elo_df.copy()
    elo_mapped["canonical_id"] = elo_mapped["team_id"].map(id_map)
    elo_mapped = elo_mapped.dropna(subset=["canonical_id"])

    # Rank percentile: higher Elo → higher percentile
    elo_mapped["elo_rank_percentile"] = 1.0 - (
        (elo_mapped["elo_rank"] - 1) / (FIXED_FIELD_SIZE - 1)
    )

    # Build lookup for team Elo by (season, canonical_id)
    elo_lookup = elo_mapped.set_index(["season", "canonical_id"])[
        ["elo", "elo_rank_percentile"]
    ]

    # Join team Elo
    team_keys = list(
        zip(features["season"].astype(int), features["canonical_team_id"])
    )
    opp_keys = list(
        zip(features["season"].astype(int), features["opponent_team_id"])
    )

    team_elo = pd.DataFrame(
        [elo_lookup.loc[k].to_dict() if k in elo_lookup.index else {"elo": np.nan, "elo_rank_percentile": np.nan} for k in team_keys]
    )
    opp_elo = pd.DataFrame(
        [elo_lookup.loc[k].to_dict() if k in elo_lookup.index else {"elo": np.nan, "elo_rank_percentile": np.nan} for k in opp_keys]
    )

    has_both = team_elo["elo"].notna() & opp_elo["elo"].notna()

    features["elo_diff"] = np.where(
        has_both.values,
        (team_elo["elo"].values - opp_elo["elo"].values),
        0.0,
    )
    features["elo_rank_percentile_diff"] = np.where(
        has_both.values,
        (team_elo["elo_rank_percentile"].values - opp_elo["elo_rank_percentile"].values),
        0.0,
    )
    features["missing_elo"] = (~has_both).astype(int).values

    return features
