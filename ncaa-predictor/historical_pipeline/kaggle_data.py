from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import load_season_config, resolve_kaggle_root, slugify_team

KAGGLE_FILE_CANDIDATES = {
    "teams": ["MTeams.csv"],
    "team_spellings": ["MTeamSpellings.csv", "MTeamSpellings.csv.zip"],
    "seasons": ["MSeasons.csv"],
    "tourney_results": ["MNCAATourneyCompactResults.csv"],
    "tourney_seeds": ["MNCAATourneySeeds.csv"],
    "massey_ordinals": ["MMasseyOrdinals.csv"],
}

CURATED_SYSTEMS = {
    "BPI",
    "DOK",
    "MAS",
    "MOR",
    "POM",
    "RTH",
    "SAG",
    "TRP",
    "WOL",
    "WLK",
}


def resolve_file(key: str, kaggle_root: Path | None = None) -> Path | None:
    root = kaggle_root or resolve_kaggle_root()
    if root is None:
        return None

    for candidate in KAGGLE_FILE_CANDIDATES[key]:
        matches = list(root.rglob(candidate))
        if matches:
            return matches[0]
    return None


def has_kaggle_bundle(kaggle_root: Path | None = None) -> bool:
    return resolve_file("teams", kaggle_root) is not None


def read_csv(key: str, kaggle_root: Path | None = None) -> pd.DataFrame:
    file_path = resolve_file(key, kaggle_root)
    if file_path is None:
        raise FileNotFoundError(f"Missing Kaggle file for key '{key}'")
    return pd.read_csv(file_path)


def kaggle_aliases_frame(kaggle_root: Path | None = None) -> pd.DataFrame:
    teams = read_csv("teams", kaggle_root)
    spellings = read_csv("team_spellings", kaggle_root)
    teams = teams.rename(columns={"TeamID": "team_id", "TeamName": "team_name"})
    spellings = spellings.rename(
        columns={"TeamID": "team_id", "TeamNameSpelling": "source_team_name"}
    )

    merged = spellings.merge(teams, on="team_id", how="left")
    merged["canonical_team_id"] = merged["team_name"].map(slugify_team)
    merged["canonical_team_name"] = merged["team_name"]
    merged["source"] = "kaggle_team_spellings"
    merged["match_method"] = "kaggle_spelling"
    merged["needs_review"] = False
    return merged[
        [
            "canonical_team_id",
            "canonical_team_name",
            "source",
            "source_team_name",
            "match_method",
            "needs_review",
        ]
    ].dropna(subset=["source_team_name", "canonical_team_id"])


def selection_day_numbers(kaggle_root: Path | None = None) -> dict[int, int]:
    seasons = read_csv("seasons", kaggle_root)
    config = load_season_config()
    seasons["DayZero"] = pd.to_datetime(seasons["DayZero"])

    day_numbers: dict[int, int] = {}
    for season in config.training_seasons:
        row = seasons[seasons["Season"] == season]
        if row.empty:
            continue
        day_zero = row.iloc[0]["DayZero"]
        selection_day = pd.Timestamp(config.selection_sunday_dates[str(season)])
        day_numbers[season] = int((selection_day - day_zero).days)
    return day_numbers


def kaggle_tournament_games(kaggle_root: Path | None = None) -> pd.DataFrame:
    results = read_csv("tourney_results", kaggle_root)
    teams = read_csv("teams", kaggle_root).rename(
        columns={"TeamID": "team_id", "TeamName": "team_name"}
    )
    seeds = read_csv("tourney_seeds", kaggle_root).rename(columns={"Seed": "seed"})
    seeds["seed_num"] = seeds["seed"].str.extract(r"(\d+)").astype(int)

    winners = results.merge(
        teams.rename(columns={"team_id": "WTeamID", "team_name": "winner_name"}),
        on="WTeamID",
        how="left",
    ).merge(
        teams.rename(columns={"team_id": "LTeamID", "team_name": "loser_name"}),
        on="LTeamID",
        how="left",
    ).merge(
        seeds.rename(columns={"Season": "Season", "TeamID": "WTeamID", "seed_num": "winner_seed"}),
        on=["Season", "WTeamID"],
        how="left",
    ).merge(
        seeds.rename(columns={"Season": "Season", "TeamID": "LTeamID", "seed_num": "loser_seed"}),
        on=["Season", "LTeamID"],
        how="left",
    )

    winners["round"] = None
    for season, season_rows in winners.groupby("Season"):
        unique_days = sorted(season_rows["DayNum"].unique())
        if len(unique_days) >= 12:
            unique_days = unique_days[2:]
        elif len(unique_days) > 10:
            unique_days = unique_days[-10:]

        round_map = {}
        if len(unique_days) >= 10:
            round_map.update({unique_days[0]: "round_of_64", unique_days[1]: "round_of_64"})
            round_map.update({unique_days[2]: "round_of_32", unique_days[3]: "round_of_32"})
            round_map.update({unique_days[4]: "sweet_16", unique_days[5]: "sweet_16"})
            round_map.update({unique_days[6]: "elite_8", unique_days[7]: "elite_8"})
            round_map[unique_days[8]] = "final_four"
            round_map[unique_days[9]] = "championship"

        winners.loc[winners["Season"] == season, "round"] = winners.loc[
            winners["Season"] == season, "DayNum"
        ].map(round_map)

    winners = winners[winners["round"].notna()].copy()
    winners["season"] = winners["Season"]
    winners["game_date"] = winners["DayNum"]
    winners["team_name"] = winners["winner_name"]
    winners["opponent_name"] = winners["loser_name"]
    winners["seed"] = winners["winner_seed"]
    winners["opponent_seed"] = winners["loser_seed"]
    winners["team_score"] = winners["WScore"]
    winners["opponent_score"] = winners["LScore"]
    winners["neutral_site"] = True
    winners["region"] = None
    winners["game_id"] = (
        winners["Season"].astype(str)
        + "-"
        + winners["DayNum"].astype(str)
        + "-"
        + winners["WTeamID"].astype(str)
        + "-"
        + winners["LTeamID"].astype(str)
    )
    winners["source_url"] = "kaggle://MNCAATourneyCompactResults.csv"
    return winners[
        [
            "season",
            "game_date",
            "round",
            "region",
            "neutral_site",
            "team_name",
            "opponent_name",
            "seed",
            "opponent_seed",
            "team_score",
            "opponent_score",
            "game_id",
            "source_url",
        ]
    ]


def kaggle_massey_snapshots(kaggle_root: Path | None = None) -> pd.DataFrame:
    ordinals = read_csv("massey_ordinals", kaggle_root)
    teams = read_csv("teams", kaggle_root).rename(
        columns={"TeamID": "TeamID", "TeamName": "source_team_name"}
    )
    day_numbers = selection_day_numbers(kaggle_root)
    ordinals = ordinals[ordinals["Season"].isin(day_numbers.keys())].copy()
    ordinals["selection_day_num"] = ordinals["Season"].map(day_numbers)
    ordinals = ordinals[ordinals["RankingDayNum"] <= ordinals["selection_day_num"]]
    ordinals["system_priority"] = ordinals["SystemName"].isin(CURATED_SYSTEMS).astype(int)
    latest = (
        ordinals.sort_values(
            ["Season", "TeamID", "system_priority", "RankingDayNum"],
            ascending=[True, True, False, False],
        )
        .groupby(["Season", "TeamID", "SystemName"], as_index=False)
        .tail(1)
    )

    aggregated = (
        latest.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            ordinal_rank=("OrdinalRank", "mean"),
            source_system_count=("SystemName", "nunique"),
        )
    )
    aggregated["source_value"] = 400 - aggregated["ordinal_rank"]
    aggregated["source_rank"] = aggregated["ordinal_rank"]
    aggregated = aggregated.merge(teams, on="TeamID", how="left")
    aggregated["season"] = aggregated["Season"]
    config = load_season_config()
    aggregated["snapshot_date"] = aggregated["season"].map(
        lambda season: config.selection_sunday_dates[str(season)]
    )
    aggregated["source_key"] = "massey"
    aggregated["source"] = "kaggle_massey_ordinals"
    aggregated["source_url"] = "kaggle://MMasseyOrdinals.csv"
    aggregated["massey_ordinal_rank"] = pd.to_numeric(
        aggregated["ordinal_rank"],
        errors="coerce",
    )
    aggregated["massey_system_count"] = pd.to_numeric(
        aggregated["source_system_count"],
        errors="coerce",
    )
    return aggregated[
        [
            "season",
            "source_team_name",
            "source",
            "source_url",
            "source_key",
            "source_rank",
            "source_value",
            "snapshot_date",
            "source_system_count",
            "massey_ordinal_rank",
            "massey_system_count",
        ]
    ]
