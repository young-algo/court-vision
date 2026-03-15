from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import load_season_config, resolve_archive_root, slugify_team

ARCHIVE_FILES = {
    "tournament_matchups": "Tournament Matchups.csv",
    "kenpom_barttorvik": "KenPom Barttorvik.csv",
    "teamsheet_ranks": "Teamsheet Ranks.csv",
    "evanmiya": "EvanMiya.csv",
    "fivethirtyeight": "538 Ratings.csv",
}

ROUND_MAP = {
    64: "round_of_64",
    32: "round_of_32",
    16: "sweet_16",
    8: "elite_8",
    4: "final_four",
    2: "championship",
}

ROUND_DAY_OFFSETS = {
    64: 4,
    32: 6,
    16: 11,
    8: 13,
    4: 18,
    2: 20,
}


def resolve_file(key: str, archive_root: Path | None = None) -> Path | None:
    root = archive_root or resolve_archive_root()
    if root is None:
        return None
    path = root / ARCHIVE_FILES[key]
    return path if path.exists() else None


def has_archive_bundle(archive_root: Path | None = None) -> bool:
    return resolve_file("tournament_matchups", archive_root) is not None


def read_csv(key: str, archive_root: Path | None = None) -> pd.DataFrame:
    path = resolve_file(key, archive_root)
    if path is None:
        raise FileNotFoundError(f"Missing archive file for key '{key}'")
    return pd.read_csv(path)


def archive_aliases_frame(archive_root: Path | None = None) -> pd.DataFrame:
    aliases: list[dict] = []
    for key in ("tournament_matchups", "kenpom_barttorvik", "teamsheet_ranks", "evanmiya", "fivethirtyeight"):
        frame = read_csv(key, archive_root)
        if "TEAM" not in frame.columns:
            continue
        for team_name in frame["TEAM"].dropna().astype(str).unique():
            aliases.append(
                {
                    "canonical_team_id": slugify_team(team_name),
                    "canonical_team_name": team_name,
                    "source": f"archive_{key}",
                    "source_team_name": team_name,
                    "match_method": "archive_team_name",
                    "needs_review": False,
                }
            )
    return pd.DataFrame(aliases).drop_duplicates(subset=["source_team_name"])


def archive_tournament_games(archive_root: Path | None = None) -> pd.DataFrame:
    frame = read_csv("tournament_matchups", archive_root).copy()
    config = load_season_config()
    valid_seasons = set(config.training_seasons)
    frame = frame[frame["YEAR"].isin(valid_seasons)].copy()
    rows: list[dict] = []

    for (season, current_round), group in frame.groupby(["YEAR", "CURRENT ROUND"], sort=True):
        if current_round not in ROUND_MAP:
            continue
        ordered = group.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)
        for index in range(0, len(ordered), 2):
            if index + 1 >= len(ordered):
                continue
            left = ordered.iloc[index]
            right = ordered.iloc[index + 1]
            winner = left if left["SCORE"] >= right["SCORE"] else right
            loser = right if left["SCORE"] >= right["SCORE"] else left
            selection_date = pd.Timestamp(config.selection_sunday_dates[str(int(season))])
            rows.append(
                {
                    "season": int(season),
                    "game_date": (selection_date + pd.Timedelta(days=ROUND_DAY_OFFSETS[current_round])).date().isoformat(),
                    "round": ROUND_MAP[current_round],
                    "region": None,
                    "neutral_site": True,
                    "team_name": winner["TEAM"],
                    "opponent_name": loser["TEAM"],
                    "seed": pd.to_numeric(winner["SEED"], errors="coerce"),
                    "opponent_seed": pd.to_numeric(loser["SEED"], errors="coerce"),
                    "team_score": pd.to_numeric(winner["SCORE"], errors="coerce"),
                    "opponent_score": pd.to_numeric(loser["SCORE"], errors="coerce"),
                    "game_id": f"{season}-{current_round}-{winner['TEAM NO']}-{loser['TEAM NO']}",
                    "source_url": "archive://Tournament Matchups.csv",
                }
            )

    return pd.DataFrame(rows)


def archive_snapshots(archive_root: Path | None = None) -> pd.DataFrame:
    config = load_season_config()
    valid_seasons = set(config.training_seasons)
    team_sheet = read_csv("teamsheet_ranks", archive_root).copy()
    kenpom = read_csv("kenpom_barttorvik", archive_root).copy()
    fte = read_csv("fivethirtyeight", archive_root).copy()
    evanmiya = read_csv("evanmiya", archive_root).copy()
    team_sheet = team_sheet[team_sheet["YEAR"].isin(valid_seasons)].copy()
    kenpom = kenpom[kenpom["YEAR"].isin(valid_seasons)].copy()
    fte = fte[fte["YEAR"].isin(valid_seasons)].copy()
    evanmiya = evanmiya[evanmiya["YEAR"].isin(valid_seasons)].copy()

    snapshots: list[pd.DataFrame] = []

    if not team_sheet.empty:
        bpi = team_sheet[
            [
                "YEAR",
                "TEAM",
                "BPI",
                "NET",
                "RESUME AVG RANK",
                "QUALITY AVG RANK",
                "KPI",
                "SOR",
                "WAB RANK",
                "TRK",
                "KP",
            ]
        ].copy()
        bpi["season"] = bpi["YEAR"].astype(int)
        bpi["source_team_name"] = bpi["TEAM"]
        bpi["source_key"] = "bpi"
        bpi["source_rank"] = pd.to_numeric(bpi["BPI"], errors="coerce")
        bpi["source_value"] = 400 - bpi["source_rank"]
        bpi["bpi_rank"] = pd.to_numeric(bpi["BPI"], errors="coerce")
        bpi["net_rank"] = pd.to_numeric(bpi["NET"], errors="coerce")
        bpi["resume_avg_rank"] = pd.to_numeric(bpi["RESUME AVG RANK"], errors="coerce")
        bpi["quality_avg_rank"] = pd.to_numeric(bpi["QUALITY AVG RANK"], errors="coerce")
        bpi["kpi_rank"] = pd.to_numeric(bpi["KPI"], errors="coerce")
        bpi["sor_rank"] = pd.to_numeric(bpi["SOR"], errors="coerce")
        bpi["wab_rank"] = pd.to_numeric(bpi["WAB RANK"], errors="coerce")
        bpi["trk_rank"] = pd.to_numeric(bpi["TRK"], errors="coerce")
        bpi["kp_rank"] = pd.to_numeric(bpi["KP"], errors="coerce")
        bpi["source"] = "archive_teamsheet_ranks"
        bpi["source_url"] = "archive://Teamsheet Ranks.csv"
        bpi["snapshot_date"] = bpi["season"].map(lambda s: config.selection_sunday_dates[str(s)])
        snapshots.append(
            bpi[
                [
                    "season",
                    "source_team_name",
                    "source",
                    "source_url",
                    "source_key",
                    "source_rank",
                    "source_value",
                    "snapshot_date",
                    "bpi_rank",
                    "net_rank",
                    "resume_avg_rank",
                    "quality_avg_rank",
                    "kpi_rank",
                    "sor_rank",
                    "wab_rank",
                    "trk_rank",
                    "kp_rank",
                ]
            ]
        )

        net = team_sheet[
            [
                "YEAR",
                "TEAM",
                "NET",
                "RESUME AVG RANK",
                "QUALITY AVG RANK",
                "KPI",
                "SOR",
                "WAB RANK",
                "TRK",
                "KP",
            ]
        ].copy()
        net["season"] = net["YEAR"].astype(int)
        net["source_team_name"] = net["TEAM"]
        net["source_key"] = "net"
        net["source_rank"] = pd.to_numeric(net["NET"], errors="coerce")
        net["source_value"] = net["source_rank"]
        net["net_rank"] = pd.to_numeric(net["NET"], errors="coerce")
        net["resume_avg_rank"] = pd.to_numeric(net["RESUME AVG RANK"], errors="coerce")
        net["quality_avg_rank"] = pd.to_numeric(net["QUALITY AVG RANK"], errors="coerce")
        net["kpi_rank"] = pd.to_numeric(net["KPI"], errors="coerce")
        net["sor_rank"] = pd.to_numeric(net["SOR"], errors="coerce")
        net["wab_rank"] = pd.to_numeric(net["WAB RANK"], errors="coerce")
        net["trk_rank"] = pd.to_numeric(net["TRK"], errors="coerce")
        net["kp_rank"] = pd.to_numeric(net["KP"], errors="coerce")
        net["source"] = "archive_teamsheet_ranks"
        net["source_url"] = "archive://Teamsheet Ranks.csv"
        net["snapshot_date"] = net["season"].map(lambda s: config.selection_sunday_dates[str(s)])
        snapshots.append(
            net[
                [
                    "season",
                    "source_team_name",
                    "source",
                    "source_url",
                    "source_key",
                    "source_rank",
                    "source_value",
                    "snapshot_date",
                    "net_rank",
                    "resume_avg_rank",
                    "quality_avg_rank",
                    "kpi_rank",
                    "sor_rank",
                    "wab_rank",
                    "trk_rank",
                    "kp_rank",
                ]
            ]
        )

    if not kenpom.empty:
        kp = kenpom[["YEAR", "TEAM", "CONF", "BADJ EM", "BADJ EM RANK", "BARTHAG"]].copy()
        kp["season"] = kp["YEAR"].astype(int)
        kp["source_team_name"] = kp["TEAM"]
        kp["source_key"] = "massey"
        kp["source_rank"] = pd.to_numeric(kp["BADJ EM RANK"], errors="coerce")
        kp["source_value"] = pd.to_numeric(kp["BADJ EM"], errors="coerce")
        kp["conference"] = kp["CONF"]
        kp["kenpom_badj_em"] = pd.to_numeric(kp["BADJ EM"], errors="coerce")
        kp["kenpom_badj_em_rank"] = pd.to_numeric(kp["BADJ EM RANK"], errors="coerce")
        kp["kenpom_barthag"] = pd.to_numeric(kp["BARTHAG"], errors="coerce")
        kp["source"] = "archive_kenpom_barttorvik"
        kp["source_url"] = "archive://KenPom Barttorvik.csv"
        kp["snapshot_date"] = kp["season"].map(lambda s: config.selection_sunday_dates[str(s)])
        snapshots.append(
            kp[
                [
                    "season",
                    "source_team_name",
                    "source",
                    "source_url",
                    "source_key",
                    "source_rank",
                    "source_value",
                    "snapshot_date",
                    "conference",
                    "kenpom_badj_em",
                    "kenpom_badj_em_rank",
                    "kenpom_barthag",
                ]
            ]
        )

    if not fte.empty:
        power = fte[["YEAR", "TEAM", "POWER RATING", "POWER RATING RANK"]].copy()
        power["season"] = power["YEAR"].astype(int)
        power["source_team_name"] = power["TEAM"]
        power["source_key"] = "massey"
        power["source_rank"] = pd.to_numeric(power["POWER RATING RANK"], errors="coerce")
        power["source_value"] = pd.to_numeric(power["POWER RATING"], errors="coerce")
        power["fivethirtyeight_power"] = pd.to_numeric(power["POWER RATING"], errors="coerce")
        power["fivethirtyeight_power_rank"] = pd.to_numeric(
            power["POWER RATING RANK"],
            errors="coerce",
        )
        power["source"] = "archive_538"
        power["source_url"] = "archive://538 Ratings.csv"
        power["snapshot_date"] = power["season"].map(lambda s: config.selection_sunday_dates[str(s)])
        snapshots.append(
            power[
                [
                    "season",
                    "source_team_name",
                    "source",
                    "source_url",
                    "source_key",
                    "source_rank",
                    "source_value",
                    "snapshot_date",
                    "fivethirtyeight_power",
                    "fivethirtyeight_power_rank",
                ]
            ]
        )

    if not evanmiya.empty:
        em = evanmiya[["YEAR", "TEAM", "RELATIVE RATING", "RELATIVE RATING RANK"]].copy()
        em["season"] = em["YEAR"].astype(int)
        em["source_team_name"] = em["TEAM"]
        em["source_key"] = "massey"
        em["source_rank"] = pd.to_numeric(em["RELATIVE RATING RANK"], errors="coerce")
        em["source_value"] = pd.to_numeric(em["RELATIVE RATING"], errors="coerce")
        em["evanmiya_relative_rating"] = pd.to_numeric(
            em["RELATIVE RATING"],
            errors="coerce",
        )
        em["evanmiya_relative_rating_rank"] = pd.to_numeric(
            em["RELATIVE RATING RANK"],
            errors="coerce",
        )
        em["source"] = "archive_evanmiya"
        em["source_url"] = "archive://EvanMiya.csv"
        em["snapshot_date"] = em["season"].map(lambda s: config.selection_sunday_dates[str(s)])
        snapshots.append(
            em[
                [
                    "season",
                    "source_team_name",
                    "source",
                    "source_url",
                    "source_key",
                    "source_rank",
                    "source_value",
                    "snapshot_date",
                    "evanmiya_relative_rating",
                    "evanmiya_relative_rating_rank",
                ]
            ]
        )

    return pd.concat(snapshots, ignore_index=True).dropna(subset=["source_team_name"])
