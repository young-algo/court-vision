from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .common import load_season_config, resolve_massey_master_path


def has_massey_master_csv(path: str | None = None) -> bool:
    return resolve_massey_master_path(path) is not None


def read_massey_master(path: str | None = None) -> pd.DataFrame:
    resolved = resolve_massey_master_path(path)
    if resolved is None:
        raise FileNotFoundError("Could not locate massey_master.csv")
    return pd.read_csv(resolved, low_memory=False)


def parse_record_win_pct(record: object) -> float | pd.NA:
    if not isinstance(record, str):
        return pd.NA
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", record)
    if not match:
        return pd.NA
    wins = int(match.group(1))
    losses = int(match.group(2))
    total_games = wins + losses
    if total_games == 0:
        return 0.0
    return wins / total_games


def infer_snapshot_week(group: pd.DataFrame) -> int | None:
    week_values = sorted(
        {
            int(week)
            for week in pd.to_numeric(group.get("week"), errors="coerce").dropna().tolist()
            if int(week) >= 0
        }
    )
    if not week_values:
        return None
    nonzero_weeks = [week for week in week_values if week > 0]
    if not nonzero_weeks:
        return week_values[-1]
    highest_week = nonzero_weeks[-1]
    return highest_week - 1 if highest_week >= 19 else highest_week


def select_pre_tournament_snapshot_rows(
    frame: pd.DataFrame,
    selection_sunday: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    dated = frame[frame["parsed_date"].notna()].copy()
    if not dated.empty:
        before_tournament = dated[dated["parsed_date"] <= selection_sunday].copy()
        if not before_tournament.empty:
            snapshot_date = before_tournament["parsed_date"].max()
            selected = frame[frame["parsed_date"] == snapshot_date].copy()
            return selected, "latest_dated_pre_selection_sunday"

        nearest_index = (dated["parsed_date"] - selection_sunday).abs().sort_values().index[0]
        snapshot_date = dated.loc[nearest_index, "parsed_date"]
        selected = frame[frame["parsed_date"] == snapshot_date].copy()
        return selected, "nearest_dated_snapshot_fallback"

    snapshot_week = infer_snapshot_week(frame)
    if snapshot_week is None:
        return frame.iloc[0:0].copy(), "missing_snapshot"
    selected = frame[pd.to_numeric(frame["week"], errors="coerce") == snapshot_week].copy()
    return selected, "week_order_fallback"


def massey_master_snapshots(path: str | None = None) -> pd.DataFrame:
    resolved = resolve_massey_master_path(path)
    if resolved is None:
        raise FileNotFoundError("Could not locate massey_master.csv")

    config = load_season_config()
    valid_seasons = set(config.training_seasons)
    frame = read_massey_master(path).copy()
    frame = frame[frame["year"].isin(valid_seasons)].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["parsed_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["season"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
    frame["median"] = pd.to_numeric(frame["median"], errors="coerce")
    frame["stdev"] = pd.to_numeric(frame["stdev"], errors="coerce")
    frame["massey_win_pct"] = frame["record"].map(parse_record_win_pct)

    snapshot_rows: list[pd.DataFrame] = []
    for season, group in frame.groupby("season"):
        selection_sunday = pd.Timestamp(config.selection_sunday_dates[str(int(season))])
        selected, selection_method = select_pre_tournament_snapshot_rows(group, selection_sunday)
        if selected.empty:
            continue
        selected = selected.copy()
        selected["snapshot_date"] = (
            selected["parsed_date"].dt.date.astype(str).replace("NaT", selection_sunday.date().isoformat())
        )
        selected["snapshot_selection_method"] = selection_method
        snapshot_rows.append(selected)

    if not snapshot_rows:
        return pd.DataFrame()

    selected = pd.concat(snapshot_rows, ignore_index=True)
    selected["source_team_name"] = selected["team"]
    selected["source"] = "massey_master_csv"
    selected["source_url"] = str(resolved)
    selected["source_key"] = "massey_master"
    selected["source_rank"] = selected["rank"]
    selected["source_value"] = 400 - pd.to_numeric(selected["rank"], errors="coerce")
    selected["conference"] = selected["conference"].replace({"": pd.NA})
    selected["massey_composite_rank"] = pd.to_numeric(selected["rank"], errors="coerce")
    selected["massey_mean_rank"] = pd.to_numeric(selected["mean"], errors="coerce")
    selected["massey_median_rank"] = pd.to_numeric(selected["median"], errors="coerce")
    selected["massey_rank_stdev"] = pd.to_numeric(selected["stdev"], errors="coerce")
    selected["massey_week"] = pd.to_numeric(selected["week"], errors="coerce")
    selected["massey_record"] = selected["record"]

    columns = [
        "season",
        "source_team_name",
        "source",
        "source_url",
        "source_key",
        "source_rank",
        "source_value",
        "snapshot_date",
        "conference",
        "massey_composite_rank",
        "massey_mean_rank",
        "massey_median_rank",
        "massey_rank_stdev",
        "massey_win_pct",
        "massey_week",
        "massey_record",
        "snapshot_selection_method",
    ]
    return selected[columns].drop_duplicates(subset=["season", "source_team_name", "source_key"])

