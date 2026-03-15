from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import pandas as pd

from .archive_data import archive_snapshots, has_archive_bundle
from .aliases import AliasResolver
from .common import CACHE_ROOT, PROCESSED_ROOT, RAW_ROOT, ensure_dirs, load_season_config, normalize_columns
from .kaggle_data import has_kaggle_bundle, kaggle_massey_snapshots
from .massey_master_data import has_massey_master_csv, massey_master_snapshots


@dataclass(frozen=True)
class SnapshotSource:
    key: str
    fetcher: Callable[[int], pd.DataFrame]


def fetch_espn_bpi(season: int) -> pd.DataFrame:
    url = f"https://www.espn.com/mens-college-basketball/bpi/_/season/{season}/sort/bpi.bpirank/dir/asc"
    frame = normalize_columns(pd.read_html(url)[0])
    return frame.assign(source="bpi", source_url=url, season=season)


def fetch_warren_nolan_net(season: int) -> pd.DataFrame:
    url = f"https://www.warrennolan.com/basketball/{season}/net-nitty"
    frame = normalize_columns(pd.read_html(url)[0])
    return frame.assign(source="net", source_url=url, season=season)


def fetch_massey(season: int) -> pd.DataFrame:
    cached = fetch_cached_massey_archive(season)
    if cached is not None:
        return cached
    url = f"https://masseyratings.com/cb/compare.htm?s={season}"
    frame = normalize_columns(pd.read_html(url)[0])
    return frame.assign(source="massey", source_url=url, season=season)


def fetch_cached_massey_archive(season: int) -> pd.DataFrame | None:
    archive_dir = CACHE_ROOT / "massey_archive"
    if not archive_dir.exists():
        return None

    candidates = sorted(archive_dir.glob(f"compare{season}-*.html"))
    if not candidates:
        return None

    latest = candidates[-1]
    tables = pd.read_html(latest)
    if not tables:
        return None

    frame = normalize_columns(tables[0])
    return frame.assign(source="massey", source_url=str(latest), season=season)


def collect_source(source: SnapshotSource, season: int, aliases: AliasResolver) -> pd.DataFrame:
    frame = source.fetcher(season)
    frame = frame.reset_index(drop=True)
    team_column_candidates = [column for column in frame.columns if "team" in column or "school" in column]
    if not team_column_candidates:
        raise ValueError(f"Could not find a team-name column for {source.key} {season}")

    team_column = team_column_candidates[0]
    frame = frame.rename(columns={team_column: "source_team_name"})
    frame["source_team_name"] = frame["source_team_name"].map(clean_source_team_name)
    frame["source_rank"] = frame.index + 1
    frame["source_value"] = derive_source_value(frame, source.key)
    frame = attach_source_native_columns(frame, source.key)
    frame = aliases.resolve_frame(frame, "source_team_name")
    frame["snapshot_date"] = load_season_config().selection_sunday_dates[str(season)]
    frame["source_key"] = source.key
    return frame


def clean_source_team_name(team_name: object) -> object:
    if not isinstance(team_name, str):
        return team_name
    cleaned = re.sub(r"\s{2,}.*$", "", team_name).strip()
    return cleaned.replace("St.", "Saint").strip()


def derive_source_value(frame: pd.DataFrame, source_key: str) -> pd.Series:
    if source_key == "net" and "net" in frame.columns:
        return pd.to_numeric(frame["net"], errors="coerce")

    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"season", "source", "source_url", "conf"}
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    if numeric_columns:
        return pd.to_numeric(frame[numeric_columns[0]], errors="coerce")

    return pd.Series(len(frame) - frame.index, index=frame.index, dtype=float)


def attach_source_native_columns(frame: pd.DataFrame, source_key: str) -> pd.DataFrame:
    enriched = frame.copy()
    if "conf" in enriched.columns:
        enriched["conference"] = enriched["conf"]

    if source_key == "bpi":
        enriched["bpi_rank"] = pd.to_numeric(enriched["source_rank"], errors="coerce")

    if source_key == "net":
        enriched["net_rank"] = pd.to_numeric(
            enriched.get("net", enriched["source_rank"]),
            errors="coerce",
        )

    if source_key == "massey":
        enriched["massey_ordinal_rank"] = pd.to_numeric(
            enriched["source_rank"],
            errors="coerce",
        )

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(PROCESSED_ROOT / "season_snapshots.parquet"))
    args = parser.parse_args()

    ensure_dirs()
    aliases = AliasResolver()
    config = load_season_config()
    sources = [
        SnapshotSource("bpi", fetch_espn_bpi),
        SnapshotSource("net", fetch_warren_nolan_net),
    ]
    if not has_kaggle_bundle():
        sources.append(SnapshotSource("massey", fetch_massey))

    collected: list[pd.DataFrame] = []
    if has_archive_bundle():
        archive_frame = archive_snapshots()
        archive_frame = aliases.resolve_frame(archive_frame, "source_team_name")
        collected.append(archive_frame)

    if has_kaggle_bundle():
        kaggle_massey = kaggle_massey_snapshots()
        kaggle_massey = aliases.resolve_frame(kaggle_massey, "source_team_name")
        collected.append(kaggle_massey)

    if has_massey_master_csv():
        massey_master = massey_master_snapshots()
        massey_master = aliases.resolve_frame(massey_master, "source_team_name")
        collected.append(massey_master)

    for season in config.training_seasons:
        for source in sources:
            try:
                collected.append(collect_source(source, season, aliases))
            except Exception as exc:  # noqa: BLE001
                collected.append(
                    pd.DataFrame(
                        [
                            {
                                "season": season,
                                "source_key": source.key,
                                "snapshot_date": config.selection_sunday_dates[str(season)],
                                "source_team_name": None,
                                "canonical_team_id": None,
                                "canonical_team_name": None,
                                "match_method": "source_failure",
                                "needs_review": True,
                                "fetch_error": str(exc),
                            }
                        ]
                    )
                )

    snapshots = pd.concat(collected, ignore_index=True)
    output_path = Path(args.output)
    if output_path.suffix == ".parquet":
        snapshots.to_parquet(output_path, index=False)
    else:
        snapshots.to_csv(output_path, index=False)

    snapshots.to_csv(RAW_ROOT / "season_snapshots_debug.csv", index=False)
    print(f"wrote {len(snapshots)} snapshot rows to {output_path}")


if __name__ == "__main__":
    main()
