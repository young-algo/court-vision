from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "historical_pipeline"
DATA_ROOT = PIPELINE_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
CACHE_ROOT = DATA_ROOT / "cache"
CONFIG_ROOT = PIPELINE_ROOT / "config"
ALIASES_PATH = PIPELINE_ROOT / "aliases" / "historical_aliases.csv"
MODEL_ROOT = ROOT / "data" / "models"
EXTERNAL_ROOT = DATA_ROOT / "external"
KAGGLE_DEFAULT_ROOT = EXTERNAL_ROOT / "kaggle"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ncaa-predictor-historical-pipeline/1.0)",
}


@dataclass(frozen=True)
class SeasonConfig:
    training_seasons: list[int]
    snapshot_policy: str
    selection_sunday_dates: dict[str, str]


def ODDS_API_KEY() -> str:
    """Return the Odds API key from the environment."""
    return os.getenv("ODDS_API_KEY", "")


def ensure_dirs() -> None:
    for directory in (RAW_ROOT, PROCESSED_ROOT, CACHE_ROOT, MODEL_ROOT, EXTERNAL_ROOT):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass


def load_season_config() -> SeasonConfig:
    payload = json.loads((CONFIG_ROOT / "seasons.json").read_text())
    return SeasonConfig(
        training_seasons=payload["training_seasons"],
        snapshot_policy=payload["snapshot_policy"],
        selection_sunday_dates=payload["selection_sunday_dates"],
    )


def slugify_team(name: str) -> str:
    return (
        name.lower()
        .replace("&", "and")
        .replace("'", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace(" ", "-")
    )


def fetch_html(url: str) -> str:
    response = requests.get(url, timeout=30, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    return response.text


def fetch_table(url: str) -> pd.DataFrame:
    tables = pd.read_html(fetch_html(url))
    if not tables:
        raise ValueError(f"No tables found at {url}")
    return tables[0]


def soup(url: str) -> BeautifulSoup:
    return BeautifulSoup(fetch_html(url), "lxml")


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    ensure_dirs()
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format for {path}")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def rolling_pairs(seasons: Iterable[int]) -> list[tuple[list[int], int]]:
    ordered = list(seasons)
    return [(ordered[:index], ordered[index]) for index in range(1, len(ordered))]


def resolve_kaggle_root(explicit: str | None = None) -> Path | None:
    candidates = [
        explicit,
        os.getenv("KAGGLE_MARCH_MADNESS_ROOT"),
        str(KAGGLE_DEFAULT_ROOT),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        resolved = Path(candidate).expanduser().resolve()
        if resolved.exists() and list(resolved.rglob("MTeams.csv")):
            return resolved

    return None


def resolve_archive_root(explicit: str | None = None) -> Path | None:
    candidates = [
        explicit,
        os.getenv("NCAA_ARCHIVE_ROOT"),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        resolved = Path(candidate).expanduser().resolve()
        if resolved.exists():
            return resolved

    return None


def resolve_massey_master_path(explicit: str | None = None) -> Path | None:
    candidates = [
        explicit,
        os.getenv("MASSEY_MASTER_CSV"),
        str(EXTERNAL_ROOT / "massey_master.csv"),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        resolved = Path(candidate).expanduser().resolve()
        if resolved.exists():
            return resolved

    return None
