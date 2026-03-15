from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
import requests

from .archive_data import archive_tournament_games, has_archive_bundle
from .aliases import AliasResolver
from .common import DEFAULT_HEADERS, PROCESSED_ROOT, RAW_ROOT, ensure_dirs, load_season_config
from .kaggle_data import has_kaggle_bundle, kaggle_tournament_games

NCAA_API_ROOT = "https://ncaa-api.henrygd.me"
SUPPORTED_MONTHS = ("03", "04")
ROUND_MAP = {
    "first round": "round_of_64",
    "second round": "round_of_32",
    "sweet 16": "sweet_16",
    "elite 8": "elite_8",
    "elite eight": "elite_8",
    "final four": "final_four",
    "championship": "championship",
}


def fetch_json(path: str) -> dict:
    try:
        response = requests.get(
            f"{NCAA_API_ROOT}{path}",
            timeout=10,
            headers=DEFAULT_HEADERS,
        )
    except requests.RequestException:
        return {}
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def scoreboard_dates(season: int) -> list[str]:
    dates: list[str] = []
    for month in SUPPORTED_MONTHS:
        payload = fetch_json(f"/schedule/basketball-men/d1/{season}/{month}")
        for game_date in payload.get("gameDates", []):
            dates.append(game_date["contest_date"])
    return sorted(set(dates))


def extract_games_for_date(season: int, contest_date: str) -> list[dict]:
    month, day, year = contest_date.split("-")
    payload = fetch_json(f"/scoreboard/basketball-men/d1/{year}/{month}/{day}/all-conf")
    rows: list[dict] = []
    for wrapper in payload.get("games", []):
        game = wrapper.get("game", {})
        round_name = (game.get("bracketRound") or "").strip()
        normalized_round = (
            html.unescape(round_name)
            .replace("®", "")
            .strip()
            .lower()
        )
        if normalized_round not in ROUND_MAP:
            continue

        away = game.get("away", {})
        home = game.get("home", {})
        rows.append(
            {
                "season": season,
                "game_date": game.get("startDate", contest_date),
                "round": ROUND_MAP[normalized_round],
                "region": game.get("bracketRegion") or None,
                "neutral_site": True,
                "team_name": away.get("names", {}).get("short"),
                "opponent_name": home.get("names", {}).get("short"),
                "seed": pd.to_numeric(away.get("seed"), errors="coerce"),
                "opponent_seed": pd.to_numeric(home.get("seed"), errors="coerce"),
                "team_score": pd.to_numeric(away.get("score"), errors="coerce"),
                "opponent_score": pd.to_numeric(home.get("score"), errors="coerce"),
                "game_id": game.get("gameID"),
                "source_url": f"{NCAA_API_ROOT}/scoreboard/basketball-men/d1/{year}/{month}/{day}/all-conf",
            }
        )
    return rows


def collect_tournament_games(season: int) -> pd.DataFrame:
    rows: list[dict] = []
    for contest_date in scoreboard_dates(season):
        rows.extend(extract_games_for_date(season, contest_date))

    if not rows:
        raise ValueError(f"No tournament games found for season {season}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(PROCESSED_ROOT / "tournament_games.parquet"))
    args = parser.parse_args()

    ensure_dirs()
    alias_resolver = AliasResolver()
    config = load_season_config()
    seasons: list[pd.DataFrame] = []

    if has_archive_bundle():
        print("collecting tournament games from archive dataset")
        tournament_games = archive_tournament_games()
        tournament_games = alias_resolver.resolve_frame(tournament_games, "team_name")
        opponent_matches = tournament_games["opponent_name"].map(alias_resolver.resolve)
        tournament_games["opponent_team_id"] = opponent_matches.map(lambda match: match.canonical_team_id)
        tournament_games["opponent_match_method"] = opponent_matches.map(lambda match: match.match_method)
        tournament_games["opponent_needs_review"] = opponent_matches.map(lambda match: match.needs_review)

        output_path = Path(args.output)
        if output_path.suffix == ".parquet":
            tournament_games.to_parquet(output_path, index=False)
        else:
            tournament_games.to_csv(output_path, index=False)

        raw_copy = RAW_ROOT / "archive_tournament_games.csv"
        tournament_games.to_csv(raw_copy, index=False)
        print(f"wrote {len(tournament_games)} tournament game rows to {output_path}")
        return

    if has_kaggle_bundle():
        print("collecting tournament games from Kaggle bundle")
        tournament_games = kaggle_tournament_games()
        tournament_games = alias_resolver.resolve_frame(tournament_games, "team_name")
        opponent_matches = tournament_games["opponent_name"].map(alias_resolver.resolve)
        tournament_games["opponent_team_id"] = opponent_matches.map(lambda match: match.canonical_team_id)
        tournament_games["opponent_match_method"] = opponent_matches.map(lambda match: match.match_method)
        tournament_games["opponent_needs_review"] = opponent_matches.map(lambda match: match.needs_review)

        output_path = Path(args.output)
        if output_path.suffix == ".parquet":
            tournament_games.to_parquet(output_path, index=False)
        else:
            tournament_games.to_csv(output_path, index=False)

        raw_copy = RAW_ROOT / "kaggle_tournament_games.csv"
        tournament_games.to_csv(raw_copy, index=False)
        print(f"wrote {len(tournament_games)} tournament game rows to {output_path}")
        return

    for season in config.training_seasons:
        print(f"collecting tournament games for {season}")
        frame = collect_tournament_games(season)
        frame = alias_resolver.resolve_frame(frame, "team_name")
        opponent_matches = frame["opponent_name"].map(alias_resolver.resolve)
        frame["opponent_team_id"] = opponent_matches.map(lambda match: match.canonical_team_id)
        frame["opponent_match_method"] = opponent_matches.map(lambda match: match.match_method)
        frame["opponent_needs_review"] = opponent_matches.map(lambda match: match.needs_review)
        seasons.append(frame)

    tournament_games = pd.concat(seasons, ignore_index=True)

    output_path = Path(args.output)
    if output_path.suffix == ".parquet":
        tournament_games.to_parquet(output_path, index=False)
    else:
        tournament_games.to_csv(output_path, index=False)

    raw_copy = RAW_ROOT / "ncaa_api_tournament_games.csv"
    tournament_games.to_csv(raw_copy, index=False)
    print(f"wrote {len(tournament_games)} tournament game rows to {output_path}")


if __name__ == "__main__":
    main()
