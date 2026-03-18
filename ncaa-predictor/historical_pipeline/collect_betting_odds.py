"""Stage 5: Collect historical betting odds from the-odds-api.com.

Walks every NCAA Tournament game date in tournament_games.parquet and fetches
the closest pre-game odds snapshot from the historical odds endpoint:

    GET /v4/historical/sports/basketball_ncaab/odds
        ?date=<selection_sunday_or_game_date>&markets=h2h,spreads&regions=us

Saves a flat parquet of game-level consensus odds to
    historical_pipeline/data/processed/tournament_odds.parquet

Columns
-------
season          int
game_date       str  (YYYY-MM-DD)
odds_api_event_id  str
home_team       str  (raw API name)
away_team       str  (raw API name)
canonical_team_id   str | None   (matched to our team IDs)
opponent_team_id    str | None
consensus_spread_home  float | None  (negative = home favored)
consensus_moneyline_home_prob  float | None  (0-1, vig-removed)
bookmaker_count int

Usage
-----
    python3 -m historical_pipeline.collect_betting_odds [--seasons 2022 2023]
    npm run historical:odds   # add this to package.json

Quota cost: 2 per game-date-window (h2h + spreads, 1 region).
The script caches raw API responses to data/cache/odds/ to minimise quota use.
"""
from __future__ import annotations

import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

from .common import (
    CACHE_ROOT,
    ODDS_API_KEY,
    PROCESSED_ROOT,
    ensure_dirs,
    load_season_config,
    slugify_team,
)

SPORT = "basketball_ncaab"
BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_CACHE_DIR = CACHE_ROOT / "odds"

# Preferred bookmakers in priority order for consensus calculation
PREFERRED_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbetus", "betonlineag",
                   "bovada", "williamhill_us", "sugarhouse", "betrivers"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = ODDS_API_KEY()
    if not key:
        raise SystemExit(
            "ODDS_API_KEY is not set. Add it to your .env file or environment."
        )
    return key


def normalize_name(name: str) -> str:
    """Lowercase, remove mascots (last word if >2 tokens), collapse spaces."""
    # Remove trailing mascot tokens like "Blue Devils", "Wildcats" etc.
    tokens = name.lower().split()
    # strip known suffixes
    for suffix in ("state", "st", "university", "tech"):
        if len(tokens) > 1 and tokens[-1] == suffix:
            pass  # keep – these are part of the school name
    return " ".join(tokens)


def _slugify(name: str) -> str:
    return slugify_team(name)


def best_match(api_name: str, team_lookup: dict[str, str], threshold: float = 0.62) -> str | None:
    """Fuzzy-match an API team name to our canonical team IDs."""
    slug = _slugify(api_name)
    if slug in team_lookup:
        return team_lookup[slug]

    best_score, best_id = 0.0, None
    for candidate_slug, team_id in team_lookup.items():
        score = SequenceMatcher(None, slug, candidate_slug).ratio()
        if score > best_score:
            best_score = score
            best_id = team_id
    if best_score >= threshold:
        return best_id
    return None


def american_to_prob(price: int) -> float:
    """Convert American odds to implied win probability (before vig removal)."""
    if price >= 0:
        return 100 / (price + 100)
    return abs(price) / (abs(price) + 100)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Normalize two implied probabilities to sum to 1 (vig removal)."""
    total = prob_a + prob_b
    if total <= 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def consensus_spread(outcomes: list[dict]) -> float | None:
    """Return average consensus spread for the home team from spread market outcomes."""
    home_points: list[float] = []
    for o in outcomes:
        pt = o.get("point")
        if pt is not None:
            # We'll collect as-is; caller decides which is home
            home_points.append(float(pt))
    if not home_points:
        return None
    return sum(home_points) / len(home_points)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cache_path(date_str: str) -> Path:
    return ODDS_CACHE_DIR / f"{date_str}.json"


def load_cache(date_str: str) -> list[dict] | None:
    p = cache_path(date_str)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def save_cache(date_str: str, data: list[dict]) -> None:
    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(date_str).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def fetch_historical_odds(date_iso: str, api_key: str) -> list[dict]:
    """Fetch historical NCAAB odds snapshot closest to date_iso (ISO 8601 UTC)."""
    cached = load_cache(date_iso)
    if cached is not None:
        print(f"  [cache] {date_iso}")
        return cached

    url = f"{BASE_URL}/historical/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "date": date_iso,
    }
    resp = requests.get(url, params=params, timeout=30)
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  [api]   {date_iso}  status={resp.status_code}  remaining={remaining}  used={used}")

    if resp.status_code == 422:
        # Date out of range or bad format – return empty
        save_cache(date_iso, [])
        return []
    resp.raise_for_status()

    payload = resp.json()
    # Historical endpoint wraps in {timestamp, previous_timestamp, next_timestamp, data: [...]}
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        data = []
    save_cache(date_iso, data)
    return data


def fetch_current_odds(api_key: str) -> list[dict]:
    """Fetch live/upcoming NCAAB odds (current season, not historical)."""
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=30)
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  [live]  status={resp.status_code}  remaining={remaining}")
    resp.raise_for_status()
    return resp.json() if isinstance(resp.json(), list) else []


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_game_odds(event: dict) -> dict:
    """Extract consensus spread and moneyline probability from a raw API event."""
    home = event.get("home_team", "")
    away = event.get("away_team", "")

    h2h_probs_home: list[float] = []
    spread_home: list[float] = []

    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])

            if key == "h2h":
                home_price = next(
                    (o["price"] for o in outcomes if o["name"] == home), None
                )
                away_price = next(
                    (o["price"] for o in outcomes if o["name"] == away), None
                )
                if home_price is not None and away_price is not None:
                    ph, pa = remove_vig(
                        american_to_prob(home_price),
                        american_to_prob(away_price),
                    )
                    h2h_probs_home.append(ph)

            elif key == "spreads":
                home_outcome = next(
                    (o for o in outcomes if o["name"] == home), None
                )
                if home_outcome and home_outcome.get("point") is not None:
                    spread_home.append(float(home_outcome["point"]))

    n_books = len(event.get("bookmakers", []))
    return {
        "odds_api_event_id": event.get("id"),
        "home_team": home,
        "away_team": away,
        "commence_time": event.get("commence_time"),
        "consensus_spread_home": (
            round(sum(spread_home) / len(spread_home), 2) if spread_home else None
        ),
        "consensus_moneyline_home_prob": (
            round(sum(h2h_probs_home) / len(h2h_probs_home), 4)
            if h2h_probs_home
            else None
        ),
        "bookmaker_count": n_books,
    }


# ---------------------------------------------------------------------------
# Match API events to tournament_games rows
# ---------------------------------------------------------------------------

def build_team_lookup(games_df: pd.DataFrame) -> dict[str, str]:
    """Build slug → canonical_team_id from all team names in games."""
    lookup: dict[str, str] = {}
    for _, row in games_df.iterrows():
        for name_col, id_col in [
            ("canonical_team_name", "canonical_team_id"),
            ("opponent_name", "opponent_team_id"),
        ]:
            nm = row.get(name_col) or row.get("team_name", "")
            tid = row.get(id_col, "")
            if nm and tid:
                lookup[_slugify(str(nm))] = str(tid)
    return lookup


def date_window_for_game(game_date: str, hours_before: int = 4) -> str:
    """Return an ISO 8601 UTC timestamp just before game time for the historical endpoint."""
    # Use noon UTC on game day as a reliable pre-game snapshot
    return f"{game_date}T12:00:00Z"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect_odds_for_seasons(
    seasons: list[int],
    games_df: pd.DataFrame,
    api_key: str,
    sleep_secs: float = 1.0,
) -> pd.DataFrame:
    team_lookup = build_team_lookup(games_df)
    rows: list[dict] = []

    for season in sorted(seasons):
        season_games = games_df[games_df["season"] == season].copy()
        if season_games.empty:
            print(f"[{season}] no games found, skipping")
            continue

        dates = sorted(season_games["game_date"].unique())
        print(f"\n[{season}] {len(dates)} unique game dates, {len(season_games)} games")

        for game_date in dates:
            snapshot_ts = date_window_for_game(game_date)
            events = fetch_historical_odds(snapshot_ts, api_key)
            time.sleep(sleep_secs)

            # Filter events to roughly game-date window (commence within ±2 days)
            for event in events:
                parsed = parse_game_odds(event)
                if parsed["consensus_spread_home"] is None and parsed["consensus_moneyline_home_prob"] is None:
                    continue

                home_id = best_match(parsed["home_team"], team_lookup)
                away_id = best_match(parsed["away_team"], team_lookup)

                rows.append(
                    {
                        "season": season,
                        "game_date": game_date,
                        "snapshot_ts": snapshot_ts,
                        **parsed,
                        "home_canonical_id": home_id,
                        "away_canonical_id": away_id,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "season", "game_date", "snapshot_ts", "odds_api_event_id",
                "home_team", "away_team", "commence_time",
                "consensus_spread_home", "consensus_moneyline_home_prob",
                "bookmaker_count", "home_canonical_id", "away_canonical_id",
            ]
        )
    return pd.DataFrame(rows)


def merge_odds_into_training(
    training_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join odds onto the training dataset (from team's perspective).

    For each training row:
      team = canonical_team_id  (the "A" side)
      opponent = opponent_team_id  (the "B" side)

    We match when:
      (home_canonical_id == team AND away_canonical_id == opponent)
      OR
      (home_canonical_id == opponent AND away_canonical_id == team)

    Tournament games are always neutral site, so "home" in the odds API is
    arbitrary (usually the higher-seeded / first-listed team). We store:
      betting_spread_a     : spread from team_a's perspective (negative = favored)
      betting_win_prob_a   : market-implied win prob for team_a (vig-removed)
    """
    if odds_df.empty:
        training_df["betting_spread_a"] = None
        training_df["betting_win_prob_a"] = None
        training_df["betting_bookmaker_count"] = 0
        training_df["missing_betting"] = 1
        return training_df

    # Build a lookup: (season, frozenset(team_a, team_b)) → odds row
    odds_lookup: dict[tuple, dict] = {}
    for _, row in odds_df.iterrows():
        key = (row["season"], frozenset(filter(None, [row["home_canonical_id"], row["away_canonical_id"]])))
        if len(key[1]) == 2:
            odds_lookup[key] = row.to_dict()

    spread_a_vals: list[float | None] = []
    prob_a_vals: list[float | None] = []
    book_counts: list[int] = []

    for _, row in training_df.iterrows():
        team_id = row.get("canonical_team_id")
        opp_id = row.get("opponent_team_id")
        season = int(row["season"])
        key = (season, frozenset(filter(None, [team_id, opp_id])))
        odds_row = odds_lookup.get(key)

        if odds_row is None:
            spread_a_vals.append(None)
            prob_a_vals.append(None)
            book_counts.append(0)
            continue

        home_id = odds_row.get("home_canonical_id")
        spread_home = odds_row.get("consensus_spread_home")
        prob_home = odds_row.get("consensus_moneyline_home_prob")
        n_books = int(odds_row.get("bookmaker_count", 0))

        # Flip if team is away in the odds API
        if home_id == team_id:
            spread_a_vals.append(spread_home)
            prob_a_vals.append(prob_home)
        else:
            spread_a_vals.append(-spread_home if spread_home is not None else None)
            prob_a_vals.append(1 - prob_home if prob_home is not None else None)
        book_counts.append(n_books)

    training_df["betting_spread_a"] = spread_a_vals
    training_df["betting_win_prob_a"] = prob_a_vals
    training_df["betting_bookmaker_count"] = book_counts
    training_df["missing_betting"] = training_df["betting_spread_a"].isna().astype(int)
    return training_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical NCAAB betting odds from the-odds-api")
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=None,
        help="Seasons to fetch (e.g. 2022 2023 2024). Defaults to all training seasons.",
    )
    parser.add_argument(
        "--games",
        default=str(PROCESSED_ROOT / "tournament_games.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(PROCESSED_ROOT / "tournament_odds.parquet"),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls.",
    )
    args = parser.parse_args()

    ensure_dirs()
    api_key = get_api_key()

    config = load_season_config()
    seasons = args.seasons if args.seasons else config.training_seasons
    # Historical odds available from 2020 onwards (API launched June 2020)
    seasons = [s for s in seasons if s >= 2021]
    if not seasons:
        print("No seasons in range for historical odds (need 2021+). Exiting.")
        return

    games_df = pd.read_parquet(args.games)

    odds_df = collect_odds_for_seasons(seasons, games_df, api_key, sleep_secs=args.sleep)
    output_path = Path(args.output)
    odds_df.to_parquet(output_path, index=False)
    matched = odds_df[odds_df["home_canonical_id"].notna() & odds_df["away_canonical_id"].notna()]
    print(f"\nWrote {len(odds_df)} odds rows to {output_path}")
    print(f"Matched to canonical teams: {len(matched)}/{len(odds_df)}")


if __name__ == "__main__":
    main()
