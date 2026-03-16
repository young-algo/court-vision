from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import PROCESSED_ROOT, ensure_dirs, slugify_team

RANK_FEATURES = [
    "bpi_rank",
    "net_rank",
    "massey_ordinal_rank",
    "massey_composite_rank",
    "massey_mean_rank",
    "massey_median_rank",
    "resume_avg_rank",
    "quality_avg_rank",
    "kpi_rank",
    "sor_rank",
    "wab_rank",
    "trk_rank",
    "kp_rank",
]
POWER_FEATURES = [
    "kenpom_badj_em",
    "evanmiya_relative_rating",
    "fivethirtyeight_power",
    "massey_win_pct",
]
LOWER_BETTER_FEATURES = ["massey_rank_stdev"]
RESUME_RANK_FEATURES = [
    "resume_avg_rank",
    "quality_avg_rank",
    "kpi_rank",
    "sor_rank",
    "wab_rank",
    "trk_rank",
    "kp_rank",
]
LEGACY_SUMMARY_FIELDS = [
    "legacy_bpi_value",
    "legacy_net_rank",
    "legacy_massey_value",
]
ROUND_ORDINAL_MAP = {
    "round_of_64": 1,
    "round_of_32": 2,
    "sweet_16": 3,
    "elite_8": 4,
    "final_four": 5,
    "championship": 6,
}


def normalize_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    cleaned = re.sub(r"\s*\([^)]*\)", "", name)
    cleaned = re.sub(r"\s{2,}.*$", "", cleaned)
    cleaned = cleaned.replace("Saint", "St")
    cleaned = cleaned.replace("State", "St")
    return slugify_team(cleaned)


def name_variants(name: object) -> list[str]:
    normalized = normalize_name(name)
    if not normalized:
        return []
    tokens = normalized.split("-")
    variants = {normalized}
    for trim in range(1, min(3, len(tokens) - 1) + 1):
        variants.add("-".join(tokens[:-trim]))
    return sorted(variants, key=len)


def best_season_match(
    source_team_name: object,
    season: int,
    season_lookup: dict[int, dict[str, str]],
) -> str | None:
    candidates = season_lookup.get(season, {})
    source_variants = name_variants(source_team_name)
    if not source_variants or not candidates:
        return None

    for variant in source_variants:
        if variant in candidates:
            return candidates[variant]

    scored = [
        (
            max(SequenceMatcher(None, variant, candidate).ratio() for variant in source_variants),
            team_id,
        )
        for candidate, team_id in candidates.items()
    ]
    best_score, team_id = max(scored, default=(0.0, None))
    return team_id if best_score >= 0.72 else None


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def min_value(frame: pd.DataFrame, column: str) -> float | pd.NA:
    values = numeric_series(frame, column).dropna()
    return float(values.min()) if not values.empty else pd.NA


def max_value(frame: pd.DataFrame, column: str) -> float | pd.NA:
    values = numeric_series(frame, column).dropna()
    return float(values.max()) if not values.empty else pd.NA


def first_text(frame: pd.DataFrame, column: str) -> str | pd.NA:
    if column not in frame.columns:
        return pd.NA
    values = frame[column].dropna()
    return str(values.iloc[0]) if not values.empty else pd.NA


def build_snapshot_summary(
    snapshots: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    tournament_teams = pd.concat(
        [
            games[["season", "team_name", "canonical_team_id"]].rename(
                columns={"team_name": "display_name", "canonical_team_id": "team_id"}
            ),
            games[["season", "opponent_name", "opponent_team_id"]].rename(
                columns={"opponent_name": "display_name", "opponent_team_id": "team_id"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["team_id"])
    tournament_teams["normalized_name"] = tournament_teams["display_name"].map(normalize_name)

    season_lookup = {
        season: {
            row.normalized_name: row.team_id
            for row in group.itertuples()
            if row.normalized_name and row.team_id
        }
        for season, group in tournament_teams.groupby("season")
    }

    mapped = snapshots.copy()
    mapped["matched_team_id"] = mapped.apply(
        lambda row: best_season_match(row.get("source_team_name"), int(row["season"]), season_lookup),
        axis=1,
    )
    mapped = mapped.dropna(subset=["matched_team_id"])

    summary_rows: list[dict] = []
    for (season, team_id), group in mapped.groupby(["season", "matched_team_id"]):
        bpi_rows = group[group["source_key"] == "bpi"]
        net_rows = group[group["source_key"] == "net"]
        massey_rows = group[group["source_key"] == "massey"]
        summary_rows.append(
            {
                "season": season,
                "team_id": team_id,
                "conference": first_text(group, "conference"),
                "legacy_bpi_value": max_value(bpi_rows, "source_value"),
                "legacy_net_rank": min_value(net_rows, "net_rank"),
                "legacy_massey_value": max_value(massey_rows, "source_value"),
                "bpi_rank": min_value(group, "bpi_rank"),
                "net_rank": min_value(group, "net_rank"),
                "massey_ordinal_rank": min_value(group, "massey_ordinal_rank"),
                "massey_composite_rank": min_value(group, "massey_composite_rank"),
                "massey_mean_rank": min_value(group, "massey_mean_rank"),
                "massey_median_rank": min_value(group, "massey_median_rank"),
                "massey_rank_stdev": min_value(group, "massey_rank_stdev"),
                "massey_win_pct": max_value(group, "massey_win_pct"),
                "kenpom_badj_em": max_value(group, "kenpom_badj_em"),
                "evanmiya_relative_rating": max_value(group, "evanmiya_relative_rating"),
                "fivethirtyeight_power": max_value(group, "fivethirtyeight_power"),
                "resume_avg_rank": min_value(group, "resume_avg_rank"),
                "quality_avg_rank": min_value(group, "quality_avg_rank"),
                "kpi_rank": min_value(group, "kpi_rank"),
                "sor_rank": min_value(group, "sor_rank"),
                "wab_rank": min_value(group, "wab_rank"),
                "trk_rank": min_value(group, "trk_rank"),
                "kp_rank": min_value(group, "kp_rank"),
            }
        )

    return pd.DataFrame(summary_rows)


def expand_perspectives(features: pd.DataFrame) -> pd.DataFrame:
    reverse = features.copy()
    pairs = {
        ("team_name", "opponent_name"),
        ("canonical_team_id", "opponent_team_id"),
        ("seed", "opponent_seed"),
        ("team_score", "opponent_score"),
    }
    for column in features.columns:
        if column.startswith("team_"):
            opposite = column.replace("team_", "opp_", 1)
            if opposite in features.columns:
                pairs.add((column, opposite))

    for left, right in sorted(pairs):
        reverse[left], reverse[right] = features[right], features[left]

    return pd.concat([features, reverse], ignore_index=True)


def rank_to_percentile(rank: pd.Series, season_max: pd.Series) -> pd.Series:
    FIXED_FIELD_SIZE = 362  # typical D1 team count; matches TS runtime denominator
    return 1 - ((rank - 1) / (FIXED_FIELD_SIZE - 1))


def add_rank_feature(frame: pd.DataFrame, feature: str) -> None:
    team_col = f"team_{feature}"
    opp_col = f"opp_{feature}"
    missing_col = f"missing_{feature}"
    diff_col = f"{feature}_percentile_diff"
    season_max = pd.to_numeric(
        frame.groupby("season")[[team_col, opp_col]].transform("max").max(axis=1),
        errors="coerce",
    )
    team_values = pd.to_numeric(frame[team_col], errors="coerce")
    opp_values = pd.to_numeric(frame[opp_col], errors="coerce")
    frame[missing_col] = (team_values.isna() | opp_values.isna()).astype(int)
    team_pct = rank_to_percentile(team_values.where(team_values.notna(), season_max), season_max)
    opp_pct = rank_to_percentile(opp_values.where(opp_values.notna(), season_max), season_max)
    frame[diff_col] = (team_pct - opp_pct).where(frame[missing_col] == 0, 0.0)


def add_power_feature(frame: pd.DataFrame, feature: str) -> None:
    team_col = f"team_{feature}"
    opp_col = f"opp_{feature}"
    missing_col = f"missing_{feature}"
    diff_col = f"{feature}_power_diff"
    team_values = pd.to_numeric(frame[team_col], errors="coerce")
    opp_values = pd.to_numeric(frame[opp_col], errors="coerce")
    frame[missing_col] = (team_values.isna() | opp_values.isna()).astype(int)
    frame[diff_col] = (team_values - opp_values).where(frame[missing_col] == 0, 0.0).fillna(0.0)


def add_lower_better_feature(frame: pd.DataFrame, feature: str) -> None:
    team_col = f"team_{feature}"
    opp_col = f"opp_{feature}"
    missing_col = f"missing_{feature}"
    diff_col = f"{feature}_advantage"
    team_values = pd.to_numeric(frame[team_col], errors="coerce")
    opp_values = pd.to_numeric(frame[opp_col], errors="coerce")
    frame[missing_col] = (team_values.isna() | opp_values.isna()).astype(int)
    frame[diff_col] = (opp_values - team_values).where(frame[missing_col] == 0, 0.0).fillna(0.0)


def apply_semantic_features(features: pd.DataFrame) -> pd.DataFrame:
    enriched = expand_perspectives(features)
    enriched["seed_diff"] = pd.to_numeric(enriched["seed"], errors="coerce") - pd.to_numeric(
        enriched["opponent_seed"],
        errors="coerce",
    )
    enriched["same_conference"] = (
        enriched["team_conference"].fillna("") == enriched["opp_conference"].fillna("")
    ).astype(int)
    enriched["margin"] = pd.to_numeric(enriched["team_score"], errors="coerce") - pd.to_numeric(
        enriched["opponent_score"],
        errors="coerce",
    )
    enriched["seed_kenpom_interaction"] = 0.0
    enriched["abs_seed_same_conference_interaction"] = (
        enriched["seed_diff"].abs() * enriched["same_conference"]
    ).astype(float)
    enriched["team_win"] = (enriched["margin"] > 0).astype(int)

    for feature in RANK_FEATURES:
        add_rank_feature(enriched, feature)

    for feature in POWER_FEATURES:
        add_power_feature(enriched, feature)

    enriched["seed_kenpom_interaction"] = (
        enriched["seed_diff"] * enriched["kenpom_badj_em_power_diff"]
    ).where(enriched["missing_kenpom_badj_em"] == 0, 0.0)

    for feature in LOWER_BETTER_FEATURES:
        add_lower_better_feature(enriched, feature)

    resume_diff_columns = [f"{feature}_percentile_diff" for feature in RESUME_RANK_FEATURES]
    resume_missing_columns = [f"missing_{feature}" for feature in RESUME_RANK_FEATURES]
    resume_available = (1 - enriched[resume_missing_columns]).sum(axis=1)
    enriched["missing_resume_rank_blend"] = (resume_available == 0).astype(int)
    enriched["resume_rank_blend_percentile_diff"] = (
        enriched[resume_diff_columns].sum(axis=1) / resume_available.replace(0, 1)
    ).where(enriched["missing_resume_rank_blend"] == 0, 0.0)

    enriched["available_predictive_rank_count"] = (
        3
        - enriched[["missing_bpi_rank", "missing_net_rank", "missing_massey_ordinal_rank"]].sum(axis=1)
    )
    enriched["available_power_feature_count"] = (
        3
        - enriched[
            [
                "missing_kenpom_badj_em",
                "missing_evanmiya_relative_rating",
                "missing_fivethirtyeight_power",
            ]
        ].sum(axis=1)
    )
    enriched["available_resume_feature_count"] = resume_available
    enriched["available_signal_count"] = (
        enriched["available_predictive_rank_count"]
        + enriched["available_power_feature_count"]
        + enriched["available_resume_feature_count"]
    )
    enriched["legacy_source_coverage_count"] = (
        enriched[[f"team_{field}" for field in LEGACY_SUMMARY_FIELDS]].notna().sum(axis=1)
    )

    # --- Round-aware features (Step 2) ---
    if "round" in enriched.columns:
        enriched["tournament_round_ordinal"] = enriched["round"].map(ROUND_ORDINAL_MAP).fillna(1).astype(float)
    else:
        enriched["tournament_round_ordinal"] = 1.0
    enriched["round_seed_interaction"] = (
        enriched["tournament_round_ordinal"] * enriched["seed_diff"].abs()
    ).astype(float)
    enriched["round_kenpom_interaction"] = (
        enriched["tournament_round_ordinal"] * enriched["kenpom_badj_em_power_diff"]
    ).where(enriched["missing_kenpom_badj_em"] == 0, 0.0)

    # --- Seed nonlinearity features (Step 4) ---
    enriched["seed_diff_squared"] = (enriched["seed_diff"] ** 2).astype(float)
    enriched["log_abs_seed_diff"] = np.log1p(enriched["seed_diff"].abs()).astype(float)

    # --- Rating disagreement features (Step 3) ---
    power_diff_columns = []
    for col_name in [
        "kenpom_badj_em_power_diff",
        "evanmiya_relative_rating_power_diff",
        "bpi_rank_percentile_diff",
        "massey_ordinal_rank_percentile_diff",
    ]:
        missing_col = f"missing_{col_name.removesuffix('_power_diff').removesuffix('_percentile_diff')}"
        if missing_col in enriched.columns:
            power_diff_columns.append(
                enriched[col_name].where(enriched[missing_col] == 0, np.nan)
            )
        else:
            power_diff_columns.append(enriched[col_name])
    diff_frame = pd.concat(power_diff_columns, axis=1)
    enriched["rating_disagreement"] = diff_frame.std(axis=1, skipna=True).fillna(0.0)
    enriched["max_min_rating_spread"] = (
        diff_frame.max(axis=1, skipna=True) - diff_frame.min(axis=1, skipna=True)
    ).fillna(0.0)

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default=str(PROCESSED_ROOT / "tournament_games.parquet"))
    parser.add_argument("--snapshots", default=str(PROCESSED_ROOT / "season_snapshots.parquet"))
    parser.add_argument("--output", default=str(PROCESSED_ROOT / "tournament_training_games.parquet"))
    args = parser.parse_args()

    ensure_dirs()
    games = pd.read_parquet(args.games)
    snapshots = pd.read_parquet(args.snapshots)

    unresolved_games = games[games["canonical_team_id"].isna() | games["opponent_team_id"].isna()]
    if not unresolved_games.empty:
        raise SystemExit("Unresolved tournament teams detected in games; review alias table before dataset build.")

    snapshot_summary = build_snapshot_summary(snapshots, games)
    features = games.merge(
        snapshot_summary.add_prefix("team_"),
        left_on=["season", "canonical_team_id"],
        right_on=["team_season", "team_team_id"],
        how="left",
    ).merge(
        snapshot_summary.add_prefix("opp_"),
        left_on=["season", "opponent_team_id"],
        right_on=["opp_season", "opp_team_id"],
        how="left",
    )

    features = apply_semantic_features(features)

    season_coverage = (
        features.assign(
            has_core_signal=(
                (features["missing_massey_ordinal_rank"] == 0) | (features["missing_kenpom_badj_em"] == 0)
            )
        )
        .groupby("season")["has_core_signal"]
        .mean()
        .reset_index(name="coverage_rate")
    )
    # Lowered from 0.9 to 0.65 to unlock all 9 configured seasons (2016-2019, 2021-2025).
    # The bottleneck is snapshot matching, not signal availability — Kaggle Massey ordinals
    # and KenPom are available for all seasons via the core signal check above.
    modeled_seasons = season_coverage[season_coverage["coverage_rate"] >= 0.65]["season"].tolist()
    if not modeled_seasons:
        raise SystemExit(
            "No seasons cleared the minimum predictive-source coverage threshold."
        )
    features = features[features["season"].isin(modeled_seasons)].copy()

    output_path = Path(args.output)
    if output_path.suffix == ".parquet":
        features.to_parquet(output_path, index=False)
    else:
        features.to_csv(output_path, index=False)

    print(f"wrote {len(features)} training rows to {output_path}")


if __name__ == "__main__":
    main()
