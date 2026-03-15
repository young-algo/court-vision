from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .common import MODEL_ROOT, PROCESSED_ROOT, rolling_pairs

CORE_CANDIDATE_SIGNAL_FEATURES = [
    "bpi_rank_percentile_diff",
    "net_rank_percentile_diff",
    "massey_ordinal_rank_percentile_diff",
    "kenpom_badj_em_power_diff",
    "evanmiya_relative_rating_power_diff",
    "fivethirtyeight_power_power_diff",
    "resume_rank_blend_percentile_diff",
]
EXPERIMENTAL_MASSEY_MASTER_SIGNAL_FEATURES = [
    "massey_composite_rank_percentile_diff",
    "massey_mean_rank_percentile_diff",
    "massey_median_rank_percentile_diff",
    "massey_rank_stdev_advantage",
    "massey_win_pct_power_diff",
]
ALL_CANDIDATE_SIGNAL_FEATURES = [
    *CORE_CANDIDATE_SIGNAL_FEATURES,
    *EXPERIMENTAL_MASSEY_MASTER_SIGNAL_FEATURES,
]
CORE_MISSING_FEATURES = [
    "missing_bpi_rank",
    "missing_net_rank",
    "missing_massey_ordinal_rank",
    "missing_kenpom_badj_em",
    "missing_evanmiya_relative_rating",
    "missing_fivethirtyeight_power",
    "missing_resume_rank_blend",
]
EXPERIMENTAL_MASSEY_MASTER_MISSING_FEATURES = [
    "missing_massey_composite_rank",
    "missing_massey_mean_rank",
    "missing_massey_median_rank",
    "missing_massey_rank_stdev",
    "missing_massey_win_pct",
]
COMMON_CANDIDATE_CONTEXT_FEATURES = [
    "seed_diff",
    "same_conference",
    "available_predictive_rank_count",
    "available_power_feature_count",
    "available_resume_feature_count",
]
CANDIDATE_SIGNAL_FEATURES = CORE_CANDIDATE_SIGNAL_FEATURES
CANDIDATE_FEATURES = [
    *CORE_CANDIDATE_SIGNAL_FEATURES,
    *COMMON_CANDIDATE_CONTEXT_FEATURES,
    *CORE_MISSING_FEATURES,
]
EXPERIMENTAL_CANDIDATE_FEATURES = [
    *ALL_CANDIDATE_SIGNAL_FEATURES,
    *COMMON_CANDIDATE_CONTEXT_FEATURES,
    *CORE_MISSING_FEATURES,
    *EXPERIMENTAL_MASSEY_MASTER_MISSING_FEATURES,
]
LEGACY_SOURCE_FEATURES = [
    "torvik_efficiency",
    "torvik_barthag",
    "bpi",
    "net",
    "resume_form",
]
LEGACY_FEATURES = [
    *LEGACY_SOURCE_FEATURES,
    "seed_diff",
    "same_conference",
    "legacy_source_coverage_count",
]
PROMOTION_BASELINES = ("equal_weight_consensus", "seed_only_logit")
LOWER_IS_BETTER = {"logLoss", "brierScore", "marginMae", "calibrationError"}
HIGHER_IS_BETTER = {"upsetRecall"}


@dataclass
class ModelBundle:
    feature_names: list[str]
    scaler: StandardScaler
    margin_model: Ridge
    win_model: LogisticRegression
    calibrator: IsotonicRegression | None


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in columns:
        if column in prepared.columns:
            series = pd.to_numeric(prepared[column], errors="coerce")
        else:
            default_value = 1.0 if column.startswith("missing_") else 0.0
            series = pd.Series(default_value, index=prepared.index, dtype=float)
        fill_value = 1.0 if column.startswith("missing_") else 0.0
        prepared[column] = series.fillna(fill_value)
    return prepared


def candidate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return numeric_frame(frame, CANDIDATE_FEATURES)[CANDIDATE_FEATURES]


def experimental_candidate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return numeric_frame(frame, EXPERIMENTAL_CANDIDATE_FEATURES)[EXPERIMENTAL_CANDIDATE_FEATURES]


def legacy_prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    legacy_inputs = [
        "team_legacy_bpi_value",
        "team_legacy_net_rank",
        "team_legacy_massey_value",
        "opp_legacy_bpi_value",
        "opp_legacy_net_rank",
        "opp_legacy_massey_value",
        "seed_diff",
        "same_conference",
        "legacy_source_coverage_count",
    ]
    for column in legacy_inputs:
        prepared[column] = pd.to_numeric(prepared.get(column), errors="coerce").fillna(0.0)

    prepared["bpi"] = prepared["team_legacy_bpi_value"] - prepared["opp_legacy_bpi_value"]
    prepared["net"] = prepared["opp_legacy_net_rank"] - prepared["team_legacy_net_rank"]
    prepared["massey"] = prepared["team_legacy_massey_value"] - prepared["opp_legacy_massey_value"]
    prepared["resume_form"] = prepared["legacy_source_coverage_count"]
    prepared["torvik_efficiency"] = prepared["bpi"] * 0.92 + prepared["massey"] * 0.08
    prepared["torvik_barthag"] = prepared["torvik_efficiency"] * 0.68
    return prepared


def legacy_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = legacy_prepare_features(frame)
    return prepared[LEGACY_FEATURES]


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 0.01, 0.99)
    return np.log(clipped / (1 - clipped))


def fit_calibrator(raw_probabilities: np.ndarray, y_true: pd.Series) -> IsotonicRegression | None:
    if len(np.unique(y_true)) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probabilities, y_true)
    return calibrator


def fit_model_bundle(features: pd.DataFrame, y_margin: pd.Series, y_win: pd.Series) -> ModelBundle:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    margin_model = Ridge(alpha=1.0).fit(scaled, y_margin)
    win_model = LogisticRegression(max_iter=4000).fit(scaled, y_win)
    raw_probabilities = win_model.predict_proba(scaled)[:, 1]
    calibrator = fit_calibrator(raw_probabilities, y_win)
    return ModelBundle(
        feature_names=list(features.columns),
        scaler=scaler,
        margin_model=margin_model,
        win_model=win_model,
        calibrator=calibrator,
    )


def predict_bundle(bundle: ModelBundle, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = bundle.scaler.transform(features[bundle.feature_names])
    predicted_margin = bundle.margin_model.predict(scaled)
    raw_probability = bundle.win_model.predict_proba(scaled)[:, 1]
    calibrated_probability = (
        bundle.calibrator.predict(raw_probability) if bundle.calibrator else raw_probability
    )
    return predicted_margin, np.clip(raw_probability, 0.01, 0.99), np.clip(calibrated_probability, 0.01, 0.99)


def expected_calibration_error(y_true: pd.Series, probabilities: np.ndarray, buckets: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    frame = pd.DataFrame({"actual": y_true.astype(float), "probability": probabilities})
    frame["bucket"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0, 1, buckets + 1),
        include_lowest=True,
        labels=False,
    )
    bucket_stats = frame.groupby("bucket", observed=True).agg(
        count=("actual", "size"),
        actual_rate=("actual", "mean"),
        probability_rate=("probability", "mean"),
    )
    weights = bucket_stats["count"] / bucket_stats["count"].sum()
    return float((weights * (bucket_stats["actual_rate"] - bucket_stats["probability_rate"]).abs()).sum())


def upset_recall(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    mask = (frame["seed_diff"] > 0) & (frame["team_win"] == 1)
    if not mask.any():
        return 0.0
    return float((probabilities[mask] >= 0.5).mean())


def metrics_for_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    raw_probabilities: np.ndarray,
    predicted_margin: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(probabilities, 0.01, 0.99)
    return {
        "logLoss": float(log_loss(frame["team_win"], clipped, labels=[0, 1])),
        "brierScore": float(brier_score_loss(frame["team_win"], clipped)),
        "marginMae": float(mean_absolute_error(frame["margin"], predicted_margin)),
        "calibrationError": expected_calibration_error(frame["team_win"], clipped),
        "upsetRecall": upset_recall(frame, clipped),
        "meanCalibrationShift": float(np.mean(np.abs(clipped - raw_probabilities))),
    }


def signal_missing_column(signal_feature: str) -> str:
    if signal_feature.endswith("_power_diff"):
        return f"missing_{signal_feature.removesuffix('_power_diff')}"
    if signal_feature.endswith("_percentile_diff"):
        return f"missing_{signal_feature.removesuffix('_percentile_diff')}"
    if signal_feature.endswith("_advantage"):
        return f"missing_{signal_feature.removesuffix('_advantage')}"
    raise ValueError(f"Unknown signal feature: {signal_feature}")


def standardized_signal(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    feature: str,
) -> tuple[pd.Series, pd.Series]:
    missing_column = signal_missing_column(feature)
    train_values = pd.to_numeric(train[feature], errors="coerce").fillna(0.0)
    holdout_values = pd.to_numeric(holdout[feature], errors="coerce").fillna(0.0)
    mean = float(train_values.mean())
    spread = float(train_values.std(ddof=0)) or 1.0
    train_scaled = ((train_values - mean) / spread) * (1 - train[missing_column])
    holdout_scaled = ((holdout_values - mean) / spread) * (1 - holdout[missing_column])
    return train_scaled, holdout_scaled


def equal_weight_consensus_frames(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_scaled_signals: list[pd.Series] = []
    holdout_scaled_signals: list[pd.Series] = []
    availability_columns = []

    for feature in CANDIDATE_SIGNAL_FEATURES:
        train_signal, holdout_signal = standardized_signal(train, holdout, feature)
        train_scaled_signals.append(train_signal)
        holdout_scaled_signals.append(holdout_signal)
        availability_columns.append(1 - train[signal_missing_column(feature)])

    train_available = pd.concat(availability_columns, axis=1).sum(axis=1).replace(0, 1)
    holdout_available = pd.concat(
        [(1 - holdout[signal_missing_column(feature)]) for feature in CANDIDATE_SIGNAL_FEATURES],
        axis=1,
    ).sum(axis=1).replace(0, 1)

    train_consensus = pd.concat(train_scaled_signals, axis=1).sum(axis=1) / train_available
    holdout_consensus = pd.concat(holdout_scaled_signals, axis=1).sum(axis=1) / holdout_available
    return (
        pd.DataFrame({"consensus_score": train_consensus}),
        pd.DataFrame({"consensus_score": holdout_consensus}),
    )


def best_power_feature_for_holdout(holdout: pd.DataFrame) -> str:
    priorities = [
        "kenpom_badj_em_power_diff",
        "evanmiya_relative_rating_power_diff",
        "fivethirtyeight_power_power_diff",
        "massey_win_pct_power_diff",
        "massey_ordinal_rank_percentile_diff",
    ]
    for feature in priorities:
        missing_column = signal_missing_column(feature)
        if missing_column in holdout.columns and (holdout[missing_column] == 0).any():
            return feature
    return priorities[-1]


def single_signal_frame(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    missing_column = signal_missing_column(feature)
    return numeric_frame(frame, [feature, missing_column])[[feature, missing_column]]


def model_prediction_frame(
    holdout: pd.DataFrame,
    model_name: str,
    predicted_margin: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    baseline_detail: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": holdout["season"].to_numpy(),
            "model": model_name,
            "baselineDetail": baseline_detail,
            "team_win": holdout["team_win"].to_numpy(),
            "margin": holdout["margin"].to_numpy(),
            "seed_diff": holdout["seed_diff"].to_numpy(),
            "predicted_margin": predicted_margin,
            "raw_probability": raw_probability,
            "probability": calibrated_probability,
        }
    )


def fit_and_predict(
    model_name: str,
    train_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    baseline_detail: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    bundle = fit_model_bundle(train_features, train["margin"], train["team_win"])
    predicted_margin, raw_probability, calibrated_probability = predict_bundle(bundle, holdout_features)
    metrics = metrics_for_predictions(holdout, calibrated_probability, raw_probability, predicted_margin)
    metrics["holdoutSeason"] = int(holdout["season"].iloc[0])
    metrics["model"] = model_name
    if baseline_detail:
        metrics["baselineDetail"] = baseline_detail
    predictions = model_prediction_frame(
        holdout,
        model_name,
        predicted_margin,
        raw_probability,
        calibrated_probability,
        baseline_detail,
    )
    return metrics, predictions


def metric_winners(metrics_by_model: dict[str, dict[str, float]]) -> dict[str, str]:
    winners: dict[str, str] = {}
    for metric in [*LOWER_IS_BETTER, *HIGHER_IS_BETTER]:
        if metric in LOWER_IS_BETTER:
            winners[metric] = min(
                metrics_by_model.items(),
                key=lambda item: item[1][metric],
            )[0]
        else:
            winners[metric] = max(
                metrics_by_model.items(),
                key=lambda item: item[1][metric],
            )[0]
    return winners


def calibration_buckets(predictions: pd.DataFrame, buckets: int = 10) -> list[dict[str, float]]:
    if predictions.empty:
        return []
    frame = predictions.copy()
    frame["bucket"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0, 1, buckets + 1),
        include_lowest=True,
        labels=False,
    )
    stats = frame.groupby("bucket", observed=True).agg(
        count=("team_win", "size"),
        actual_rate=("team_win", "mean"),
        probability_rate=("probability", "mean"),
    )
    return [
        {
            "bucket": int(index),
            "count": int(row["count"]),
            "actualRate": float(row["actual_rate"]),
            "probabilityRate": float(row["probability_rate"]),
        }
        for index, row in stats.iterrows()
    ]


def slice_mask(frame: pd.DataFrame, slice_name: str) -> pd.Series:
    if slice_name == "upset_games":
        return ((frame["seed_diff"] > 0) & (frame["team_win"] == 1)) | (
            (frame["seed_diff"] < 0) & (frame["team_win"] == 0)
        )
    if slice_name == "one_possession_games":
        return frame["margin"].abs() <= 3
    if slice_name == "large_seed_gap_games":
        return frame["seed_diff"].abs() >= 5
    raise ValueError(f"Unknown slice: {slice_name}")


def slice_report(predictions_by_model: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    slices: dict[str, dict[str, object]] = {}
    for slice_name in ("upset_games", "one_possession_games", "large_seed_gap_games"):
        metrics_by_model: dict[str, dict[str, float]] = {}
        for model_name, predictions in predictions_by_model.items():
            subset = predictions[slice_mask(predictions, slice_name)].copy()
            if subset.empty:
                continue
            metrics_by_model[model_name] = metrics_for_predictions(
                subset,
                subset["probability"].to_numpy(),
                subset["raw_probability"].to_numpy(),
                subset["predicted_margin"].to_numpy(),
            )
        slices[slice_name] = {
            "metrics": metrics_by_model,
            "winner": metric_winners(metrics_by_model) if metrics_by_model else {},
        }
    return slices


def feature_semantics() -> list[dict[str, str]]:
    semantics: list[dict[str, str]] = []
    for feature in ALL_CANDIDATE_SIGNAL_FEATURES:
        family = "predictive_ratings" if feature.endswith("_power_diff") else "rank_transforms"
        direction = "higher_is_better"
        if feature == "resume_rank_blend_percentile_diff":
            family = "resume_context"
        if feature.startswith("massey_") and feature.endswith("_advantage"):
            family = "consensus_dispersion"
        if feature in EXPERIMENTAL_MASSEY_MASTER_SIGNAL_FEATURES:
            family = f"experimental_{family}"
        semantics.append({"feature": feature, "family": family, "direction": direction})
    context_features = {
        "seed_diff": ("resume_context", "lower_is_better"),
        "same_conference": ("resume_context", "higher_is_better"),
        "available_predictive_rank_count": ("missingness", "higher_is_better"),
        "available_power_feature_count": ("missingness", "higher_is_better"),
        "available_resume_feature_count": ("missingness", "higher_is_better"),
    }
    missing_features = [
        "missing_bpi_rank",
        "missing_net_rank",
        "missing_massey_ordinal_rank",
        "missing_massey_composite_rank",
        "missing_massey_mean_rank",
        "missing_massey_median_rank",
        "missing_massey_rank_stdev",
        "missing_kenpom_badj_em",
        "missing_evanmiya_relative_rating",
        "missing_fivethirtyeight_power",
        "missing_massey_win_pct",
        "missing_resume_rank_blend",
    ]
    for feature, (family, direction) in context_features.items():
        semantics.append({"feature": feature, "family": family, "direction": direction})
    for feature in missing_features:
        semantics.append({"feature": feature, "family": "missingness", "direction": "lower_is_better"})
    return semantics


def source_coverage(frame: pd.DataFrame) -> dict[str, object]:
    source_map = {
        "bpi": ["missing_bpi_rank"],
        "net": ["missing_net_rank"],
        "massey_ordinal": ["missing_massey_ordinal_rank"],
        "massey_master": [
            "missing_massey_composite_rank",
            "missing_massey_mean_rank",
            "missing_massey_median_rank",
            "missing_massey_rank_stdev",
            "missing_massey_win_pct",
        ],
        "kenpom_badj_em": ["missing_kenpom_badj_em"],
        "evanmiya_relative_rating": ["missing_evanmiya_relative_rating"],
        "fivethirtyeight_power": ["missing_fivethirtyeight_power"],
        "resume_rank_blend": ["missing_resume_rank_blend"],
    }
    by_source: dict[str, list[int]] = {}
    summary: list[str] = []
    for label, missing_columns in source_map.items():
        seasons = sorted(
            frame.groupby("season").apply(
                lambda group: any(
                    missing_column in group.columns and bool((group[missing_column] == 0).any())
                    for missing_column in missing_columns
                ),
                include_groups=False,
            ).pipe(lambda series: series[series].index.tolist())
        )
        by_source[label] = [int(season) for season in seasons]
        summary.append(f"{label}: {len(seasons)} seasons")
    return {"summary": summary, "seasonsBySource": by_source}


def source_weights_from_candidate(bundle: ModelBundle) -> dict[str, float]:
    coefficients = bundle.margin_model.coef_
    weight_map = {
        feature: abs(float(coefficients[index]))
        for index, feature in enumerate(bundle.feature_names)
        if feature in CORE_CANDIDATE_SIGNAL_FEATURES
    }
    total = sum(weight_map.values()) or 1.0
    return {feature: round(value / total, 4) for feature, value in weight_map.items()}


def candidate_feature_importance(bundle: ModelBundle) -> list[dict[str, float | str]]:
    importance = []
    for index, feature in enumerate(bundle.feature_names):
        importance.append(
            {
                "feature": feature,
                "marginCoefficient": float(bundle.margin_model.coef_[index]),
                "winCoefficient": float(bundle.win_model.coef_[0][index]),
                "absoluteMagnitude": float(
                    abs(bundle.margin_model.coef_[index]) + abs(bundle.win_model.coef_[0][index])
                ),
            }
        )
    return sorted(importance, key=lambda row: row["absoluteMagnitude"], reverse=True)


def calibration_anchor_rows(
    raw_probability: np.ndarray,
    calibrator: IsotonicRegression | None,
) -> list[dict[str, float]]:
    quantiles = pd.Series(raw_probability).quantile([0.05, 0.15, 0.25, 0.5, 0.75, 0.9]).to_numpy()
    calibrated = calibrator.predict(quantiles) if calibrator else quantiles
    return [
        {"raw": float(raw), "calibrated": float(cal)}
        for raw, cal in zip(quantiles, calibrated)
    ]


def candidate_artifact(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    aggregate_metrics: dict[str, float],
    training_seasons: list[int],
    artifact_path: str,
) -> dict[str, object]:
    all_features = candidate_feature_frame(frame)
    _, raw_probability, _ = predict_bundle(bundle, all_features)
    return {
        "id": "tournament-consensus-candidate",
        "label": "Learned Tournament Consensus Candidate",
        "model_type": "learned_tournament_consensus",
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "snapshot_policy": "selection_sunday_latest_stable",
        "training_seasons": training_seasons,
        "data_sources": [
            "Kaggle MMasseyOrdinals",
            "Local Massey master composite archive",
            "Archive team sheet ranks",
            "Archive KenPom/Barttorvik",
            "Archive EvanMiya",
            "Archive 538 ratings",
        ],
        "source_coverage": source_coverage(frame),
        "source_weights": source_weights_from_candidate(bundle),
        "margin_model": {
            "intercept": float(bundle.margin_model.intercept_),
            "coefficients": {
                feature: float(bundle.margin_model.coef_[index])
                for index, feature in enumerate(bundle.feature_names)
            },
        },
        "win_model": {
            "intercept": float(bundle.win_model.intercept_[0]),
            "coefficients": {
                feature: float(bundle.win_model.coef_[0][index])
                for index, feature in enumerate(bundle.feature_names)
            },
        },
        "calibration": {"anchors": calibration_anchor_rows(raw_probability, bundle.calibrator)},
        "metrics": aggregate_metrics,
        "artifact_path": artifact_path,
        "notes": "Training-only candidate artifact built from source-native feature semantics. Not promoted by default.",
        "runtimeCompatible": False,
        "featureSemantics": feature_semantics(),
    }


def is_runtime_compatible(artifact: dict[str, object]) -> bool:
    runtime_features = {
        "torvik_efficiency",
        "torvik_barthag",
        "bpi",
        "net",
        "resume_form",
        "seed_diff",
        "same_conference",
        "observed_source_count",
        "source_consensus_margin",
        "latent_consensus_margin",
    }
    margin_keys = set(artifact["margin_model"]["coefficients"].keys())
    win_keys = set(artifact["win_model"]["coefficients"].keys())
    return margin_keys.issubset(runtime_features) and win_keys.issubset(runtime_features)


def should_promote(
    candidate_metrics: dict[str, float],
    baselines: dict[str, dict[str, float]],
) -> bool:
    for baseline in PROMOTION_BASELINES:
        if baseline not in baselines:
            return False
        if candidate_metrics["logLoss"] >= baselines[baseline]["logLoss"]:
            return False
        if candidate_metrics["brierScore"] >= baselines[baseline]["brierScore"]:
            return False
    return True


def rolling_backtest(frame: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame]]:
    metrics_rows: list[dict[str, object]] = []
    predictions: dict[str, list[pd.DataFrame]] = {}
    seasons = sorted(frame["season"].dropna().astype(int).unique())

    for train_seasons, holdout_season in rolling_pairs(seasons):
        train = frame[frame["season"].isin(train_seasons)].copy()
        holdout = frame[frame["season"] == holdout_season].copy()
        if train.empty or holdout.empty:
            continue

        candidate_metrics, candidate_predictions = fit_and_predict(
            "candidate_stack",
            candidate_feature_frame(train),
            candidate_feature_frame(holdout),
            train,
            holdout,
        )
        metrics_rows.append(candidate_metrics)
        predictions.setdefault("candidate_stack", []).append(candidate_predictions)

        experimental_metrics, experimental_predictions = fit_and_predict(
            "candidate_stack_with_massey_master",
            experimental_candidate_feature_frame(train),
            experimental_candidate_feature_frame(holdout),
            train,
            holdout,
        )
        metrics_rows.append(experimental_metrics)
        predictions.setdefault("candidate_stack_with_massey_master", []).append(experimental_predictions)

        seed_metrics, seed_predictions = fit_and_predict(
            "seed_only_logit",
            numeric_frame(train, ["seed_diff"])[["seed_diff"]],
            numeric_frame(holdout, ["seed_diff"])[["seed_diff"]],
            train,
            holdout,
        )
        metrics_rows.append(seed_metrics)
        predictions.setdefault("seed_only_logit", []).append(seed_predictions)

        equal_train, equal_holdout = equal_weight_consensus_frames(train, holdout)
        equal_metrics, equal_predictions = fit_and_predict(
            "equal_weight_consensus",
            equal_train,
            equal_holdout,
            train,
            holdout,
        )
        metrics_rows.append(equal_metrics)
        predictions.setdefault("equal_weight_consensus", []).append(equal_predictions)

        bpi_metrics, bpi_predictions = fit_and_predict(
            "best_single_source_bpi",
            single_signal_frame(train, "bpi_rank_percentile_diff"),
            single_signal_frame(holdout, "bpi_rank_percentile_diff"),
            train,
            holdout,
        )
        metrics_rows.append(bpi_metrics)
        predictions.setdefault("best_single_source_bpi", []).append(bpi_predictions)

        net_metrics, net_predictions = fit_and_predict(
            "best_single_source_net",
            single_signal_frame(train, "net_rank_percentile_diff"),
            single_signal_frame(holdout, "net_rank_percentile_diff"),
            train,
            holdout,
        )
        metrics_rows.append(net_metrics)
        predictions.setdefault("best_single_source_net", []).append(net_predictions)

        power_feature = best_power_feature_for_holdout(holdout)
        power_metrics, power_predictions = fit_and_predict(
            "best_single_source_power",
            single_signal_frame(train, power_feature),
            single_signal_frame(holdout, power_feature),
            train,
            holdout,
            baseline_detail=power_feature,
        )
        metrics_rows.append(power_metrics)
        predictions.setdefault("best_single_source_power", []).append(power_predictions)

        compat_metrics, compat_predictions = fit_and_predict(
            "current_stack_compat",
            legacy_feature_frame(train),
            legacy_feature_frame(holdout),
            train,
            holdout,
        )
        metrics_rows.append(compat_metrics)
        predictions.setdefault("current_stack_compat", []).append(compat_predictions)

    compiled_predictions = {
        model_name: pd.concat(model_frames, ignore_index=True)
        for model_name, model_frames in predictions.items()
    }
    return metrics_rows, compiled_predictions


def aggregate_metrics(backtest_rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for model_name, group in backtest_rows.groupby("model"):
        aggregates[model_name] = {
            "logLoss": float(group["logLoss"].mean()),
            "brierScore": float(group["brierScore"].mean()),
            "marginMae": float(group["marginMae"].mean()),
            "calibrationError": float(group["calibrationError"].mean()),
            "upsetRecall": float(group["upsetRecall"].mean()),
        }
    return aggregates


def print_summary(aggregates: dict[str, dict[str, float]]) -> None:
    summary = (
        pd.DataFrame.from_dict(aggregates, orient="index")
        .sort_values(["logLoss", "brierScore", "marginMae"])
        .round(4)
    )
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(PROCESSED_ROOT / "tournament_training_games.parquet"))
    parser.add_argument("--output", default=str(MODEL_ROOT / "tournament-consensus-candidate.json"))
    parser.add_argument("--report-output", default=str(MODEL_ROOT / "tournament-consensus-report.json"))
    parser.add_argument(
        "--backtest-output",
        default=str(PROCESSED_ROOT / "tournament_backtest_by_season.csv"),
    )
    parser.add_argument("--latest-output", default=str(MODEL_ROOT / "tournament-consensus-latest.json"))
    parser.add_argument("--promote-if-best", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(args.dataset).copy()
    frame = numeric_frame(
        frame,
        [
            *CANDIDATE_FEATURES,
            "margin",
            "team_win",
            "seed_diff",
            "same_conference",
            "legacy_source_coverage_count",
            "team_legacy_bpi_value",
            "team_legacy_net_rank",
            "team_legacy_massey_value",
            "opp_legacy_bpi_value",
            "opp_legacy_net_rank",
            "opp_legacy_massey_value",
        ],
    )

    backtest_rows, predictions_by_model = rolling_backtest(frame)
    if not backtest_rows:
        raise SystemExit("Not enough seasonal coverage to run rolling validation.")

    backtest_frame = pd.DataFrame(backtest_rows)
    aggregate = aggregate_metrics(backtest_frame)
    print_summary(aggregate)

    candidate_bundle = fit_model_bundle(candidate_feature_frame(frame), frame["margin"], frame["team_win"])
    candidate_metrics = aggregate["candidate_stack"]
    training_seasons = sorted(frame["season"].dropna().astype(int).unique().tolist())
    candidate_payload = candidate_artifact(
        candidate_bundle,
        frame,
        candidate_metrics,
        training_seasons,
        "data/models/tournament-consensus-candidate.json",
    )

    report = {
        "generatedAt": pd.Timestamp.now("UTC").isoformat(),
        "datasetCoverage": {
            "rowCount": int(len(frame)),
            "seasonCount": int(frame["season"].nunique()),
            "modeledSeasons": training_seasons,
            "sourceCoverage": source_coverage(frame),
        },
        "aggregateMetrics": aggregate,
        "aggregateWinners": metric_winners(aggregate),
        "backtestBySeason": json.loads(backtest_frame.to_json(orient="records")),
        "calibrationBuckets": {
            model_name: calibration_buckets(predictions)
            for model_name, predictions in predictions_by_model.items()
        },
        "featureImportance": candidate_feature_importance(candidate_bundle),
        "featureSemantics": feature_semantics(),
        "errorSlices": slice_report(predictions_by_model),
        "promotion": {
            "requested": bool(args.promote_if_best),
            "eligibleByMetrics": should_promote(
                candidate_metrics,
                {name: aggregate[name] for name in PROMOTION_BASELINES if name in aggregate},
            ),
            "runtimeCompatible": is_runtime_compatible(candidate_payload),
            "latestArtifactUpdated": False,
        },
    }

    output_path = Path(args.output)
    report_path = Path(args.report_output)
    backtest_path = Path(args.backtest_output)
    latest_path = Path(args.latest_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    backtest_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(candidate_payload, indent=2))
    report_path.write_text(json.dumps(report, indent=2))
    backtest_frame.to_csv(backtest_path, index=False)

    if args.promote_if_best:
        eligible = report["promotion"]["eligibleByMetrics"]
        runtime_compatible = report["promotion"]["runtimeCompatible"]
        if eligible and runtime_compatible:
            shutil.copyfile(output_path, latest_path)
            report["promotion"]["latestArtifactUpdated"] = True
            report_path.write_text(json.dumps(report, indent=2))

    print(f"wrote candidate artifact to {output_path}")
    print(f"wrote diagnostic report to {report_path}")
    print(f"wrote backtest output to {backtest_path}")


if __name__ == "__main__":
    main()
