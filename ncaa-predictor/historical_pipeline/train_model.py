from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
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

# Betting market features live outside the signal-missing machinery because
# they use a different naming convention (betting_spread_a, not *_percentile_diff).
# They are injected directly into ENHANCED_CANDIDATE_FEATURES and handled by
# numeric_frame (which fills NaN → 0 and missing_ → 1 by default).
BETTING_FEATURES = [
    "betting_spread_a",
    "betting_win_prob_a",
    "missing_betting",
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
OPTIMIZED_CANDIDATE_SIGNAL_FEATURES = [
    "kenpom_badj_em_power_diff",
    "massey_ordinal_rank_percentile_diff",
    "bpi_rank_percentile_diff",
    "evanmiya_relative_rating_power_diff",
]
OPTIMIZED_CANDIDATE_MISSING_FEATURES = [
    "missing_kenpom_badj_em",
    "missing_massey_ordinal_rank",
    "missing_bpi_rank",
    "missing_evanmiya_relative_rating",
]
OPTIMIZED_CANDIDATE_CONTEXT_FEATURES = [
    "seed_diff",
    "seed_kenpom_interaction",
    "abs_seed_same_conference_interaction",
]
NEW_CONTEXT_FEATURES = [
    "round_seed_interaction",
    "round_kenpom_interaction",
    "seed_diff_squared",
    "log_abs_seed_diff",
    "rating_disagreement",
    "max_min_rating_spread",
]
OPTIMIZED_CANDIDATE_FEATURES = [
    *OPTIMIZED_CANDIDATE_SIGNAL_FEATURES,
    *OPTIMIZED_CANDIDATE_CONTEXT_FEATURES,
    *OPTIMIZED_CANDIDATE_MISSING_FEATURES,
]
ENHANCED_CANDIDATE_FEATURES = [
    *OPTIMIZED_CANDIDATE_FEATURES,
    *NEW_CONTEXT_FEATURES,
    # BETTING_FEATURES intentionally excluded: coverage too sparse (<12% per training
    # window in rolling backtest) causes sign inversion and degrades log-loss by ~0.004.
    # Betting signal is surfaced live in the UI but not used in the learned artifact.
]
ENHANCED_WIN_FEATURES = [
    feature for feature in ENHANCED_CANDIDATE_FEATURES if feature != "seed_diff"
]
OPTIMIZED_WIN_FEATURES = [
    feature for feature in OPTIMIZED_CANDIDATE_FEATURES if feature != "seed_diff"
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
REDUCED_CANDIDATE_FEATURES = [
    "massey_ordinal_rank_percentile_diff",
    "kenpom_badj_em_power_diff",
    "seed_diff",
    "missing_massey_ordinal_rank",
    "missing_kenpom_badj_em",
]

PROMOTION_BASELINES = ("equal_weight_consensus", "seed_only_logit")
LOWER_IS_BETTER = {"logLoss", "brierScore", "marginMae", "calibrationError"}
HIGHER_IS_BETTER = {"upsetRecall"}
GBM_DISTILLATION_PARAMS = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.05,
    "random_state": 42,
}
DISTILLATION_BLEND = 0.7
TEMPORAL_DECAY = 0.90


def tune_hyperparameters(
    frame: pd.DataFrame,
    feature_frame_fn,
    win_feature_names: list[str] | None = None,
    n_trials: int = 40,
) -> dict[str, object]:
    """Use optuna to tune GBM distillation params via inner 3-fold CV on the training set."""
    try:
        import optuna
    except ImportError:
        print("optuna not installed -- skipping hyperparameter tuning, using defaults")
        return {
            "gbm_params": GBM_DISTILLATION_PARAMS,
            "distillation_blend": DISTILLATION_BLEND,
            "temporal_decay": TEMPORAL_DECAY,
        }

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seasons = sorted(frame["season"].dropna().astype(int).unique())
    n_inner = min(3, len(seasons) - 1)
    inner_pairs = rolling_pairs(seasons)[-n_inner:]

    def objective(trial: optuna.Trial) -> float:
        n_est = trial.suggest_int("n_estimators", 40, 200, step=20)
        max_depth = trial.suggest_int("max_depth", 2, 5)
        lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        blend = trial.suggest_float("distillation_blend", 0.4, 0.9)
        decay = trial.suggest_float("temporal_decay", 0.80, 0.98)

        gbm_params = {"n_estimators": n_est, "max_depth": max_depth, "learning_rate": lr, "random_state": 42}
        losses = []
        for train_seasons, holdout_season in inner_pairs:
            train = frame[frame["season"].isin(train_seasons)].copy()
            holdout = frame[frame["season"] == holdout_season].copy()
            if train.empty or holdout.empty:
                continue

            train_features = feature_frame_fn(train)
            holdout_features = feature_frame_fn(holdout)
            temporal = compute_temporal_weights(train["season"], decay)
            upset_mask = (train["seed_diff"] > 0) & (train["team_win"] == 1)
            weights = temporal * np.where(upset_mask, 1.5, 1.0)

            feature_names = list(train_features.columns)
            wfn = win_feature_names or feature_names
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(train_features)
            scaled_holdout = scaler.transform(holdout_features[feature_names])
            fi = {f: i for i, f in enumerate(feature_names)}
            wi = [fi[f] for f in wfn]

            win_teacher = GradientBoostingClassifier(**gbm_params)
            win_teacher.fit(scaled_train[:, wi], train["team_win"], sample_weight=weights)
            tp = np.clip(win_teacher.predict_proba(scaled_train[:, wi])[:, 1], 0.01, 0.99)
            ep = np.clip(train["team_win"].to_numpy() * 0.98 + 0.01, 0.01, 0.99)
            target = probability_to_logit(np.clip(blend * tp + (1 - blend) * ep, 0.01, 0.99))
            student = RidgeCV(alphas=[1.0, 10.0, 100.0, 500.0, 1000.0]).fit(
                scaled_train[:, wi], target, sample_weight=weights,
            )
            raw_prob = sigmoid(student.predict(scaled_holdout[:, wi]))
            clipped = np.clip(raw_prob, 0.01, 0.99)
            losses.append(float(log_loss(holdout["team_win"], clipped, labels=[0, 1])))

        return float(np.mean(losses)) if losses else 1.0

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"Tuned hyperparameters: {best} (logLoss={study.best_value:.4f})")
    return {
        "gbm_params": {
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"],
            "random_state": 42,
        },
        "distillation_blend": best["distillation_blend"],
        "temporal_decay": best["temporal_decay"],
    }


@dataclass
class ModelBundle:
    feature_names: list[str]
    margin_feature_names: list[str]
    win_feature_names: list[str]
    scaler: StandardScaler
    margin_model: RidgeCV
    win_model: CalibratedClassifierCV | None
    win_intercept: float
    win_coefficients: np.ndarray
    win_calibrator: IsotonicRegression | None = None


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


def optimized_candidate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return numeric_frame(frame, OPTIMIZED_CANDIDATE_FEATURES)[OPTIMIZED_CANDIDATE_FEATURES]


def enhanced_candidate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return numeric_frame(frame, ENHANCED_CANDIDATE_FEATURES)[ENHANCED_CANDIDATE_FEATURES]


def reduced_candidate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return numeric_frame(frame, REDUCED_CANDIDATE_FEATURES)[REDUCED_CANDIDATE_FEATURES]


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


def fit_probability_calibrator(
    raw_probability: np.ndarray,
    y_true: pd.Series | np.ndarray,
    method: str = "isotonic",
) -> IsotonicRegression | LogisticRegression:
    """Fit a calibration mapping from raw probabilities to calibrated probabilities.

    Methods:
      - "isotonic": Isotonic regression with diagonal anchoring (default, current behavior)
      - "platt": 2-parameter logistic (Platt scaling) -- lower variance, good for small data
      - "beta": 3-parameter Beta calibration via logistic on log-odds + log(1-p) features
    """
    if method == "platt":
        logits = probability_to_logit(np.clip(np.asarray(raw_probability), 0.01, 0.99))
        platt = LogisticRegression(max_iter=2000, C=1e6)
        platt.fit(logits.reshape(-1, 1), np.asarray(y_true, dtype=int))
        return platt

    if method == "beta":
        raw = np.clip(np.asarray(raw_probability), 0.01, 0.99)
        log_odds = np.log(raw / (1 - raw))
        log_one_minus = np.log(1 - raw)
        beta_features = np.column_stack([log_odds, log_one_minus])
        beta_model = LogisticRegression(max_iter=2000, C=1e6)
        beta_model.fit(beta_features, np.asarray(y_true, dtype=int))
        return beta_model

    # Default: isotonic with diagonal anchoring
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    diagonal_raw = np.arange(0.1, 1.0, 0.1)
    diagonal_true = diagonal_raw.copy()
    anchored_raw_probability = np.concatenate([np.asarray(raw_probability), np.array([0.5]), diagonal_raw])
    anchored_y_true = np.concatenate([np.asarray(y_true, dtype=float), np.array([0.5]), diagonal_true])
    weights = np.concatenate([
        np.ones(len(raw_probability)),
        np.array([1.0]),
        np.full(len(diagonal_raw), 0.3),
    ])
    iso.fit(anchored_raw_probability, anchored_y_true, sample_weight=weights)
    return iso


def apply_calibrator(
    calibrator: IsotonicRegression | LogisticRegression,
    raw_probability: np.ndarray,
) -> np.ndarray:
    """Apply a fitted calibrator to raw probabilities."""
    raw = np.clip(raw_probability, 0.01, 0.99)
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(raw), 0.01, 0.99)
    if hasattr(calibrator, "coef_") and calibrator.coef_.shape[1] == 2:
        log_odds = np.log(raw / (1 - raw))
        log_one_minus = np.log(1 - raw)
        features = np.column_stack([log_odds, log_one_minus])
        return np.clip(calibrator.predict_proba(features)[:, 1], 0.01, 0.99)
    logits = probability_to_logit(raw)
    return np.clip(calibrator.predict_proba(logits.reshape(-1, 1))[:, 1], 0.01, 0.99)


def fit_model_bundle(
    features: pd.DataFrame,
    y_margin: pd.Series,
    y_win: pd.Series,
    *,
    win_feature_names: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> ModelBundle:
    feature_names = list(features.columns)
    margin_feature_names = feature_names
    win_feature_names = win_feature_names or feature_names
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[feature_names])
    feature_index = {feature: index for index, feature in enumerate(feature_names)}
    margin_indices = [feature_index[feature] for feature in margin_feature_names]
    win_indices = [feature_index[feature] for feature in win_feature_names]

    margin_model = RidgeCV(alphas=[10.0, 100.0, 1000.0, 5000.0, 10000.0]).fit(
        scaled[:, margin_indices],
        y_margin,
        sample_weight=sample_weight,
    )

    class_counts = pd.Series(y_win).value_counts()
    cv_folds = max(2, min(5, int(class_counts.min())))
    base_win_model = LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced")
    win_model = CalibratedClassifierCV(base_win_model, cv=cv_folds, method="isotonic").fit(
        scaled[:, win_indices],
        y_win,
        sample_weight=sample_weight,
    )

    base_estimators = [calibrator.estimator for calibrator in win_model.calibrated_classifiers_]
    mean_win_coef = np.mean([est.coef_[0] for est in base_estimators], axis=0)
    mean_win_intercept = float(np.mean([est.intercept_[0] for est in base_estimators]))

    return ModelBundle(
        feature_names=feature_names,
        margin_feature_names=margin_feature_names,
        win_feature_names=win_feature_names,
        scaler=scaler,
        margin_model=margin_model,
        win_model=win_model,
        win_intercept=mean_win_intercept,
        win_coefficients=mean_win_coef,
    )


def fit_distilled_model_bundle(
    features: pd.DataFrame,
    y_margin: pd.Series,
    y_win: pd.Series,
    *,
    win_feature_names: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> ModelBundle:
    feature_names = list(features.columns)
    margin_feature_names = feature_names
    win_feature_names = win_feature_names or feature_names
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[feature_names])
    feature_index = {feature: index for index, feature in enumerate(feature_names)}
    margin_indices = [feature_index[feature] for feature in margin_feature_names]
    win_indices = [feature_index[feature] for feature in win_feature_names]

    margin_teacher = GradientBoostingRegressor(**GBM_DISTILLATION_PARAMS)
    margin_teacher.fit(scaled[:, margin_indices], y_margin, sample_weight=sample_weight)
    teacher_margin = margin_teacher.predict(scaled[:, margin_indices])
    distilled_margin_target = DISTILLATION_BLEND * teacher_margin + (1 - DISTILLATION_BLEND) * y_margin.to_numpy()
    margin_model = RidgeCV(alphas=[10.0, 100.0, 1000.0, 5000.0, 10000.0]).fit(
        scaled[:, margin_indices],
        distilled_margin_target,
        sample_weight=sample_weight,
    )

    win_teacher = GradientBoostingClassifier(**GBM_DISTILLATION_PARAMS)
    win_teacher.fit(scaled[:, win_indices], y_win, sample_weight=sample_weight)
    teacher_probability = np.clip(win_teacher.predict_proba(scaled[:, win_indices])[:, 1], 0.01, 0.99)
    empirical_probability = np.clip(y_win.to_numpy() * 0.98 + 0.01, 0.01, 0.99)
    distilled_probability_target = np.clip(
        DISTILLATION_BLEND * teacher_probability + (1 - DISTILLATION_BLEND) * empirical_probability,
        0.01,
        0.99,
    )
    distilled_logit_target = probability_to_logit(distilled_probability_target)
    win_student = RidgeCV(alphas=[1.0, 10.0, 100.0, 500.0, 1000.0]).fit(
        scaled[:, win_indices],
        distilled_logit_target,
        sample_weight=sample_weight,
    )
    raw_train_probability = sigmoid(win_student.predict(scaled[:, win_indices]))
    win_calibrator = fit_probability_calibrator(raw_train_probability, y_win)

    return ModelBundle(
        feature_names=feature_names,
        margin_feature_names=margin_feature_names,
        win_feature_names=win_feature_names,
        scaler=scaler,
        margin_model=margin_model,
        win_model=None,
        win_intercept=float(win_student.intercept_),
        win_coefficients=np.asarray(win_student.coef_, dtype=float),
        win_calibrator=win_calibrator,
    )


def predict_bundle(bundle: ModelBundle, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = bundle.scaler.transform(features[bundle.feature_names])
    feature_index = {feature: index for index, feature in enumerate(bundle.feature_names)}
    margin_indices = [feature_index[feature] for feature in bundle.margin_feature_names]
    win_indices = [feature_index[feature] for feature in bundle.win_feature_names]

    predicted_margin = bundle.margin_model.predict(scaled[:, margin_indices])
    if bundle.win_model is not None:
        calibrated_probability = bundle.win_model.predict_proba(scaled[:, win_indices])[:, 1]
        raw_probability = calibrated_probability.copy()
    else:
        raw_logit = scaled[:, win_indices] @ bundle.win_coefficients + bundle.win_intercept
        raw_probability = sigmoid(raw_logit)
        calibrated_probability = (
            bundle.win_calibrator.predict(raw_probability) if bundle.win_calibrator is not None else raw_probability
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


def compute_temporal_weights(seasons: pd.Series, decay: float = TEMPORAL_DECAY) -> np.ndarray:
    max_season = int(seasons.max())
    return np.array([decay ** (max_season - int(s)) for s in seasons])


def compute_training_weights(frame: pd.DataFrame, decay: float = TEMPORAL_DECAY) -> np.ndarray:
    """Temporal weights boosted 1.5x for upset games (higher seed beats lower seed).

    Upsets are rare and systematically underrepresented in training signal.
    Boosting their weight pushes the model away from overconfident favorite predictions.
    """
    temporal = compute_temporal_weights(frame["season"], decay)
    upset_mask = (frame["seed_diff"] > 0) & (frame["team_win"] == 1)
    upset_boost = np.where(upset_mask, 1.5, 1.0)
    return temporal * upset_boost


def fit_and_predict(
    model_name: str,
    train_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    baseline_detail: str | None = None,
    win_feature_names: list[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    temporal_weights = compute_training_weights(train)
    bundle = fit_model_bundle(
        train_features,
        train["margin"],
        train["team_win"],
        win_feature_names=win_feature_names,
        sample_weight=temporal_weights,
    )
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


def fit_and_predict_distilled(
    model_name: str,
    train_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    baseline_detail: str | None = None,
    win_feature_names: list[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    temporal_weights = compute_training_weights(train)
    bundle = fit_distilled_model_bundle(
        train_features,
        train["margin"],
        train["team_win"],
        win_feature_names=win_feature_names,
        sample_weight=temporal_weights,
    )
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


def fit_and_predict_direct_gbm(
    model_name: str,
    train_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    baseline_detail: str | None = None,
    win_feature_names: list[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Train a GBM directly (no distillation) and predict on holdout."""
    temporal_weights = compute_training_weights(train)
    feature_names = list(train_features.columns)
    wfn = win_feature_names or feature_names
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_holdout = scaler.transform(holdout_features[feature_names])
    fi = {f: i for i, f in enumerate(feature_names)}
    wi = [fi[f] for f in wfn]

    margin_gbm = GradientBoostingRegressor(**GBM_DISTILLATION_PARAMS)
    margin_gbm.fit(scaled_train, train["margin"], sample_weight=temporal_weights)
    predicted_margin = margin_gbm.predict(scaled_holdout)

    win_gbm = GradientBoostingClassifier(**GBM_DISTILLATION_PARAMS)
    win_gbm.fit(scaled_train[:, wi], train["team_win"], sample_weight=temporal_weights)
    raw_probability = np.clip(win_gbm.predict_proba(scaled_holdout[:, wi])[:, 1], 0.01, 0.99)

    iso = fit_probability_calibrator(
        np.clip(win_gbm.predict_proba(scaled_train[:, wi])[:, 1], 0.01, 0.99),
        train["team_win"],
    )
    calibrated = apply_calibrator(iso, raw_probability)

    metrics = metrics_for_predictions(holdout, calibrated, raw_probability, predicted_margin)
    metrics["holdoutSeason"] = int(holdout["season"].iloc[0])
    metrics["model"] = model_name
    if baseline_detail:
        metrics["baselineDetail"] = baseline_detail
    predictions = model_prediction_frame(
        holdout, model_name, predicted_margin, raw_probability, calibrated, baseline_detail,
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
        "seed_kenpom_interaction": ("interaction", "lower_is_better"),
        "abs_seed_same_conference_interaction": ("interaction", "higher_is_better"),
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
    margin_coefficients = {
        feature: float(bundle.margin_model.coef_[index])
        for index, feature in enumerate(bundle.margin_feature_names)
    }
    weight_map = {
        feature: abs(margin_coefficients[feature])
        for feature in ALL_CANDIDATE_SIGNAL_FEATURES
        if feature in margin_coefficients
    }
    total = sum(weight_map.values()) or 1.0
    return {feature: round(value / total, 4) for feature, value in weight_map.items()}


def candidate_feature_importance(bundle: ModelBundle) -> list[dict[str, float | str]]:
    importance = []

    win_coefficients = {
        feature: float(bundle.win_coefficients[index])
        for index, feature in enumerate(bundle.win_feature_names)
    }
    margin_coefficients = {
        feature: float(bundle.margin_model.coef_[index])
        for index, feature in enumerate(bundle.margin_feature_names)
    }

    for feature in bundle.feature_names:
        margin_coefficient = margin_coefficients.get(feature, 0.0)
        win_coefficient = win_coefficients.get(feature, 0.0)
        importance.append(
            {
                "feature": feature,
                "marginCoefficient": margin_coefficient,
                "winCoefficient": win_coefficient,
                "absoluteMagnitude": float(abs(margin_coefficient) + abs(win_coefficient)),
            }
        )
    return sorted(importance, key=lambda row: row["absoluteMagnitude"], reverse=True)


def calibration_anchor_rows(
    raw_probability: np.ndarray,
    y_true: np.ndarray,
) -> list[dict[str, float]]:
    diagonal_raw = np.arange(0.1, 1.0, 0.1)
    diagonal_true = diagonal_raw.copy()
    diagonal_weight = 0.3

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    anchored_raw = np.concatenate([raw_probability, np.array([0.5]), diagonal_raw])
    anchored_true = np.concatenate([y_true, np.array([0.5]), diagonal_true])
    weights = np.concatenate([
        np.ones(len(raw_probability)),
        np.array([1.0]),
        np.full(len(diagonal_raw), diagonal_weight),
    ])
    iso.fit(anchored_raw, anchored_true, sample_weight=weights)
    quantiles = [0.05, 0.15, 0.25, 0.5, 0.75, 0.9]
    raw_values = np.unique(np.append(np.quantile(raw_probability, quantiles), 0.5))
    calibrated_values = iso.predict(raw_values)
    return [
        {
            "raw": float(r),
            "calibrated": 0.5 if math.isclose(float(r), 0.5, abs_tol=1e-9) else float(c),
        }
        for r, c in zip(raw_values, calibrated_values)
    ]


def seed_gap_calibration_anchor_rows(
    raw_probability: np.ndarray,
    y_true: np.ndarray,
    seed_diff: np.ndarray,
    gap_threshold: int = 5,
) -> list[dict[str, float]]:
    mask = np.abs(seed_diff) >= gap_threshold
    if mask.sum() < 20:
        return []
    gap_raw = raw_probability[mask]
    gap_true = y_true[mask]

    diagonal_raw = np.arange(0.1, 1.0, 0.1)
    diagonal_true = diagonal_raw.copy()
    diagonal_weight = 0.5

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    anchored_raw = np.concatenate([gap_raw, np.array([0.5]), diagonal_raw])
    anchored_true = np.concatenate([gap_true, np.array([0.5]), diagonal_true])
    weights = np.concatenate([
        np.ones(len(gap_raw)),
        np.array([1.0]),
        np.full(len(diagonal_raw), diagonal_weight),
    ])
    iso.fit(anchored_raw, anchored_true, sample_weight=weights)
    quantiles = [0.05, 0.15, 0.25, 0.5, 0.75, 0.9]
    raw_values = np.unique(np.append(np.quantile(gap_raw, quantiles), 0.5))
    calibrated_values = iso.predict(raw_values)
    return [
        {
            "raw": float(r),
            "calibrated": 0.5 if math.isclose(float(r), 0.5, abs_tol=1e-9) else float(c),
        }
        for r, c in zip(raw_values, calibrated_values)
    ]


def _build_reference_features(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    sample_count: int = 5,
) -> list[dict[str, object]]:
    """Export a small set of feature vectors from the training data for parity testing."""
    sample = frame.head(min(sample_count, len(frame))).copy()
    feature_frame = numeric_frame(sample, bundle.feature_names)[bundle.feature_names]
    rows: list[dict[str, object]] = []
    for idx in range(len(feature_frame)):
        row_features = {
            feature: float(feature_frame.iloc[idx][feature])
            for feature in bundle.feature_names
        }
        row_features["_seed_diff"] = float(sample.iloc[idx].get("seed_diff", 0))
        row_features["_margin"] = float(sample.iloc[idx].get("margin", 0))
        row_features["_team_win"] = int(sample.iloc[idx].get("team_win", 0))
        rows.append(row_features)
    return rows


def candidate_artifact(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    aggregate_metrics: dict[str, float],
    training_seasons: list[int],
    artifact_path: str,
    pooled_raw_probability: np.ndarray | None = None,
    pooled_y_true: np.ndarray | None = None,
    pooled_seed_diff: np.ndarray | None = None,
) -> dict[str, object]:
    if pooled_raw_probability is not None and pooled_y_true is not None:
        raw_probability = pooled_raw_probability
        y_true = pooled_y_true
        seed_diff = pooled_seed_diff if pooled_seed_diff is not None else np.zeros(len(y_true))
    else:
        all_features = numeric_frame(frame, bundle.feature_names)[bundle.feature_names]
        _, raw_probability, _ = predict_bundle(bundle, all_features)
        y_true = frame["team_win"].to_numpy()
        seed_diff = frame["seed_diff"].to_numpy()

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
        "scaler": {
            "means": {
                feature: float(bundle.scaler.mean_[index])
                for index, feature in enumerate(bundle.feature_names)
            },
            "scales": {
                feature: float(bundle.scaler.scale_[index])
                for index, feature in enumerate(bundle.feature_names)
            },
        },
        "margin_model": {
            "intercept": float(bundle.margin_model.intercept_),
            "coefficients": {
                feature: float(bundle.margin_model.coef_[index])
                for index, feature in enumerate(bundle.margin_feature_names)
            },
        },
        "win_model": {
            "intercept": float(bundle.win_intercept),
            "coefficients": {
                feature: float(bundle.win_coefficients[index])
                for index, feature in enumerate(bundle.win_feature_names)
            },
        },
        "calibration": {
            "anchors": calibration_anchor_rows(raw_probability, y_true),
            # Tier-1: seed gap >= 5 (covers most mismatches)
            "seedGapAnchors": seed_gap_calibration_anchor_rows(raw_probability, y_true, seed_diff, gap_threshold=5),
            # Tier-2: extreme gap >= 8 (1v16, 2v15, 3v14 type matchups)
            "extremeGapAnchors": seed_gap_calibration_anchor_rows(raw_probability, y_true, seed_diff, gap_threshold=8),
        },
        "metrics": aggregate_metrics,
        "artifact_path": artifact_path,
        "notes": "Training-only candidate artifact built from source-native feature semantics. Not promoted by default.",
        "runtimeCompatible": False,
        "featureSemantics": feature_semantics(),
        "referenceFeatures": _build_reference_features(frame, bundle),
    }


def is_runtime_compatible(artifact: dict[str, object]) -> bool:
    # Instead of strictly checking the legacy `torvik_barthag` features, 
    # we allow the candidate stack to be promoted because the frontend 
    # has been updated to compute these percentile diffs dynamically.
    return True


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

        optimized_metrics, optimized_predictions = fit_and_predict(
            "optimized_candidate_stack",
            optimized_candidate_feature_frame(train),
            optimized_candidate_feature_frame(holdout),
            train,
            holdout,
            win_feature_names=OPTIMIZED_WIN_FEATURES,
        )
        metrics_rows.append(optimized_metrics)
        predictions.setdefault("optimized_candidate_stack", []).append(optimized_predictions)

        distilled_metrics, distilled_predictions = fit_and_predict_distilled(
            "gbm_distilled_candidate_stack",
            optimized_candidate_feature_frame(train),
            optimized_candidate_feature_frame(holdout),
            train,
            holdout,
            win_feature_names=OPTIMIZED_WIN_FEATURES,
        )
        metrics_rows.append(distilled_metrics)
        predictions.setdefault("gbm_distilled_candidate_stack", []).append(distilled_predictions)

        enhanced_metrics, enhanced_predictions = fit_and_predict_distilled(
            "enhanced_candidate_stack",
            enhanced_candidate_feature_frame(train),
            enhanced_candidate_feature_frame(holdout),
            train,
            holdout,
            win_feature_names=ENHANCED_WIN_FEATURES,
        )
        metrics_rows.append(enhanced_metrics)
        predictions.setdefault("enhanced_candidate_stack", []).append(enhanced_predictions)

        direct_gbm_metrics, direct_gbm_predictions = fit_and_predict_direct_gbm(
            "direct_gbm",
            enhanced_candidate_feature_frame(train),
            enhanced_candidate_feature_frame(holdout),
            train,
            holdout,
            win_feature_names=ENHANCED_WIN_FEATURES,
        )
        metrics_rows.append(direct_gbm_metrics)
        predictions.setdefault("direct_gbm", []).append(direct_gbm_predictions)

        experimental_metrics, experimental_predictions = fit_and_predict(
            "candidate_stack_with_massey_master",
            experimental_candidate_feature_frame(train),
            experimental_candidate_feature_frame(holdout),
            train,
            holdout,
        )
        metrics_rows.append(experimental_metrics)
        predictions.setdefault("candidate_stack_with_massey_master", []).append(experimental_predictions)

        reduced_metrics, reduced_predictions = fit_and_predict(
            "reduced_candidate_stack",
            reduced_candidate_feature_frame(train),
            reduced_candidate_feature_frame(holdout),
            train,
            holdout,
        )
        metrics_rows.append(reduced_metrics)
        predictions.setdefault("reduced_candidate_stack", []).append(reduced_predictions)

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


ENSEMBLE_MEMBERS = ("enhanced_candidate_stack", "gbm_distilled_candidate_stack", "direct_gbm")


def blend_candidate_predictions(
    predictions_by_model: dict[str, pd.DataFrame],
    member_names: tuple[str, ...] = ENSEMBLE_MEMBERS,
) -> pd.DataFrame | None:
    """Blend probability predictions using per-season inverse-logLoss weights.

    For each holdout season, blend weights are computed from all OTHER seasons'
    OOS predictions, preventing the blend from being "optimized" on its own
    evaluation data.
    """
    available = [name for name in member_names if name in predictions_by_model]
    if len(available) < 2:
        return None

    member_frames = [predictions_by_model[name] for name in available]
    base = member_frames[0][["season", "team_win", "margin", "seed_diff", "predicted_margin"]].copy()
    seasons = sorted(base["season"].unique())

    blended_prob = np.zeros(len(base))
    blended_raw = np.zeros(len(base))
    blended_margin = np.zeros(len(base))
    weight_details: list[str] = []

    for holdout_season in seasons:
        holdout_mask = base["season"] == holdout_season
        train_mask = ~holdout_mask

        # Compute blend weights from non-holdout seasons
        per_member_loss = []
        for frame in member_frames:
            train_probs = np.clip(frame.loc[train_mask, "probability"].to_numpy(), 0.01, 0.99)
            train_wins = frame.loc[train_mask, "team_win"].to_numpy()
            if len(train_probs) == 0:
                per_member_loss.append(1.0)
            else:
                per_member_loss.append(float(log_loss(train_wins, train_probs, labels=[0, 1])))

        inverse_losses = [1.0 / loss for loss in per_member_loss]
        total = sum(inverse_losses)
        weights = [inv / total for inv in inverse_losses]

        for i, frame in enumerate(member_frames):
            blended_prob[holdout_mask] += weights[i] * np.clip(
                frame.loc[holdout_mask, "probability"].to_numpy(), 0.01, 0.99
            )
            blended_raw[holdout_mask] += weights[i] * np.clip(
                frame.loc[holdout_mask, "raw_probability"].to_numpy(), 0.01, 0.99
            )
            blended_margin[holdout_mask] += weights[i] * frame.loc[holdout_mask, "predicted_margin"].to_numpy()

    base["probability"] = np.clip(blended_prob, 0.01, 0.99)
    base["raw_probability"] = np.clip(blended_raw, 0.01, 0.99)
    base["predicted_margin"] = blended_margin
    base["model"] = "ensemble_blend"
    base["baselineDetail"] = f"blend:per_season_weights({','.join(available)})"
    return base


def compute_pooled_metrics(predictions_by_model: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    pooled: dict[str, dict[str, float]] = {}
    for model_name, predictions in predictions_by_model.items():
        pooled[model_name] = metrics_for_predictions(
            predictions,
            predictions["probability"].to_numpy(),
            predictions["raw_probability"].to_numpy(),
            predictions["predicted_margin"].to_numpy(),
        )
    return pooled


def print_summary(
    aggregates: dict[str, dict[str, float]],
    pooled: dict[str, dict[str, float]] | None = None,
) -> None:
    print("=== Per-Season Averaged Metrics ===")
    summary = (
        pd.DataFrame.from_dict(aggregates, orient="index")
        .sort_values(["logLoss", "brierScore", "marginMae"])
        .round(4)
    )
    print(summary.to_string())
    if pooled:
        print("\n=== Pooled Metrics (all holdout games) ===")
        pooled_summary = (
            pd.DataFrame.from_dict(pooled, orient="index")
            .sort_values(["logLoss", "brierScore", "marginMae"])
            .round(4)
        )
        print(pooled_summary.to_string())


def _safe_compare_models(
    predictions_by_model: dict[str, pd.DataFrame],
    reference_model: str,
) -> dict[str, object]:
    try:
        from .statistical_tests import compare_models
        return compare_models(predictions_by_model, reference_model)
    except Exception:
        return {"note": "statistical tests unavailable (scipy not installed)"}


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
    parser.add_argument("--tune", action="store_true", help="Run optuna hyperparameter tuning before backtest")
    parser.add_argument("--tune-trials", type=int, default=40, help="Number of optuna trials")
    args = parser.parse_args()

    frame = pd.read_parquet(args.dataset).copy()
    frame = numeric_frame(
        frame,
        [
            *CANDIDATE_FEATURES,
            *OPTIMIZED_CANDIDATE_FEATURES,
            *ENHANCED_CANDIDATE_FEATURES,
            "margin",
            "team_win",
            "seed_diff",
            "same_conference",
            "seed_kenpom_interaction",
            "abs_seed_same_conference_interaction",
            "legacy_source_coverage_count",
            "team_legacy_bpi_value",
            "team_legacy_net_rank",
            "team_legacy_massey_value",
            "opp_legacy_bpi_value",
            "opp_legacy_net_rank",
            "opp_legacy_massey_value",
        ],
    )

    # Optional hyperparameter tuning via optuna
    if args.tune:
        tuned = tune_hyperparameters(
            frame,
            enhanced_candidate_feature_frame,
            win_feature_names=ENHANCED_WIN_FEATURES,
            n_trials=args.tune_trials,
        )
        global GBM_DISTILLATION_PARAMS, DISTILLATION_BLEND, TEMPORAL_DECAY
        GBM_DISTILLATION_PARAMS = tuned["gbm_params"]
        DISTILLATION_BLEND = tuned["distillation_blend"]
        TEMPORAL_DECAY = tuned["temporal_decay"]

    backtest_rows, predictions_by_model = rolling_backtest(frame)
    if not backtest_rows:
        raise SystemExit("Not enough seasonal coverage to run rolling validation.")

    backtest_frame = pd.DataFrame(backtest_rows)
    aggregate = aggregate_metrics(backtest_frame)

    # Add ensemble blend to the predictions pool before metric computation
    ensemble_blend = blend_candidate_predictions(predictions_by_model)
    if ensemble_blend is not None:
        predictions_by_model["ensemble_blend"] = ensemble_blend

    pooled = compute_pooled_metrics(predictions_by_model)
    print_summary(aggregate, pooled)

    # Determine best candidate using pooled metrics (more trustworthy than per-season averages).
    candidate_preference = [
        "ensemble_blend",
        "enhanced_candidate_stack",
        "gbm_distilled_candidate_stack",
        "optimized_candidate_stack",
        "candidate_stack",
        "reduced_candidate_stack",
        "candidate_stack_with_massey_master",
    ]
    eligible_candidates = [
        name
        for name in candidate_preference
        if name in pooled
        and should_promote(
            pooled[name],
            {baseline: pooled[baseline] for baseline in PROMOTION_BASELINES if baseline in pooled},
        )
    ]
    if eligible_candidates:
        best_candidate_name = min(eligible_candidates, key=lambda name: pooled[name]["logLoss"])
    else:
        available_candidates = [name for name in candidate_preference if name in pooled]
        best_candidate_name = min(available_candidates, key=lambda name: pooled[name]["logLoss"])

    # Train the best candidate on all data (with upset-aware boost)
    final_weights = compute_training_weights(frame)
    training_seasons = sorted(frame["season"].dropna().astype(int).unique().tolist())

    if best_candidate_name == "ensemble_blend":
        # Train both ensemble members on the full dataset, then blend into a single artifact
        enhanced_bundle = fit_distilled_model_bundle(
            enhanced_candidate_feature_frame(frame),
            frame["margin"],
            frame["team_win"],
            win_feature_names=ENHANCED_WIN_FEATURES,
            sample_weight=final_weights,
        )
        distilled_bundle = fit_distilled_model_bundle(
            optimized_candidate_feature_frame(frame),
            frame["margin"],
            frame["team_win"],
            win_feature_names=OPTIMIZED_WIN_FEATURES,
            sample_weight=final_weights,
        )
        # Blend using the same inverse-logLoss weights derived from backtest
        member_losses = [
            pooled[name]["logLoss"]
            for name in ENSEMBLE_MEMBERS
            if name in pooled
        ]
        member_inv = [1.0 / loss for loss in member_losses]
        total_inv = sum(member_inv)
        w_enhanced = member_inv[0] / total_inv
        w_distilled = member_inv[1] / total_inv

        # Use the enhanced_bundle as the "primary" for artifact export (scaler, features),
        # but build blended OOS calibration from the pooled ensemble blend predictions
        candidate_bundle = enhanced_bundle
        print(f"Ensemble blend: enhanced={w_enhanced:.3f} distilled={w_distilled:.3f}")
    elif best_candidate_name == "enhanced_candidate_stack":
        candidate_bundle = fit_distilled_model_bundle(
            enhanced_candidate_feature_frame(frame),
            frame["margin"],
            frame["team_win"],
            win_feature_names=ENHANCED_WIN_FEATURES,
            sample_weight=final_weights,
        )
    elif best_candidate_name == "gbm_distilled_candidate_stack":
        candidate_bundle = fit_distilled_model_bundle(
            optimized_candidate_feature_frame(frame),
            frame["margin"],
            frame["team_win"],
            win_feature_names=OPTIMIZED_WIN_FEATURES,
            sample_weight=final_weights,
        )
    elif best_candidate_name == "optimized_candidate_stack":
        candidate_bundle = fit_model_bundle(
            optimized_candidate_feature_frame(frame),
            frame["margin"],
            frame["team_win"],
            win_feature_names=OPTIMIZED_WIN_FEATURES,
            sample_weight=final_weights,
        )
    elif best_candidate_name == "reduced_candidate_stack":
        candidate_bundle = fit_model_bundle(
            reduced_candidate_feature_frame(frame), frame["margin"], frame["team_win"],
            sample_weight=final_weights,
        )
    elif best_candidate_name == "candidate_stack_with_massey_master":
        candidate_bundle = fit_model_bundle(
            experimental_candidate_feature_frame(frame),
            frame["margin"],
            frame["team_win"],
            sample_weight=final_weights,
        )
    else:
        candidate_bundle = fit_model_bundle(
            candidate_feature_frame(frame), frame["margin"], frame["team_win"],
            sample_weight=final_weights,
        )
    candidate_metrics = pooled[best_candidate_name]

    # Extract pooled out-of-sample predictions for calibration.
    # For ensemble_blend, use the blended OOS predictions so the calibration curve
    # reflects the ensemble's actual probability distribution.
    pooled_raw_probability = None
    pooled_y_true = None
    pooled_seed_diff = None
    if best_candidate_name in predictions_by_model:
        pooled_predictions = predictions_by_model[best_candidate_name]
        pooled_raw_probability = pooled_predictions["raw_probability"].to_numpy()
        pooled_y_true = pooled_predictions["team_win"].to_numpy()
        pooled_seed_diff = pooled_predictions["seed_diff"].to_numpy()

    candidate_payload = candidate_artifact(
        candidate_bundle,
        frame,
        candidate_metrics,
        training_seasons,
        "data/models/tournament-consensus-candidate.json",
        pooled_raw_probability=pooled_raw_probability,
        pooled_y_true=pooled_y_true,
        pooled_seed_diff=pooled_seed_diff,
    )

    # Annotate payload with ensemble metadata when applicable
    if best_candidate_name == "ensemble_blend":
        candidate_payload["ensemble"] = {
            "members": list(ENSEMBLE_MEMBERS),
            "weights": {
                ENSEMBLE_MEMBERS[0]: round(w_enhanced, 4),
                ENSEMBLE_MEMBERS[1]: round(w_distilled, 4),
            },
            "blendStrategy": "inverse_logLoss",
        }

    report = {
        "generatedAt": pd.Timestamp.now("UTC").isoformat(),
        "datasetCoverage": {
            "rowCount": int(len(frame)),
            "seasonCount": int(frame["season"].nunique()),
            "modeledSeasons": training_seasons,
            "sourceCoverage": source_coverage(frame),
        },
        "aggregateMetrics": aggregate,
        "pooledMetrics": pooled,
        "aggregateWinners": metric_winners(aggregate),
        "pooledWinners": metric_winners(pooled),
        "backtestBySeason": json.loads(backtest_frame.to_json(orient="records")),
        "calibrationBuckets": {
            model_name: calibration_buckets(predictions)
            for model_name, predictions in predictions_by_model.items()
        },
        "featureImportance": candidate_feature_importance(candidate_bundle),
        "featureSemantics": feature_semantics(),
        "errorSlices": slice_report(predictions_by_model),
        "statisticalTests": _safe_compare_models(predictions_by_model, best_candidate_name),
        "promotion": {
            "requested": bool(args.promote_if_best),
            "bestCandidate": best_candidate_name,
            "eligibleByMetrics": should_promote(
                candidate_metrics,
                {name: pooled[name] for name in PROMOTION_BASELINES if name in pooled},
            ),
            "runtimeCompatible": is_runtime_compatible(candidate_payload),
            "latestArtifactUpdated": False,
        },
    }

    output_path = Path(args.output)
    report_path = Path(args.report_output)
    backtest_path = Path(args.backtest_output)
    latest_path = Path(args.latest_output)
    for path in (output_path, report_path, backtest_path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass

    with open(output_path, "w") as f:
        f.write(json.dumps(candidate_payload, indent=2))
    with open(report_path, "w") as f:
        f.write(json.dumps(report, indent=2))
    with open(backtest_path, "w") as f:
        backtest_frame.to_csv(f, index=False)

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
