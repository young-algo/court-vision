"""Statistical significance tests for model comparison.

Provides paired bootstrap confidence intervals and McNemar's test to
determine whether observed metric differences between models are
statistically significant vs. noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def paired_bootstrap_ci(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for log-loss difference (A - B).

    Returns dict with keys: mean_diff, ci_lower, ci_upper, p_value.
    If the CI excludes 0, the difference is significant at the given level.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    clipped_a = np.clip(probs_a, 0.01, 0.99)
    clipped_b = np.clip(probs_b, 0.01, 0.99)

    per_game_loss_a = -(y_true * np.log(clipped_a) + (1 - y_true) * np.log(1 - clipped_a))
    per_game_loss_b = -(y_true * np.log(clipped_b) + (1 - y_true) * np.log(1 - clipped_b))
    per_game_diff = per_game_loss_a - per_game_loss_b

    observed_diff = float(per_game_diff.mean())

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        boot_diffs[i] = per_game_diff[indices].mean()

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    p_value = float(np.mean(boot_diffs >= 0)) if observed_diff < 0 else float(np.mean(boot_diffs <= 0))

    return {
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": min(2 * p_value, 1.0),
        "significant": ci_lower > 0 or ci_upper < 0,
    }


def mcnemar_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """McNemar's test for paired classification agreement.

    Tests whether models A and B disagree on classification more than
    would be expected by chance.
    """
    pred_a = (probs_a >= threshold).astype(int)
    pred_b = (probs_b >= threshold).astype(int)
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())

    n_discordant = b + c
    if n_discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "significant": False}

    chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
    from scipy import stats as scipy_stats
    p_value = float(1 - scipy_stats.chi2.cdf(chi2, df=1))

    return {
        "chi2": chi2,
        "p_value": p_value,
        "b": b,
        "c": c,
        "significant": p_value < 0.05,
    }


def compare_models(
    predictions_by_model: dict[str, pd.DataFrame],
    reference_model: str = "enhanced_candidate_stack",
) -> dict[str, dict[str, object]]:
    """Compare all models against a reference using bootstrap CI and McNemar."""
    if reference_model not in predictions_by_model:
        return {}

    ref = predictions_by_model[reference_model]
    y_true = ref["team_win"].to_numpy()
    ref_probs = np.clip(ref["probability"].to_numpy(), 0.01, 0.99)

    results: dict[str, dict[str, object]] = {}
    for model_name, preds in predictions_by_model.items():
        if model_name == reference_model:
            continue
        model_probs = np.clip(preds["probability"].to_numpy(), 0.01, 0.99)
        if len(model_probs) != len(ref_probs):
            continue

        bootstrap = paired_bootstrap_ci(y_true, ref_probs, model_probs)
        try:
            mcnemar = mcnemar_test(y_true, ref_probs, model_probs)
        except ImportError:
            mcnemar = {"note": "scipy not available"}

        results[model_name] = {
            "bootstrap": bootstrap,
            "mcnemar": mcnemar,
        }

    return results
