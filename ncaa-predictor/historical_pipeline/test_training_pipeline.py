from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from historical_pipeline.build_training_dataset import apply_semantic_features
from historical_pipeline.train_model import should_promote


def synthetic_training_frame() -> pd.DataFrame:
    rows = []
    for season, offset in ((2021, 0.0), (2022, 0.3), (2023, 0.6)):
        rows.extend(
            [
                {
                    "season": season,
                    "bpi_rank_percentile_diff": 0.45 + offset,
                    "net_rank_percentile_diff": 0.40 + offset,
                    "massey_ordinal_rank_percentile_diff": 0.35 + offset,
                    "massey_composite_rank_percentile_diff": 0.42 + offset,
                    "massey_mean_rank_percentile_diff": 0.38 + offset,
                    "massey_median_rank_percentile_diff": 0.36 + offset,
                    "massey_rank_stdev_advantage": 1.6 + offset,
                    "kenpom_badj_em_power_diff": 8.0 + offset,
                    "evanmiya_relative_rating_power_diff": 7.5 + offset,
                    "fivethirtyeight_power_power_diff": 5.5 + offset,
                    "massey_win_pct_power_diff": 0.18 + offset / 10,
                    "resume_rank_blend_percentile_diff": 0.30 + offset,
                    "seed_diff": -6,
                    "same_conference": 0,
                    "missing_bpi_rank": 0,
                    "missing_net_rank": 0,
                    "missing_massey_ordinal_rank": 0,
                    "missing_massey_composite_rank": 0,
                    "missing_massey_mean_rank": 0,
                    "missing_massey_median_rank": 0,
                    "missing_massey_rank_stdev": 0,
                    "missing_kenpom_badj_em": 0,
                    "missing_evanmiya_relative_rating": 0,
                    "missing_fivethirtyeight_power": 0,
                    "missing_massey_win_pct": 0,
                    "missing_resume_rank_blend": 0,
                    "available_predictive_rank_count": 3,
                    "available_power_feature_count": 3,
                    "available_resume_feature_count": 4,
                    "margin": 9,
                    "team_win": 1,
                    "legacy_source_coverage_count": 3,
                    "team_legacy_bpi_value": 220,
                    "team_legacy_net_rank": 12,
                    "team_legacy_massey_value": 24,
                    "opp_legacy_bpi_value": 180,
                    "opp_legacy_net_rank": 32,
                    "opp_legacy_massey_value": 12,
                },
                {
                    "season": season,
                    "bpi_rank_percentile_diff": -0.45 - offset,
                    "net_rank_percentile_diff": -0.40 - offset,
                    "massey_ordinal_rank_percentile_diff": -0.35 - offset,
                    "massey_composite_rank_percentile_diff": -0.42 - offset,
                    "massey_mean_rank_percentile_diff": -0.38 - offset,
                    "massey_median_rank_percentile_diff": -0.36 - offset,
                    "massey_rank_stdev_advantage": -1.6 - offset,
                    "kenpom_badj_em_power_diff": -8.0 - offset,
                    "evanmiya_relative_rating_power_diff": -7.5 - offset,
                    "fivethirtyeight_power_power_diff": -5.5 - offset,
                    "massey_win_pct_power_diff": -0.18 - offset / 10,
                    "resume_rank_blend_percentile_diff": -0.30 - offset,
                    "seed_diff": 6,
                    "same_conference": 0,
                    "missing_bpi_rank": 0,
                    "missing_net_rank": 0,
                    "missing_massey_ordinal_rank": 0,
                    "missing_massey_composite_rank": 0,
                    "missing_massey_mean_rank": 0,
                    "missing_massey_median_rank": 0,
                    "missing_massey_rank_stdev": 0,
                    "missing_kenpom_badj_em": 0,
                    "missing_evanmiya_relative_rating": 0,
                    "missing_fivethirtyeight_power": 0,
                    "missing_massey_win_pct": 0,
                    "missing_resume_rank_blend": 0,
                    "available_predictive_rank_count": 3,
                    "available_power_feature_count": 3,
                    "available_resume_feature_count": 4,
                    "margin": -9,
                    "team_win": 0,
                    "legacy_source_coverage_count": 3,
                    "team_legacy_bpi_value": 180,
                    "team_legacy_net_rank": 32,
                    "team_legacy_massey_value": 12,
                    "opp_legacy_bpi_value": 220,
                    "opp_legacy_net_rank": 12,
                    "opp_legacy_massey_value": 24,
                },
                {
                    "season": season,
                    "bpi_rank_percentile_diff": 0.10 + offset,
                    "net_rank_percentile_diff": 0.06 + offset,
                    "massey_ordinal_rank_percentile_diff": 0.08 + offset,
                    "massey_composite_rank_percentile_diff": 0.09 + offset,
                    "massey_mean_rank_percentile_diff": 0.07 + offset,
                    "massey_median_rank_percentile_diff": 0.06 + offset,
                    "massey_rank_stdev_advantage": 0.4 + offset,
                    "kenpom_badj_em_power_diff": 2.0 + offset,
                    "evanmiya_relative_rating_power_diff": 1.5 + offset,
                    "fivethirtyeight_power_power_diff": 1.0 + offset,
                    "massey_win_pct_power_diff": 0.05 + offset / 10,
                    "resume_rank_blend_percentile_diff": 0.04 + offset,
                    "seed_diff": 2,
                    "same_conference": 1,
                    "missing_bpi_rank": 0,
                    "missing_net_rank": 0,
                    "missing_massey_ordinal_rank": 0,
                    "missing_massey_composite_rank": 0,
                    "missing_massey_mean_rank": 0,
                    "missing_massey_median_rank": 0,
                    "missing_massey_rank_stdev": 0,
                    "missing_kenpom_badj_em": 0,
                    "missing_evanmiya_relative_rating": 0,
                    "missing_fivethirtyeight_power": 0,
                    "missing_massey_win_pct": 0,
                    "missing_resume_rank_blend": 0,
                    "available_predictive_rank_count": 3,
                    "available_power_feature_count": 3,
                    "available_resume_feature_count": 4,
                    "margin": 2,
                    "team_win": 1,
                    "legacy_source_coverage_count": 3,
                    "team_legacy_bpi_value": 205,
                    "team_legacy_net_rank": 25,
                    "team_legacy_massey_value": 18,
                    "opp_legacy_bpi_value": 198,
                    "opp_legacy_net_rank": 20,
                    "opp_legacy_massey_value": 17,
                },
                {
                    "season": season,
                    "bpi_rank_percentile_diff": -0.10 - offset,
                    "net_rank_percentile_diff": -0.06 - offset,
                    "massey_ordinal_rank_percentile_diff": -0.08 - offset,
                    "massey_composite_rank_percentile_diff": -0.09 - offset,
                    "massey_mean_rank_percentile_diff": -0.07 - offset,
                    "massey_median_rank_percentile_diff": -0.06 - offset,
                    "massey_rank_stdev_advantage": -0.4 - offset,
                    "kenpom_badj_em_power_diff": -2.0 - offset,
                    "evanmiya_relative_rating_power_diff": -1.5 - offset,
                    "fivethirtyeight_power_power_diff": -1.0 - offset,
                    "massey_win_pct_power_diff": -0.05 - offset / 10,
                    "resume_rank_blend_percentile_diff": -0.04 - offset,
                    "seed_diff": -2,
                    "same_conference": 1,
                    "missing_bpi_rank": 0,
                    "missing_net_rank": 0,
                    "missing_massey_ordinal_rank": 0,
                    "missing_massey_composite_rank": 0,
                    "missing_massey_mean_rank": 0,
                    "missing_massey_median_rank": 0,
                    "missing_massey_rank_stdev": 0,
                    "missing_kenpom_badj_em": 0,
                    "missing_evanmiya_relative_rating": 0,
                    "missing_fivethirtyeight_power": 0,
                    "missing_massey_win_pct": 0,
                    "missing_resume_rank_blend": 0,
                    "available_predictive_rank_count": 3,
                    "available_power_feature_count": 3,
                    "available_resume_feature_count": 4,
                    "margin": -2,
                    "team_win": 0,
                    "legacy_source_coverage_count": 3,
                    "team_legacy_bpi_value": 198,
                    "team_legacy_net_rank": 20,
                    "team_legacy_massey_value": 17,
                    "opp_legacy_bpi_value": 205,
                    "opp_legacy_net_rank": 25,
                    "opp_legacy_massey_value": 18,
                },
            ]
        )
    return pd.DataFrame(rows)


class TrainingPipelineTests(unittest.TestCase):
    def test_apply_semantic_features_preserves_rank_and_power_direction(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "season": 2025,
                    "team_name": "Alpha",
                    "opponent_name": "Beta",
                    "canonical_team_id": "alpha",
                    "opponent_team_id": "beta",
                    "seed": 4,
                    "opponent_seed": 9,
                    "team_score": 75,
                    "opponent_score": 68,
                    "team_bpi_rank": 10,
                    "opp_bpi_rank": 25,
                    "team_net_rank": 8,
                    "opp_net_rank": 20,
                    "team_massey_ordinal_rank": 12,
                    "opp_massey_ordinal_rank": 30,
                    "team_massey_composite_rank": 11,
                    "opp_massey_composite_rank": 28,
                    "team_massey_mean_rank": 14.5,
                    "opp_massey_mean_rank": 31.2,
                    "team_massey_median_rank": 13.0,
                    "opp_massey_median_rank": 29.0,
                    "team_massey_rank_stdev": 2.8,
                    "opp_massey_rank_stdev": 6.7,
                    "team_massey_win_pct": 0.84,
                    "opp_massey_win_pct": 0.64,
                    "team_kenpom_badj_em": 24.0,
                    "opp_kenpom_badj_em": 10.0,
                    "team_evanmiya_relative_rating": 21.0,
                    "opp_evanmiya_relative_rating": 9.0,
                    "team_fivethirtyeight_power": 18.0,
                    "opp_fivethirtyeight_power": 11.0,
                    "team_resume_avg_rank": 15,
                    "opp_resume_avg_rank": 30,
                    "team_quality_avg_rank": 18,
                    "opp_quality_avg_rank": 35,
                    "team_kpi_rank": 16,
                    "opp_kpi_rank": 28,
                    "team_sor_rank": 14,
                    "opp_sor_rank": 25,
                    "team_wab_rank": 12,
                    "opp_wab_rank": 27,
                    "team_trk_rank": 11,
                    "opp_trk_rank": 24,
                    "team_kp_rank": 9,
                    "opp_kp_rank": 21,
                    "team_conference": "ACC",
                    "opp_conference": "SEC",
                    "team_legacy_bpi_value": 220,
                    "opp_legacy_bpi_value": 180,
                    "team_legacy_net_rank": 8,
                    "opp_legacy_net_rank": 20,
                    "team_legacy_massey_value": 24,
                    "opp_legacy_massey_value": 10,
                }
            ]
        )

        enriched = apply_semantic_features(frame)
        forward = enriched.iloc[0]
        reverse = enriched.iloc[1]

        self.assertGreater(forward["bpi_rank_percentile_diff"], 0)
        self.assertGreater(forward["massey_composite_rank_percentile_diff"], 0)
        self.assertGreater(forward["massey_rank_stdev_advantage"], 0)
        self.assertGreater(forward["massey_win_pct_power_diff"], 0)
        self.assertGreater(forward["kenpom_badj_em_power_diff"], 0)
        self.assertLess(forward["seed_kenpom_interaction"], 0)
        self.assertEqual(forward["abs_seed_same_conference_interaction"], 0)
        self.assertEqual(forward["missing_bpi_rank"], 0)
        self.assertEqual(forward["missing_massey_composite_rank"], 0)
        self.assertEqual(forward["missing_resume_rank_blend"], 0)
        self.assertLess(reverse["seed_kenpom_interaction"], 0)
        self.assertEqual(reverse["abs_seed_same_conference_interaction"], 0)
        self.assertLess(reverse["bpi_rank_percentile_diff"], 0)
        self.assertLess(reverse["massey_composite_rank_percentile_diff"], 0)
        self.assertLess(reverse["massey_rank_stdev_advantage"], 0)
        self.assertLess(reverse["massey_win_pct_power_diff"], 0)
        self.assertLess(reverse["kenpom_badj_em_power_diff"], 0)

    def test_should_promote_requires_candidate_to_beat_guardrail_baselines(self) -> None:
        self.assertTrue(
            should_promote(
                {"logLoss": 0.49, "brierScore": 0.16},
                {
                    "equal_weight_consensus": {"logLoss": 0.50, "brierScore": 0.17},
                    "seed_only_logit": {"logLoss": 0.55, "brierScore": 0.19},
                },
            )
        )
        self.assertFalse(
            should_promote(
                {"logLoss": 0.51, "brierScore": 0.16},
                {
                    "equal_weight_consensus": {"logLoss": 0.50, "brierScore": 0.17},
                    "seed_only_logit": {"logLoss": 0.55, "brierScore": 0.19},
                },
            )
        )

    def test_training_script_writes_candidate_report_and_backtest_outputs(self) -> None:
        dataset = synthetic_training_frame()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_path = temp_root / "synthetic.parquet"
            candidate_path = temp_root / "candidate.json"
            report_path = temp_root / "report.json"
            backtest_path = temp_root / "backtest.csv"
            latest_path = temp_root / "latest.json"
            dataset.to_parquet(dataset_path, index=False)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "historical_pipeline.train_model",
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(candidate_path),
                    "--report-output",
                    str(report_path),
                    "--backtest-output",
                    str(backtest_path),
                    "--latest-output",
                    str(latest_path),
                ],
                cwd="/Users/kevinturner/Documents/Code/Personal/ncaa-predictor-complete/ncaa-predictor",
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("wrote candidate artifact", completed.stdout)
            self.assertTrue(candidate_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(backtest_path.exists())
            self.assertFalse(latest_path.exists())

            report = json.loads(report_path.read_text())
            self.assertIn("candidate_stack", report["aggregateMetrics"])
            self.assertIn("optimized_candidate_stack", report["aggregateMetrics"])
            self.assertIn("gbm_distilled_candidate_stack", report["aggregateMetrics"])
            self.assertIn("seed_only_logit", report["aggregateMetrics"])
            self.assertIn("equal_weight_consensus", report["aggregateMetrics"])
            self.assertIn("promotion", report)


if __name__ == "__main__":
    unittest.main()
