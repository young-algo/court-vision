from __future__ import annotations

import unittest

import pandas as pd

from historical_pipeline.massey_master_data import parse_record_win_pct, select_pre_tournament_snapshot_rows


class MasseyMasterDataTests(unittest.TestCase):
    def test_parse_record_win_pct(self) -> None:
        self.assertAlmostEqual(parse_record_win_pct("21-5"), 21 / 26)
        self.assertEqual(parse_record_win_pct("0-0"), 0.0)
        self.assertTrue(pd.isna(parse_record_win_pct("n/a")))

    def test_selects_latest_dated_snapshot_before_selection_sunday(self) -> None:
        frame = pd.DataFrame(
            [
                {"season": 2023, "week": 0, "parsed_date": pd.NaT, "team": "Alpha"},
                {"season": 2023, "week": 16, "parsed_date": pd.Timestamp("2023-02-26"), "team": "Alpha"},
                {"season": 2023, "week": 17, "parsed_date": pd.Timestamp("2023-03-05"), "team": "Alpha"},
                {"season": 2023, "week": 18, "parsed_date": pd.Timestamp("2023-03-12"), "team": "Alpha"},
                {"season": 2023, "week": 19, "parsed_date": pd.Timestamp("2023-04-01"), "team": "Alpha"},
            ]
        )

        selected, method = select_pre_tournament_snapshot_rows(frame, pd.Timestamp("2023-03-12"))
        self.assertEqual(method, "latest_dated_pre_selection_sunday")
        self.assertEqual(selected["week"].dropna().unique().tolist(), [18])

    def test_falls_back_to_inferred_week_when_dates_missing(self) -> None:
        frame = pd.DataFrame(
            [
                {"season": 2018, "week": 0, "parsed_date": pd.NaT, "team": "Alpha"},
                {"season": 2018, "week": 17, "parsed_date": pd.NaT, "team": "Alpha"},
                {"season": 2018, "week": 18, "parsed_date": pd.NaT, "team": "Alpha"},
                {"season": 2018, "week": 19, "parsed_date": pd.NaT, "team": "Alpha"},
            ]
        )

        selected, method = select_pre_tournament_snapshot_rows(frame, pd.Timestamp("2018-03-11"))
        self.assertEqual(method, "week_order_fallback")
        self.assertEqual(selected["week"].dropna().unique().tolist(), [18])


if __name__ == "__main__":
    unittest.main()
