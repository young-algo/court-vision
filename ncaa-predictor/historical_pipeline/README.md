# Historical Tournament Pipeline

This package builds a learned NCAA tournament consensus model from free annual pre-tournament snapshots.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r historical_pipeline/requirements.txt
```

## Pipeline

1. `python3 -m historical_pipeline.collect_tournament_games`
2. `python3 -m historical_pipeline.collect_snapshots`
3. `python3 -m historical_pipeline.build_training_dataset`
4. `python3 -m historical_pipeline.train_model`

To promote a newly trained artifact only when it clears the guardrail baselines:

```bash
python3 -m historical_pipeline.train_model --promote-if-best
```

## Kaggle Bundle

If you download a Kaggle March Madness data bundle, place it under:

```bash
historical_pipeline/data/external/kaggle/
```

Or point the pipeline at it with:

```bash
export KAGGLE_MARCH_MADNESS_ROOT=/path/to/march-machine-learning-mania-data
```

The pipeline looks for the standard men’s files:

- `MTeams.csv`
- `MTeamSpellings.csv`
- `MSeasons.csv`
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneySeeds.csv`
- `MMasseyOrdinals.csv`

When those files are present, the pipeline:

- uses Kaggle tournament results instead of the NCAA API fallback
- expands aliases from `MTeamSpellings.csv`
- uses `MMasseyOrdinals.csv` as the preferred historical pre-tournament ordinal source

On this machine, the pipeline also auto-detects the official bundle under:

```bash
/Users/kevinturner/Downloads/march-machine-learning-mania-2026
```

if the repo-local `historical_pipeline/data/external/kaggle/` directory does not contain the Kaggle files yet.

## Local Massey Master CSV

If you have a locally built historical Massey archive export, point the pipeline at it with:

```bash
export MASSEY_MASTER_CSV=/path/to/massey_master.csv
```

Or place it at:

```bash
historical_pipeline/data/external/massey_master.csv
```

The current machine also auto-detects:

```bash
/Users/kevinturner/Documents/Code/Personal/March Matchup — NCAA Tournament Predictor/massey_master.csv
```

When present, the snapshot collector selects the latest dated row on or before each season’s Selection Sunday and adds:

- Massey composite rank
- mean rank
- median rank
- rank dispersion (`stdev`)
- win percentage parsed from `record`

## Outputs

- `historical_pipeline/data/processed/tournament_games.parquet`
- `historical_pipeline/data/processed/season_snapshots.parquet`
- `historical_pipeline/data/processed/tournament_training_games.parquet`
- `historical_pipeline/data/processed/tournament_backtest_by_season.csv`
- `data/models/tournament-consensus-candidate.json`
- `data/models/tournament-consensus-report.json`
- `data/models/tournament-consensus-latest.json` only when promotion is requested and the candidate clears the baseline gates

## Notes

- The collectors prefer direct season pages first.
- Kaggle is the best free structured historical source when available locally.
- The local Massey master CSV is treated as an experimental source family in training. Its features are included in the dataset and report, and compared in `candidate_stack_with_massey_master`, but they are not forced into the default candidate stack unless they materially help.
- If a free source is missing for a season, the snapshot is flagged and the training row keeps explicit missingness indicators.
- The live Massey archive is browser-challenge protected for plain HTTP scraping; cached archive HTML under `historical_pipeline/data/cache/massey_archive/` is supported, and Kaggle `MMasseyOrdinals.csv` is the preferred operational substitute.
