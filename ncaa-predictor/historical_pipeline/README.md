# Historical Tournament Pipeline

This package builds the learned NCAA tournament model used by the app runtime.

## What it does

The pipeline:

1. collects historical NCAA tournament game results
2. collects pre-tournament snapshot features by season
3. builds a training dataset with matchup-level engineered features
4. trains and backtests multiple candidate models
5. promotes a runtime artifact only when it beats guardrail baselines

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r historical_pipeline/requirements.txt
```

## Pipeline commands

```bash
python3 -m historical_pipeline.collect_tournament_games
python3 -m historical_pipeline.collect_snapshots
python3 -m historical_pipeline.build_training_dataset
python3 -m historical_pipeline.train_model
```

Promote only if the best candidate clears the baselines:

```bash
python3 -m historical_pipeline.train_model --promote-if-best
```

## Current training strategy

The training script evaluates several model families, including:

- `candidate_stack`
- `reduced_candidate_stack`
- `optimized_candidate_stack`
- `candidate_stack_with_massey_master`
- `gbm_distilled_candidate_stack`
- several baselines (`seed_only_logit`, `equal_weight_consensus`, single-source models, current compat stack)

### GBM distillation

The current best-performing learned candidate is a **GBM-distilled linear student**.

That means:

- Gradient boosting teacher models are trained for margin and win probability.
- Their predictions are blended with observed targets.
- Linear student models are fit on top of the same standardized features.
- The final exported artifact remains TypeScript-runtime-friendly:
  - linear coefficients
  - scaler params
  - calibration anchors

This preserves the simple JSON artifact contract while capturing nonlinear signal during training.

## Feature families

The pipeline can use:

- rank percentile diffs
  - BPI
  - NET
  - Massey ordinal
  - optional Massey master derived ranks
- power diffs
  - KenPom/Barttorvik adjusted efficiency margin
  - EvanMiya relative rating
  - 538 power
  - optional Massey win pct
- resume blend features
- missingness indicators
- context features
  - `seed_diff`
  - `same_conference`
  - availability counts
- interaction features
  - `seed_kenpom_interaction`
  - `abs_seed_same_conference_interaction`

The optimized/distilled stack intentionally excludes direct `seed_diff` from the win model while keeping it in the margin model.

## Guardrail promotion

Promotion requires the selected candidate to beat both:

- `equal_weight_consensus`
- `seed_only_logit`

on:

- log loss
- Brier score

If promotion is requested and the best candidate clears both guardrails, it is copied to:

- `data/models/tournament-consensus-latest.json`

## Data sources

### Kaggle bundle

If you download a Kaggle March Madness data bundle, place it under:

```bash
historical_pipeline/data/external/kaggle/
```

Or point the pipeline at it with:

```bash
export KAGGLE_MARCH_MADNESS_ROOT=/path/to/march-machine-learning-mania-data
```

The pipeline looks for standard men’s files such as:

- `MTeams.csv`
- `MTeamSpellings.csv`
- `MSeasons.csv`
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneySeeds.csv`
- `MMasseyOrdinals.csv`

When present, Kaggle is the preferred structured historical source.

### Local Massey master CSV

If you have a locally built historical Massey archive export, point the pipeline at it with:

```bash
export MASSEY_MASTER_CSV=/path/to/massey_master.csv
```

Or place it at:

```bash
historical_pipeline/data/external/massey_master.csv
```

When available, it adds experimental Massey-derived features such as:

- composite rank
- mean rank
- median rank
- rank dispersion
- win percentage from record parsing

## Outputs

Processed data:

- `historical_pipeline/data/processed/tournament_games.parquet`
- `historical_pipeline/data/processed/season_snapshots.parquet`
- `historical_pipeline/data/processed/tournament_training_games.parquet`
- `historical_pipeline/data/processed/tournament_backtest_by_season.csv`

Model artifacts:

- `data/models/tournament-consensus-candidate.json`
- `data/models/tournament-consensus-report.json`
- `data/models/tournament-consensus-latest.json` when promoted

## Reports

The training report includes:

- aggregate metrics by model
- seasonal backtest rows
- calibration buckets
- feature importance
- source coverage
- error slices
- promotion decision metadata

## Notes

- Missing historical inputs are handled explicitly with missingness indicators.
- The runtime calibration export preserves a neutral `0.5 -> 0.5` anchor.
- The live app consumes only the promoted JSON artifact; Python training dependencies are not needed at runtime.
