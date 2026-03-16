# NCAA Predictor

NCAA Predictor is a full-stack tournament prediction app with a React frontend, an Express API, and a historical Python training pipeline that produces the learned model artifact used at runtime.

## Stack

- React + Vite
- Express 5
- TypeScript
- React Query + wouter
- Python + pandas + scikit-learn

## Project layout

- `client/` — React SPA
- `server/` — Express API and runtime prediction engine
- `shared/` — shared types and request schemas
- `historical_pipeline/` — data collection, dataset building, and model training
- `data/models/` — trained model artifacts

## Quick start

### App development

```bash
npm install
npm run dev
```

Dev server:
- `http://localhost:5100`

### Typecheck and tests

```bash
npm run check
npm test
```

## Historical model pipeline

Create and activate a virtualenv first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r historical_pipeline/requirements.txt
```

Run the pipeline:

```bash
npm run historical:games
npm run historical:snapshots
npm run historical:dataset
npm run historical:train
```

Promote only if the trained candidate beats the guardrail baselines:

```bash
npm run historical:train -- --promote-if-best
```

### Current learned model

The current promoted learned runtime model is a **GBM-distilled linear student**:

- gradient boosting teacher models capture nonlinear signal
- a linear student is fit to the teacher outputs
- the exported artifact stays runtime-friendly:
  - standardized feature means/scales
  - linear margin coefficients
  - linear win-model coefficients
  - isotonic calibration anchors

This keeps inference simple in TypeScript while improving backtest performance.

## Seed snapshot generation

The live app uses `client/src/data/teams.json` as the current-season seed snapshot.

Refresh it with:

```bash
python scripts/merge_data.py
```

Expected input files live in the workspace root:

Required:
- `torvik_data.json`
- `bpi_data.json`
- `net_data.json`

Optional:
- `evanmiya_data.json`
- `fivethirtyeight_data.json`
- `538_data.json`

When present, EvanMiya and 538 ratings are merged into the runtime snapshot and can be consumed by the learned model.

## Runtime model behavior

At runtime, the app combines:

1. current-season seed snapshot data in `client/src/data/teams.json`
2. a promoted learned artifact in `data/models/tournament-consensus-latest.json`
3. `server/services/prediction-service.ts`

The prediction service supports:
- matchup predictions
- daily game projections
- bracket simulation
- calibrated win probabilities

## API endpoints

- `GET /api/teams`
- `GET /api/games`
- `GET /api/predictions/matchup`
- `POST /api/predictions/bracket`
- `GET /api/model-runs/latest`

## Build

```bash
npm run build
npm start
```

Build output:
- `dist/public/`
- `dist/data/teams.json`
- `dist/models/`
- `dist/index.cjs`

## Notes

- If a promoted learned artifact is present, the app uses `learned_tournament_consensus`.
- Otherwise it falls back to the heuristic `latent_public_consensus` model.
- Historical pipeline details are documented in `historical_pipeline/README.md`.
