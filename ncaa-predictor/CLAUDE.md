# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Development
npm run dev              # Start Express + Vite dev server on port 5100

# Build & production
npm run build            # Vite client build + esbuild server bundle → dist/
npm start                # Run production bundle (dist/index.cjs)

# Type checking & tests
npm run check            # tsc (no emit)
npm test                 # Node built-in test runner: server/**/*.test.ts

# Historical ML pipeline (requires .venv activated)
source .venv/bin/activate
pip install -r historical_pipeline/requirements.txt
npm run historical:games      # Stage 1: collect tournament game results
npm run historical:snapshots  # Stage 2: collect pre-tournament rating snapshots
npm run historical:dataset    # Stage 3: build training dataset (parquet)
npm run historical:train      # Stage 4: train model → data/models/

# Refresh seed snapshot from raw source files
python scripts/merge_data.py  # → client/src/data/teams.json

# Database (optional, not actively used)
npm run db:push          # Drizzle push schema to PostgreSQL (needs DATABASE_URL)
```

## Architecture

### Monorepo layout

Single `package.json` at root. Three code zones share TypeScript via path aliases (`@/*` → `client/src/*`, `@shared/*` → `shared/*`):

- **`shared/schema.ts`** — Single source of truth for all types and Zod request schemas. Both client and server import from here.
- **`server/`** — Express 5 REST API. In dev, Vite middleware serves client assets; in prod, serves static files from `dist/public/`.
- **`client/`** — React SPA (Vite, wouter hash router, React Query, shadcn/ui with Radix primitives, Tailwind, Recharts, Framer Motion).
- **`historical_pipeline/`** — Python (pandas, scikit-learn, duckdb) ML pipeline that produces trained model JSON artifacts consumed by the server.

### Data flow

1. **Seed data** (`client/src/data/teams.json`) — merged Torvik/BPI/NET snapshot loaded at server startup via `server/services/seed-data.ts`. Provides teams, ratings, games, and bracket field.
2. **Learned artifact** (`data/models/tournament-consensus-latest.json`) — sklearn model exported as JSON (scaler params, linear regression coefficients, calibration anchors). Loaded by `server/services/learned-tournament-artifact.ts`.
3. **PredictionService** (`server/services/prediction-service.ts`) — core engine combining 5 component opinions (Torvik efficiency, Torvik barthag, BPI, NET, resume/form) into weighted consensus margin → normal CDF win probability → calibration curve interpolation. Also runs Monte Carlo bracket simulations.

### API routes (`server/routes.ts`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/teams` | GET | Team list for season |
| `/api/games` | GET | Games for date with projections |
| `/api/predictions/matchup` | GET | Head-to-head prediction |
| `/api/predictions/bracket` | POST | Monte Carlo bracket simulation |
| `/api/model-runs/latest` | GET | Current model metadata/metrics |

### Client pages

- **Matchup Predictor** — head-to-head team comparison with prediction breakdown
- **Daily Slate** — day's games with projections
- **Bracket Simulator** — full 64-team tournament Monte Carlo simulation

### ML pipeline stages

The historical pipeline runs as four sequential stages, each as a Python module (`python3 -m historical_pipeline.<stage>`). It supports multiple data sources: Kaggle March Madness bundle (preferred), NCAA API, and optional local Massey master CSV. Output artifacts land in `data/models/`. Use `--promote-if-best` on train_model to only promote a candidate when it clears guardrail baselines.

### Build output

`npm run build` produces `dist/` with:
- `dist/public/` — Vite client bundle
- `dist/data/teams.json` — seed snapshot copied from client
- `dist/models/` — trained model artifact (if available)
- `dist/index.cjs` — esbuild-bundled Express server (CommonJS)

### Two model types

- `latent_public_consensus` — original heuristic-weighted consensus (no trained artifact needed)
- `learned_tournament_consensus` — uses trained sklearn artifact with calibration; preferred when artifact is present
