# Court Vision

See the bracket before it happens. NCAA March Madness predictions built from multi-source rating consensus, machine learning, and Monte Carlo simulation.

![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## What It Does

Court Vision combines multiple public college basketball rating systems into a consensus prediction model, then runs thousands of Monte Carlo simulations to project tournament outcomes.

**Three views:**
- **Matchup Predictor** — head-to-head comparison with probability breakdown and feature attributions
- **Daily Slate** — today's games with projected spreads and win probabilities
- **Bracket Simulator** — full 64-team tournament simulation with round-by-round advancement odds

## How It Works

### Prediction Flow

```mermaid
graph TB
    subgraph Training["Historical ML Pipeline (Python, offline)"]
        direction TB
        G["Tournament Games\n9 seasons, 2016–2025"]
        S["Rating Snapshots\nKenPom · BPI · NET · Massey\nEvanMiya · 538"]
        G & S --> D["Feature Engineering\n30+ features per matchup"]
        D --> LOSO["Leave-One-Season-Out\nCross-Validation"]
        LOSO --> |"GBM teacher\n→ Ridge student"| ARTIFACT["Model Artifact (JSON)\ncoefficients · scaler\ncalibration anchors"]
    end

    subgraph Startup["Server Startup"]
        direction TB
        SEED["teams.json\ncurrent-season snapshot"] --> PS["Prediction\nService"]
        ARTIFACT --> PS
        PS --> LATENT["Latent Consensus Model\nPCA + z-score blend\nof 5 rating sources"]
    end

    subgraph Predict["Making a Prediction"]
        direction TB
        REQ["Team A vs Team B\n+ venue + round"] --> COMP

        subgraph COMP["Build 5 Component Opinions"]
            direction LR
            C1["Torvik\nEfficiency"]
            C2["Torvik\nBarthag"]
            C3["ESPN\nBPI"]
            C4["NET\nRating"]
            C5["Resume\n& Form"]
        end

        COMP --> CONSENSUS["Weighted Consensus Margin\neach source votes on expected spread"]
        CONSENSUS --> FEATURES["Feature Vector\nrating diffs · seed interactions\nround context · disagreement"]
        FEATURES --> LEARNED["Learned Model\nRidge margin + logistic win prob"]
        LEARNED --> CALIB

        subgraph CALIB["Calibration"]
            direction LR
            ISO["Isotonic\nGeneral"]
            SGAP["Isotonic\nSeed-Gap ≥5"]
            XGAP["Isotonic\nExtreme ≥8"]
        end

        CALIB --> SHRINK["Shrinkage\nadjust for disagreement\n& source coverage"]
        SHRINK --> PROB["Final Win Probability\n+ margin · spread · confidence tier"]
    end

    subgraph Bracket["Bracket Simulation"]
        direction TB
        FIELD["64-Team Field\n4 regions × 16 seeds"] --> MC["Monte Carlo\n4,000 iterations"]
        MC --> |"each game calls\npredictMatchup\nwith round context"| SIM["Simulate 63 Games\nR64 → R32 → S16 → E8\n→ F4 → Championship"]
        SIM --> ODDS["Round-by-Round\nAdvancement Odds\nper team"]
    end

    PS -.-> Predict
    PS -.-> Bracket

    style Training fill:#f0f4ff,stroke:#4a6fa5
    style Startup fill:#f5f0ff,stroke:#7a5fa5
    style Predict fill:#f0fff4,stroke:#4aa56f
    style Bracket fill:#fff8f0,stroke:#a5874a
    style COMP fill:#e8f0e8,stroke:#4a8a4a
    style CALIB fill:#fef3e0,stroke:#c98a2e
```

The prediction engine blends five component opinions (Torvik efficiency, Torvik Barthag, BPI, NET, resume/form) into a weighted consensus margin. A trained ML model (GBM-distilled Ridge regression) refines this into calibrated win probabilities using features like:

- Multi-source rating differentials (KenPom, EvanMiya, BPI, NET, Massey)
- Seed differential with nonlinear transforms
- Round-aware interactions (seed gaps matter less in later rounds)
- Cross-source disagreement (when ratings conflict, pull toward 50/50)
- Seed-gap-specific calibration (separate calibration curve for upset-prone matchups)
- Temporal weighting (recent seasons contribute more)

The model is trained on 9 seasons of tournament data (2016–2025, excluding 2020) using leave-one-season-out cross-validation.

**Current metrics** (pooled LOSO holdout, 1,132 games):

| Metric | Value |
|--------|-------|
| Log Loss | 0.566 |
| Brier Score | 0.191 |
| Calibration Error | 0.018 |
| Upset Recall | 27.3% |
| Margin MAE | 9.5 pts |

## Stack

- **Frontend:** React, Vite, shadcn/ui, Tailwind, Recharts, Framer Motion, wouter
- **Backend:** Express 5, TypeScript
- **ML Pipeline:** Python, pandas, scikit-learn, duckdb
- **Shared:** TypeScript types and Zod schemas across client and server

## Project Layout

```
client/                React SPA
server/                Express API and runtime prediction engine
shared/                Shared types and request schemas
historical_pipeline/   Data collection, dataset building, model training (Python)
data/models/           Trained model artifacts (JSON)
```

## Getting Started

### Prerequisites

- Node.js 20+
- Python 3.11+ (for the ML pipeline only)

### Development

```bash
npm install
npm run dev        # Express + Vite dev server on http://localhost:5100
```

### Production

```bash
npm run build      # Vite client + esbuild server -> dist/
npm start          # Run production bundle
```

### Type Checking & Tests

```bash
npm run check      # TypeScript type check
npm test           # Node built-in test runner
```

## Historical Model Pipeline

The pipeline collects tournament data, builds training datasets, and produces the learned model artifact used at runtime.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r historical_pipeline/requirements.txt

npm run historical:games      # Stage 1: collect tournament game results
npm run historical:snapshots  # Stage 2: collect pre-tournament rating snapshots
npm run historical:dataset    # Stage 3: build training dataset
npm run historical:train      # Stage 4: train model -> data/models/
```

Use `--promote-if-best` on the train stage to only promote a candidate model when it clears guardrail baselines (must beat both seed-only logit and equal-weight consensus on log loss and Brier score).

### How the Learned Model Works

The promoted model is a **GBM-distilled linear student**:

1. Gradient boosting teacher models capture nonlinear signal from the training data
2. A linear student (Ridge regression) is fit to the teacher's soft outputs
3. The exported artifact stays runtime-friendly — just standardized feature means/scales, linear coefficients, and isotonic calibration anchors

This keeps inference simple in TypeScript while benefiting from the GBM's ability to find interactions.

## Seed Snapshot

The live app uses `client/src/data/teams.json` as the current-season data snapshot. Refresh it with:

```bash
python scripts/merge_data.py
```

Required input files: `torvik_data.json`, `bpi_data.json`, `net_data.json`
Optional: `evanmiya_data.json`, `fivethirtyeight_data.json`

## Runtime Behavior

At runtime, the app loads:
1. Current-season snapshot (`client/src/data/teams.json`)
2. Promoted learned artifact (`data/models/tournament-consensus-latest.json`)
3. Prediction engine (`server/services/prediction-service.ts`)

If no learned artifact is present, the app falls back to a heuristic `latent_public_consensus` model.

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/teams` | GET | Team list for current season |
| `/api/games` | GET | Games for a date with projections |
| `/api/predictions/matchup` | GET | Head-to-head prediction |
| `/api/predictions/bracket` | POST | Monte Carlo bracket simulation |
| `/api/model-runs/latest` | GET | Current model metadata and metrics |

## License

MIT
