# Data Pipeline

## Current state

This repo now serves predictions from the backend instead of bundling the prediction engine into the client. The current data layer is still seeded from a merged 2026 team snapshot, but the serving contract is in place for:

- `/api/teams`
- `/api/games`
- `/api/predictions/matchup`
- `/api/predictions/bracket`
- `/api/model-runs/latest`

The server reads the canonical snapshot from `client/src/data/teams.json` in development and from `dist/data/teams.json` in production builds.

## Refreshing the seed snapshot

The seed snapshot still comes from the three raw sources in the workspace root:

| Source | Output File |
| --- | --- |
| Bart Torvik T-Rank | `torvik_data.json` |
| ESPN BPI | `bpi_data.json` |
| Warren Nolan NET | `net_data.json` |

Run:

```bash
python scripts/merge_data.py
```

That rebuilds `client/src/data/teams.json`, which now feeds both the client and the backend seed services.

## Production build behavior

`npm run build` now:

1. Builds the Vite client
2. Copies `client/src/data/teams.json` to `dist/data/teams.json`
3. Bundles the Express server

This keeps the seed snapshot available to the server after bundling.

## Next pipeline step

The app architecture is ready for a fuller historical pipeline. The next real upgrade is replacing the single merged snapshot with:

- canonical team/game/player IDs
- historical schedules and results
- daily feature snapshots
- trained model artifacts and backtest outputs
