import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { PredictionService } from "./services/prediction-service";
import { loadSeedData } from "./services/seed-data";
import {
  loadLearnedTournamentArtifact,
  type LearnedTournamentArtifact,
} from "./services/learned-tournament-artifact";

const seedData = loadSeedData();

function loadArtifactWithReferences(): (LearnedTournamentArtifact & { referenceFeatures?: Array<Record<string, number>> }) | null {
  const candidates = [
    path.resolve(process.cwd(), "dist/models/tournament-consensus-latest.json"),
    path.resolve(process.cwd(), "data/models/tournament-consensus-latest.json"),
    path.resolve(process.cwd(), "data/models/tournament-consensus-candidate.json"),
  ];
  const artifactPath = candidates.find((p) => fs.existsSync(p));
  if (!artifactPath) return null;
  return JSON.parse(fs.readFileSync(artifactPath, "utf-8"));
}

test("learned artifact scaler means and scales have matching feature keys", () => {
  const artifact = loadArtifactWithReferences();
  if (!artifact) {
    return;
  }

  const marginKeys = Object.keys(artifact.margin_model.coefficients);
  const winKeys = Object.keys(artifact.win_model.coefficients);
  const scalerMeanKeys = Object.keys(artifact.scaler.means);
  const scalerScaleKeys = Object.keys(artifact.scaler.scales);

  for (const key of marginKeys) {
    assert.ok(scalerMeanKeys.includes(key), `margin coefficient key "${key}" missing from scaler means`);
    assert.ok(scalerScaleKeys.includes(key), `margin coefficient key "${key}" missing from scaler scales`);
  }

  for (const key of winKeys) {
    assert.ok(scalerMeanKeys.includes(key), `win coefficient key "${key}" missing from scaler means`);
    assert.ok(scalerScaleKeys.includes(key), `win coefficient key "${key}" missing from scaler scales`);
  }
});

test("TS feature map produces all keys required by the learned artifact", () => {
  const artifact = loadArtifactWithReferences();
  if (!artifact) {
    return;
  }

  const predictionService = new PredictionService(
    seedData.teams,
    seedData.snapshots,
    seedData.modelRun,
    artifact,
  );

  const teamA = seedData.teams[0];
  const teamB = seedData.teams[1];
  assert.ok(teamA);
  assert.ok(teamB);

  const result = predictionService.predictMatchup(teamA.id, teamB.id, "2026-03-14", "neutral", 2026, 1);

  const requiredKeys = [
    ...Object.keys(artifact.margin_model.coefficients),
    ...Object.keys(artifact.win_model.coefficients),
  ];
  const uniqueKeys = [...new Set(requiredKeys)];

  for (const key of uniqueKeys) {
    assert.ok(
      key in artifact.scaler.means,
      `Feature "${key}" is in model coefficients but missing from scaler -- potential parity issue`,
    );
  }

  assert.ok(result.prediction.winProbabilityA >= 0.01);
  assert.ok(result.prediction.winProbabilityA <= 0.99);
  assert.ok(result.prediction.winProbabilityA + result.prediction.winProbabilityB > 0.99);
  assert.ok(result.prediction.winProbabilityA + result.prediction.winProbabilityB < 1.01);
});

test("calibration anchors are approximately monotonically increasing", () => {
  const artifact = loadArtifactWithReferences();
  if (!artifact) {
    return;
  }

  for (const anchors of [artifact.calibration.anchors, artifact.calibration.seedGapAnchors ?? [], artifact.calibration.extremeGapAnchors ?? []]) {
    if (anchors.length < 2) continue;
    const sorted = [...anchors].sort((a, b) => a.raw - b.raw);
    let violations = 0;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].calibrated < sorted[i - 1].calibrated - 0.02) {
        violations++;
      }
    }
    assert.ok(
      violations <= 1,
      `Too many monotonicity violations (${violations}) in calibration anchors`,
    );
  }
});
