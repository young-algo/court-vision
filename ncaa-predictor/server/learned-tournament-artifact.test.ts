import test from "node:test";
import assert from "node:assert/strict";
import { buildModelRunFromArtifact, loadLearnedTournamentArtifact } from "./services/learned-tournament-artifact";
import { loadSeedData } from "./services/seed-data";
import { PredictionService } from "./services/prediction-service";

test("learned tournament artifact loads from the bootstrap contract", () => {
  const artifact = loadLearnedTournamentArtifact();

  assert.ok(artifact);
  assert.equal(artifact.model_type, "learned_tournament_consensus");
  assert.ok(artifact.training_seasons.length > 0);
  assert.ok(artifact.calibration.anchors.length >= 6);
});

test("prediction service uses learned artifact metadata when available", () => {
  const artifact = loadLearnedTournamentArtifact();
  const seedData = loadSeedData();

  assert.ok(artifact);

  const modelRun = buildModelRunFromArtifact(artifact, seedData.modelRun.coverage);
  const service = new PredictionService(
    seedData.teams,
    seedData.snapshots,
    modelRun,
    artifact,
  );

  const duke = seedData.teams.find((team) => team.name === "Duke");
  const houston = seedData.teams.find((team) => team.name === "Houston");

  assert.ok(duke);
  assert.ok(houston);

  const result = service.predictMatchup(duke.id, houston.id, "2026-03-14", "neutral", 2025);

  assert.equal(result.modelRun.modelType, "learned_tournament_consensus");
  assert.deepEqual(result.prediction.diagnostics.trainingSeasons, artifact.training_seasons);
  assert.equal(
    result.prediction.diagnostics.calibrationMethod.includes("learned annual tournament consensus"),
    true,
  );
  assert.equal(result.prediction.components.length, 5);
});

