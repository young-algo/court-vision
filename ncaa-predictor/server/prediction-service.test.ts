import test from "node:test";
import assert from "node:assert/strict";
import { PredictionService } from "./services/prediction-service";
import { loadSeedData } from "./services/seed-data";

const seedData = loadSeedData();
const predictionService = new PredictionService(
  seedData.teams,
  seedData.snapshots,
  seedData.modelRun,
);

test("seed data builds a full seeded field and tournament slate", () => {
  assert.ok(seedData.teams.length >= 64);
  assert.equal(seedData.bracketField.length, 64);
  assert.equal(seedData.games.length, 32);
  assert.equal(new Set(seedData.bracketField.map((entry) => entry.teamId)).size, 64);
});

test("favorite projection is directionally correct for strongest vs weakest seeded team", () => {
  const duke = seedData.teams.find((team) => team.name === "Duke");
  const southCarolina = seedData.teams.find((team) => team.name === "South Carolina");

  assert.ok(duke);
  assert.ok(southCarolina);

  const result = predictionService.predictMatchup(duke.id, southCarolina.id, "2026-03-14", "neutral");

  assert.ok(result.prediction.expectedMargin > 0);
  assert.ok(result.prediction.winProbabilityA > 0.75);
  assert.equal(result.prediction.impliedSpread.startsWith("Duke"), true);
  assert.equal(result.prediction.components.length, 5);
  assert.equal(result.prediction.diagnostics.sourceCount, 5);
  assert.ok(
    Math.abs(
      result.prediction.components.reduce((sum, component) => sum + component.weight, 0) - 1,
    ) < 0.000001,
  );
});

test("neutral-site predictions are approximately symmetric when team order flips", () => {
  const duke = seedData.teams.find((team) => team.name === "Duke");
  const houston = seedData.teams.find((team) => team.name === "Houston");

  assert.ok(duke);
  assert.ok(houston);

  const forward = predictionService.predictMatchup(duke.id, houston.id, "2026-03-14", "neutral");
  const reverse = predictionService.predictMatchup(houston.id, duke.id, "2026-03-14", "neutral");

  assert.ok(Math.abs(forward.prediction.expectedMargin + reverse.prediction.expectedMargin) < 0.001);
  assert.ok(
    Math.abs(forward.prediction.winProbabilityA - reverse.prediction.winProbabilityB) < 0.001,
  );
});

test("consensus model tracks estimated public sources when a ranking is missing", () => {
  const duke = seedData.teams.find((team) => team.name === "Duke");
  const cincinnati = seedData.teams.find((team) => team.name === "Cincinnati");

  assert.ok(duke);
  assert.ok(cincinnati);

  const result = predictionService.predictMatchup(cincinnati.id, duke.id, "2026-03-14", "neutral");
  const bpiComponent = result.prediction.components.find((component) => component.key === "bpi");

  assert.ok(bpiComponent);
  assert.equal(bpiComponent.availability, "estimated");
  assert.ok(result.prediction.diagnostics.missingSourceKeys.includes("bpi"));
  assert.ok(result.prediction.diagnostics.observedSourceCount < result.prediction.diagnostics.sourceCount);
});

test("bracket simulation odds are normalized and monotonic by round", () => {
  const result = predictionService.simulateBracket(seedData.bracketField, 1200, 2026);
  const championSum = result.roundOdds.reduce((sum, team) => sum + team.champion, 0);

  assert.ok(Math.abs(championSum - 1) < 0.000001);
  assert.equal(result.mostLikelyFinalFour.length, 4);
  assert.ok(result.titleOdds[0].probability > 0);

  result.roundOdds.forEach((team) => {
    assert.ok(team.roundOf32 >= team.sweet16);
    assert.ok(team.sweet16 >= team.elite8);
    assert.ok(team.elite8 >= team.finalFour);
    assert.ok(team.finalFour >= team.championship);
    assert.ok(team.championship >= team.champion);
  });
});
