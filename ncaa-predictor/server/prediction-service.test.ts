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

test("seed data builds a seeded field and tournament slate", () => {
  assert.ok(seedData.teams.length >= 60);
  assert.ok(seedData.bracketField.length >= 60);
  assert.ok(seedData.games.length >= 28);
  assert.equal(new Set(seedData.bracketField.map((entry) => entry.teamId)).size, seedData.bracketField.length);
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

test("P(A wins) + P(B wins) = 1 for every team pair in the bracket", () => {
  const field = seedData.bracketField.slice(0, 8);
  for (let i = 0; i < field.length; i++) {
    for (let j = i + 1; j < field.length; j++) {
      const result = predictionService.predictMatchup(
        field[i].teamId,
        field[j].teamId,
        "2026-03-14",
        "neutral",
      );
      const sum = result.prediction.winProbabilityA + result.prediction.winProbabilityB;
      assert.ok(
        Math.abs(sum - 1) < 0.001,
        `P(A)+P(B) = ${sum} for ${field[i].teamId} vs ${field[j].teamId}`,
      );
    }
  }
});

test("win probability is bounded between 0.01 and 0.99", () => {
  for (let i = 0; i < Math.min(10, seedData.teams.length - 1); i++) {
    const result = predictionService.predictMatchup(
      seedData.teams[i].id,
      seedData.teams[i + 1].id,
      "2026-03-14",
      "neutral",
    );
    assert.ok(result.prediction.winProbabilityA >= 0.01);
    assert.ok(result.prediction.winProbabilityA <= 0.99);
    assert.ok(result.prediction.winProbabilityB >= 0.01);
    assert.ok(result.prediction.winProbabilityB <= 0.99);
  }
});

test("expected margin is finite and within reasonable bounds", () => {
  const duke = seedData.teams.find((team) => team.name === "Duke")!;
  const weakest = seedData.teams[seedData.teams.length - 1]!;
  const result = predictionService.predictMatchup(duke.id, weakest.id, "2026-03-14", "neutral");
  assert.ok(Number.isFinite(result.prediction.expectedMargin));
  assert.ok(Math.abs(result.prediction.expectedMargin) < 50);
});

test("confidence tier is one of high, medium, low", () => {
  const validTiers = new Set(["high", "medium", "low"]);
  for (let i = 0; i < Math.min(5, seedData.teams.length - 1); i++) {
    const result = predictionService.predictMatchup(
      seedData.teams[i].id,
      seedData.teams[i + 1].id,
      "2026-03-14",
      "neutral",
    );
    assert.ok(validTiers.has(result.prediction.confidenceTier));
  }
});

test("extreme seed gap matchup produces reasonable upset probability", () => {
  const topSeed = seedData.bracketField.find((e) => e.seed === 1);
  const bottomSeed = seedData.bracketField.find((e) => e.seed === 16 && e.region === topSeed?.region);
  assert.ok(topSeed);
  assert.ok(bottomSeed);

  const result = predictionService.predictMatchup(topSeed.teamId, bottomSeed.teamId, "2026-03-14", "neutral", 2026, 1);
  assert.ok(result.prediction.winProbabilityA > 0.85, `1-seed should be heavily favored but got ${result.prediction.winProbabilityA}`);
  assert.ok(result.prediction.winProbabilityB > 0.01, "16-seed should have nonzero chance");
  assert.ok(result.prediction.winProbabilityB < 0.25, "16-seed should not be given more than 25%");
});

test("home court advantage shifts probabilities in expected direction", () => {
  const teamA = seedData.teams[10];
  const teamB = seedData.teams[11];
  assert.ok(teamA);
  assert.ok(teamB);

  const neutral = predictionService.predictMatchup(teamA.id, teamB.id, "2026-03-14", "neutral");
  const homeA = predictionService.predictMatchup(teamA.id, teamB.id, "2026-03-14", "home_a");
  const homeB = predictionService.predictMatchup(teamA.id, teamB.id, "2026-03-14", "home_b");

  assert.ok(homeA.prediction.winProbabilityA > neutral.prediction.winProbabilityA - 0.001);
  assert.ok(homeB.prediction.winProbabilityB > neutral.prediction.winProbabilityB - 0.001);
});
