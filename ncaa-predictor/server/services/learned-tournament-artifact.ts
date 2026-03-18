import fs from "fs";
import path from "path";
import { z } from "zod";
import { CURRENT_SEASON, type ModelRun, type ModelType, type SourceCoverage } from "@shared/schema";

const calibrationAnchorSchema = z.object({
  raw: z.number(),
  calibrated: z.number(),
});

const learnedArtifactSchema = z.object({
  id: z.string(),
  label: z.string(),
  model_type: z.string(),
  generated_at: z.string(),
  snapshot_policy: z.string(),
  training_seasons: z.array(z.number()),
  data_sources: z.array(z.string()),
  source_coverage: z.object({
    summary: z.array(z.string()),
    seasonsBySource: z.record(z.array(z.number())),
  }),
  source_weights: z.record(z.number()),
  scaler: z.object({
    means: z.record(z.number()),
    scales: z.record(z.number()),
  }),
  margin_model: z.object({
    intercept: z.number(),
    coefficients: z.record(z.number()),
  }),
  win_model: z.object({
    intercept: z.number(),
    coefficients: z.record(z.number()),
  }),
  calibration: z.object({
    anchors: z.array(calibrationAnchorSchema),
    seedGapAnchors: z.array(calibrationAnchorSchema).optional(),
    extremeGapAnchors: z.array(calibrationAnchorSchema).optional(),
  }),
  metrics: z.object({
    logLoss: z.number(),
    brierScore: z.number(),
    marginMae: z.number(),
    calibrationError: z.number(),
    upsetRecall: z.number(),
  }),
  artifact_path: z.string(),
  notes: z.string(),
});

export type LearnedTournamentArtifact = z.infer<typeof learnedArtifactSchema> & {
  model_type: ModelType;
  source_coverage: SourceCoverage;
  metrics: ModelRun["metrics"];
};

function resolveArtifactPath() {
  const candidates = [
    path.resolve(process.cwd(), "dist/models/tournament-consensus-latest.json"),
    path.resolve(process.cwd(), "data/models/tournament-consensus-latest.json"),
  ];

  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}

export function loadLearnedTournamentArtifact(): LearnedTournamentArtifact | null {
  const artifactPath = resolveArtifactPath();
  if (!artifactPath) {
    return null;
  }

  const raw = JSON.parse(fs.readFileSync(artifactPath, "utf-8"));
  const parsed = learnedArtifactSchema.safeParse(raw);
  if (!parsed.success) {
    console.error("[artifact] Learned artifact failed schema validation:", parsed.error.format());
    console.error("[artifact] Falling back to unvalidated load");
    return raw as LearnedTournamentArtifact;
  }
  return parsed.data as LearnedTournamentArtifact;
}

export function buildModelRunFromArtifact(
  artifact: LearnedTournamentArtifact,
  coverage: ModelRun["coverage"],
): ModelRun {
  return {
    id: artifact.id,
    label: artifact.label,
    modelType: artifact.model_type,
    season: CURRENT_SEASON,
    generatedAt: artifact.generated_at,
    notes: artifact.notes,
    trainingSeasons: artifact.training_seasons,
    dataSources: artifact.data_sources,
    scheduleSource: "seeded_rankings",
    snapshotPolicy: artifact.snapshot_policy,
    sourceCoverage: artifact.source_coverage,
    artifactPath: artifact.artifact_path,
    coverage,
    metrics: artifact.metrics,
  };
}
