import fs from "fs";
import path from "path";
import { CURRENT_SEASON, type ModelRun, type ModelType, type SourceCoverage } from "@shared/schema";

export interface LearnedTournamentArtifact {
  id: string;
  label: string;
  model_type: ModelType;
  generated_at: string;
  snapshot_policy: string;
  training_seasons: number[];
  data_sources: string[];
  source_coverage: SourceCoverage;
  source_weights: Record<string, number>;
  margin_model: {
    intercept: number;
    coefficients: Record<string, number>;
  };
  win_model: {
    intercept: number;
    coefficients: Record<string, number>;
  };
  calibration: {
    anchors: Array<{ raw: number; calibrated: number }>;
  };
  metrics: ModelRun["metrics"];
  artifact_path: string;
  notes: string;
}

function resolveArtifactPath() {
  const candidates = [
    path.resolve(process.cwd(), "dist/models/tournament-consensus-latest.json"),
    path.resolve(process.cwd(), "data/models/tournament-consensus-latest.json"),
  ];

  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}

export function loadLearnedTournamentArtifact() {
  const artifactPath = resolveArtifactPath();
  if (!artifactPath) {
    return null;
  }

  return JSON.parse(fs.readFileSync(artifactPath, "utf-8")) as LearnedTournamentArtifact;
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
