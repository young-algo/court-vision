import { z } from "zod";

export const CURRENT_SEASON = 2026;
export const BRACKET_REGIONS = ["East", "West", "South", "Midwest"] as const;
export const BRACKET_ROUNDS = [
  "round_of_64",
  "round_of_32",
  "sweet_16",
  "elite_8",
  "final_four",
  "championship",
  "champion",
] as const;

export const venueSchema = z.enum(["neutral", "home_a", "home_b"]);
export type Venue = z.infer<typeof venueSchema>;

export const confidenceTierSchema = z.enum(["high", "medium", "low"]);
export type ConfidenceTier = z.infer<typeof confidenceTierSchema>;
export type ModelType =
  | "latent_public_consensus"
  | "learned_tournament_consensus";

export const gameStatusSchema = z.enum(["scheduled", "simulated", "final"]);
export type GameStatus = z.infer<typeof gameStatusSchema>;

export const regionSchema = z.enum(BRACKET_REGIONS);
export type BracketRegion = z.infer<typeof regionSchema>;

export const roundSchema = z.enum(BRACKET_ROUNDS);
export type BracketRound = z.infer<typeof roundSchema>;

export interface TeamRecord {
  wins: number;
  losses: number;
}

export interface TeamBranding {
  primaryColor: string;
  secondaryColor: string;
}

export interface Team {
  id: string;
  season: number;
  slug: string;
  name: string;
  shortName: string;
  conference: string;
  record: TeamRecord;
  recordLabel: string;
  torvikRank: number;
  branding: TeamBranding;
}

export interface TeamRatingSnapshot {
  teamId: string;
  season: number;
  asOfDate: string;
  offense: number;
  defense: number;
  efficiencyMargin: number;
  barthag: number;
  tempo: number;
  bpi: number | null;
  bpiOffense: number | null;
  bpiDefense: number | null;
  bpiRank: number | null;
  netRank: number | null;
  evanmiyaRelativeRating: number | null;
  fivethirtyeightPower: number | null;
  resumeScore: number;
  recentForm: number;
  conferenceStrength: number;
  rosterContinuity: number;
  availabilityScore: number;
}

export interface Game {
  id: string;
  season: number;
  date: string;
  label: string;
  status: GameStatus;
  context: "featured_slate" | "ncaa_tournament";
  round: Exclude<BracketRound, "champion"> | null;
  region: BracketRegion | null;
  teamAId: string;
  teamBId: string;
  venue: Venue;
  scheduleSource: "seeded_rankings";
}

export type PredictionComponentKey =
  | "torvik_efficiency"
  | "torvik_barthag"
  | "bpi"
  | "net"
  | "resume_form";

export type PredictionComponentFamily =
  | "public_rating"
  | "public_rank"
  | "contextual_prior";

export type PredictionComponentAvailability = "observed" | "estimated";

export interface PredictionComponent {
  key: PredictionComponentKey;
  name: string;
  description: string;
  family: PredictionComponentFamily;
  availability: PredictionComponentAvailability;
  weight: number;
  expectedMargin: number;
  winProbabilityA: number;
}

export interface FeatureAttribution {
  label: string;
  impact: number;
  favoredTeamId: string;
  description: string;
}

export interface Prediction {
  teamAId: string;
  teamBId: string;
  date: string;
  venue: Venue;
  expectedMargin: number;
  winProbabilityA: number;
  winProbabilityB: number;
  impliedSpread: string;
  uncertaintyBand: {
    low: number;
    high: number;
  };
  confidenceTier: ConfidenceTier;
  upsetScore: number;
  keyFactors: string[];
  featureAttributions: FeatureAttribution[];
  components: PredictionComponent[];
  diagnostics: {
    sourceCount: number;
    observedSourceCount: number;
    missingSourceKeys: PredictionComponentKey[];
    disagreementIndex: number;
    seasonPhase: number;
    calibrationMethod: string;
    trainingSeasons: number[];
    sourceCoverageSummary: string[];
  };
  modelRunId: string;
  generatedAt: string;
}

export interface GameProjection extends Game {
  teamA: Team;
  teamB: Team;
  teamASnapshot: TeamRatingSnapshot;
  teamBSnapshot: TeamRatingSnapshot;
  prediction: Prediction;
}

export interface ModelRunMetrics {
  logLoss: number;
  brierScore: number;
  marginMae: number;
  calibrationError: number;
  upsetRecall: number;
}

export interface SourceCoverage {
  summary: string[];
  seasonsBySource: Record<string, number[]>;
}

export interface ModelRun {
  id: string;
  label: string;
  modelType: ModelType;
  season: number;
  generatedAt: string;
  notes: string;
  trainingSeasons: number[];
  dataSources: string[];
  scheduleSource: "seeded_rankings";
  snapshotPolicy: string;
  sourceCoverage: SourceCoverage;
  artifactPath: string | null;
  coverage: {
    teams: number;
    fieldSize: number;
    slateDates: string[];
  };
  metrics: ModelRunMetrics;
}

export interface MatchupPredictionResponse {
  teamA: Team;
  teamB: Team;
  snapshotA: TeamRatingSnapshot;
  snapshotB: TeamRatingSnapshot;
  prediction: Prediction;
  modelRun: ModelRun;
}

export interface TeamsResponse {
  season: number;
  teams: Team[];
  modelRun: ModelRun;
}

export interface GamesResponse {
  season: number;
  date: string;
  games: GameProjection[];
  modelRun: ModelRun;
}

export interface BracketEntry {
  teamId: string;
  region: BracketRegion;
  seed: number;
}

export interface BracketRoundOdds {
  teamId: string;
  roundOf32: number;
  sweet16: number;
  elite8: number;
  finalFour: number;
  championship: number;
  champion: number;
}

export interface BracketSimulationResult {
  season: number;
  simulations: number;
  field: BracketEntry[];
  titleOdds: Array<{
    teamId: string;
    probability: number;
  }>;
  roundOdds: BracketRoundOdds[];
  mostLikelyFinalFour: string[];
  mostLikelyChampionship: {
    teamAId: string;
    teamBId: string;
    probability: number;
  };
  modelRun: ModelRun;
  generatedAt: string;
}

export interface ModelRunResponse {
  modelRun: ModelRun;
}

export const teamIdSchema = z.string().min(1);
export const seasonSchema = z.coerce.number().int().min(2000).max(2100);
export const dateSchema = z.string().regex(/^\d{4}-\d{2}-\d{2}$/);

export const matchupQuerySchema = z.object({
  teamAId: teamIdSchema,
  teamBId: teamIdSchema,
  season: seasonSchema.optional().default(CURRENT_SEASON),
  date: dateSchema.optional().default("2026-03-14"),
  venue: venueSchema.optional().default("neutral"),
});

export const gamesQuerySchema = z.object({
  season: seasonSchema.optional().default(CURRENT_SEASON),
  date: dateSchema.optional().default("2026-03-14"),
});

export const teamsQuerySchema = z.object({
  season: seasonSchema.optional().default(CURRENT_SEASON),
});

export const bracketEntrySchema = z.object({
  teamId: teamIdSchema,
  region: regionSchema,
  seed: z.number().int().min(1).max(16),
});

export const bracketRequestSchema = z.object({
  season: seasonSchema.optional().default(CURRENT_SEASON),
  entries: z.array(bracketEntrySchema).length(64).optional(),
  simulations: z.coerce.number().int().min(100).max(20000).optional().default(4000),
  seed: z.coerce.number().int().optional(),
});
