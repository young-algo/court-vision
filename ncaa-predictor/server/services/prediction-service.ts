import {
  BRACKET_REGIONS,
  type BracketEntry,
  type BracketRegion,
  type BracketSimulationResult,
  type ConfidenceTier,
  type FeatureAttribution,
  type Game,
  type GameProjection,
  type ModelRun,
  type Prediction,
  type PredictionComponent,
  type PredictionComponentAvailability,
  type Team,
  type TeamRatingSnapshot,
  type Venue,
} from "@shared/schema";
import type { LearnedTournamentArtifact } from "./learned-tournament-artifact";

const D1_AVG_EFFICIENCY = 100;
const HOME_COURT_ADVANTAGE = 3.4;
const SIGMA = 11.1;

/**
 * Matchup-dependent standard deviation.
 * High-tempo games have more possessions → higher variance.
 * Later tournament rounds feature more evenly-matched teams → higher variance.
 * Both effects are modest and additive on the base SIGMA.
 */
function matchupSigma(
  tempoA: number,
  tempoB: number,
  tournamentRound: number,
): number {
  const avgTempo = (tempoA + tempoB) / 2;
  const tempoFactor = (avgTempo - 67) * 0.03;
  const roundFactor = tournamentRound > 0 ? 0.4 * Math.sqrt(tournamentRound) : 0;
  return SIGMA + tempoFactor + roundFactor;
}
const ROUND_ONE_PAIRINGS: Array<[number, number]> = [
  [1, 16],
  [8, 9],
  [5, 12],
  [4, 13],
  [6, 11],
  [3, 14],
  [7, 10],
  [2, 15],
];

const CALIBRATION_ANCHORS: Array<{ raw: number; calibrated: number }> = [
  { raw: 0.02, calibrated: 0.04 },
  { raw: 0.08, calibrated: 0.10 },
  { raw: 0.15, calibrated: 0.17 },
  { raw: 0.25, calibrated: 0.26 },
  { raw: 0.35, calibrated: 0.35 },
  { raw: 0.5, calibrated: 0.5 },
  { raw: 0.65, calibrated: 0.65 },
  { raw: 0.75, calibrated: 0.74 },
  { raw: 0.85, calibrated: 0.83 },
  { raw: 0.92, calibrated: 0.90 },
  { raw: 0.98, calibrated: 0.96 },
];

type InternalOpinion = Omit<PredictionComponent, "weight"> & {
  baseWeight: number;
  reliability: number;
};

const SOURCE_KEYS: PredictionComponent["key"][] = [
  "torvik_efficiency",
  "torvik_barthag",
  "bpi",
  "net",
  "resume_form",
];

type LatentConsensusModel = {
  componentWeights: Record<PredictionComponent["key"], number>;
  teamScores: Map<string, number>;
  method: string;
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function erf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  const input = Math.abs(x);
  const t = 1 / (1 + p * input);
  const y =
    1 -
    (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) *
      t *
      Math.exp(-input * input));
  return sign * y;
}

function normalCdf(x: number, mean: number, sigma: number) {
  return 0.5 * (1 + erf((x - mean) / (sigma * Math.SQRT2)));
}

function inverseNormalCdf(probability: number) {
  const a = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  if (probability < pLow) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }

  if (probability <= pHigh) {
    const q = probability - 0.5;
    const r = q * q;
    return (
      (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
      q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }

  const q = Math.sqrt(-2 * Math.log(1 - probability));
  return -(
    (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

function interpolateCalibration(
  rawProbability: number,
  anchors: Array<{ raw: number; calibrated: number }> = CALIBRATION_ANCHORS,
) {
  if (rawProbability <= anchors[0]!.raw) {
    return anchors[0]!.calibrated;
  }

  if (rawProbability >= anchors[anchors.length - 1]!.raw) {
    return anchors[anchors.length - 1]!.calibrated;
  }

  for (let index = 1; index < anchors.length; index += 1) {
    const left = anchors[index - 1]!;
    const right = anchors[index]!;
    if (rawProbability <= right.raw) {
      const t = (rawProbability - left.raw) / (right.raw - left.raw);
      return left.calibrated + t * (right.calibrated - left.calibrated);
    }
  }

  return rawProbability;
}

function standardDeviation(values: number[]) {
  if (values.length <= 1) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function dot(left: number[], right: number[]) {
  return left.reduce((sum, value, index) => sum + value * right[index]!, 0);
}

function l2Normalize(values: number[]) {
  const magnitude = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
  return values.map((value) => value / magnitude);
}

function estimatedBpiSignal(snapshot: TeamRatingSnapshot) {
  return (
    snapshot.efficiencyMargin * 0.74 +
    snapshot.conferenceStrength * 0.035 +
    (snapshot.recentForm - 50) * 0.06
  );
}

function resumeSignal(snapshot: TeamRatingSnapshot) {
  return (
    snapshot.resumeScore * 0.41 +
    snapshot.recentForm * 0.26 +
    snapshot.conferenceStrength * 0.14 +
    snapshot.rosterContinuity * 100 * 0.09 +
    snapshot.availabilityScore * 100 * 0.1
  );
}

function fallbackNetPower(snapshot: TeamRatingSnapshot) {
  return (
    snapshot.efficiencyMargin * 0.58 +
    snapshot.resumeScore * 0.14 +
    snapshot.conferenceStrength * 0.04
  );
}

function powerIteration(correlationMatrix: number[][], iterations = 24) {
  let vector = Array.from({ length: correlationMatrix.length }, () => 1 / correlationMatrix.length);

  for (let index = 0; index < iterations; index += 1) {
    const nextVector = correlationMatrix.map((row) => dot(row, vector));
    vector = l2Normalize(nextVector);
  }

  return vector;
}

function homeCourtAdjustment(venue: Venue) {
  if (venue === "home_a") return HOME_COURT_ADVANTAGE;
  if (venue === "home_b") return -HOME_COURT_ADVANTAGE;
  return 0;
}

function formatSpread(teamA: Team, teamB: Team, margin: number) {
  if (Math.abs(margin) < 0.25) {
    return "Pick'em";
  }
  const favored = margin >= 0 ? teamA.shortName : teamB.shortName;
  return `${favored} -${Math.abs(margin).toFixed(1)}`;
}

function seededRng(seed: number) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function probabilityToMargin(probability: number) {
  return inverseNormalCdf(clamp(probability, 0.01, 0.99)) * SIGMA;
}

function rankToPower(rank: number | null, fallback: number) {
  if (rank == null) {
    return fallback;
  }

  const percentile = clamp((101 - rank) / 100, 0, 1);
  return 4 + 17 * percentile + 7 * Math.sqrt(percentile);
}

function seasonPhase(teamA: Team, teamB: Team) {
  const gamesA = teamA.record.wins + teamA.record.losses;
  const gamesB = teamB.record.wins + teamB.record.losses;
  return clamp((gamesA + gamesB) / 2 / 34, 0, 1);
}

function availabilityReliability(availability: PredictionComponentAvailability) {
  return availability === "observed" ? 1 : 0.58;
}

export class PredictionService {
  private readonly teamsById = new Map<string, Team>();
  private readonly snapshotsByTeamId = new Map<string, TeamRatingSnapshot>();
  private readonly latentConsensus: LatentConsensusModel;
  private readonly learnedArtifact: LearnedTournamentArtifact | null;

  constructor(
    private readonly teams: Team[],
    private readonly snapshots: TeamRatingSnapshot[],
    private readonly modelRun: ModelRun,
    learnedArtifact: LearnedTournamentArtifact | null = null,
  ) {
    teams.forEach((team) => this.teamsById.set(team.id, team));
    snapshots.forEach((snapshot) =>
      this.snapshotsByTeamId.set(snapshot.teamId, snapshot),
    );
    this.latentConsensus = this.buildLatentConsensusModel();
    this.learnedArtifact = learnedArtifact;
  }

  getTeam(teamId: string) {
    return this.teamsById.get(teamId);
  }

  getSnapshot(teamId: string) {
    return this.snapshotsByTeamId.get(teamId);
  }

  listTeams() {
    return this.teams;
  }

  getModelRun() {
    return this.modelRun;
  }

  projectGames(games: Game[]) {
    return games.map((game) => this.projectGame(game));
  }

  projectGame(game: Game): GameProjection {
    const matchup = this.predictMatchup(
      game.teamAId,
      game.teamBId,
      game.date,
      game.venue,
    );

    return {
      ...game,
      teamA: matchup.teamA,
      teamB: matchup.teamB,
      teamASnapshot: matchup.snapshotA,
      teamBSnapshot: matchup.snapshotB,
      prediction: matchup.prediction,
    };
  }

  predictMatchup(
    teamAId: string,
    teamBId: string,
    date: string,
    venue: Venue,
    season = this.modelRun.season,
    tournamentRound = 0,
  ) {
    const teamA = this.requireTeam(teamAId);
    const teamB = this.requireTeam(teamBId);
    const snapshotA = this.requireSnapshot(teamAId);
    const snapshotB = this.requireSnapshot(teamBId);
    const phase = seasonPhase(teamA, teamB);

    const components = this.buildConsensusComponents(
      teamA,
      teamB,
      snapshotA,
      snapshotB,
      venue,
    );
    const totalWeight = components.reduce(
      (sum, component) => sum + component.baseWeight * component.reliability,
      0,
    );
    const normalizedComponents: PredictionComponent[] = components.map((component) => ({
      ...component,
      weight: (component.baseWeight * component.reliability) / totalWeight,
    }));

    const sourceConsensusMargin = normalizedComponents.reduce(
      (sum, component) => sum + component.expectedMargin * component.weight,
      0,
    );
    const latentScoreDiff =
      this.latentConsensus.teamScores.get(teamAId)! -
      this.latentConsensus.teamScores.get(teamBId)!;
    const latentConsensusMargin = latentScoreDiff * 5.75 + homeCourtAdjustment(venue) * 0.25;
    const sameConference = teamA.conference === teamB.conference ? 1 : 0;
    const estimatedSeedA = clamp(Math.ceil(teamA.torvikRank / 4), 1, 16);
    const estimatedSeedB = clamp(Math.ceil(teamB.torvikRank / 4), 1, 16);
    const seedDiff = estimatedSeedA - estimatedSeedB;
    const expectedMargin = this.learnedArtifact
      ? this.learnedExpectedMargin(
          normalizedComponents,
          sourceConsensusMargin,
          latentConsensusMargin,
          seedDiff,
          sameConference,
          snapshotA,
          snapshotB,
          teamA,
          teamB,
          tournamentRound,
        )
      : sourceConsensusMargin * 0.72 + latentConsensusMargin * 0.28;
    const disagreementIndex = standardDeviation(
      normalizedComponents.map((component) => component.expectedMargin),
    );
    const observedSourceCount = normalizedComponents.filter(
      (component) => component.availability === "observed",
    ).length;
    // Use matchup-dependent sigma: wider for high-tempo games and later rounds.
    // Also widen for extreme seed gaps (|gap| >= 8) where upsets are empirically more common.
    const baseSigma = matchupSigma(snapshotA.tempo, snapshotB.tempo, tournamentRound);
    const adjustedSigma = baseSigma + disagreementIndex * 0.2 + (Math.abs(seedDiff) >= 8 ? 1.5 : 0);
    const rawProbabilityA = clamp(
      1 - normalCdf(0, expectedMargin, adjustedSigma),
      0.01,
      0.99,
    );
    const winProbabilityA = this.calibratedProbability(
      rawProbabilityA,
      disagreementIndex,
      observedSourceCount,
      normalizedComponents.length,
      normalizedComponents,
      seedDiff,
      sameConference,
      snapshotA,
      snapshotB,
      teamA,
      teamB,
      tournamentRound,
    );
    const winProbabilityB = 1 - winProbabilityA;
    const bandRadius =
      8 +
      disagreementIndex * 0.75 +
      (normalizedComponents.length - observedSourceCount) * 0.9;
    const confidenceTier = this.confidenceTier(
      expectedMargin,
      disagreementIndex,
      winProbabilityA,
      observedSourceCount / normalizedComponents.length,
    );
    const featureAttributions = this.buildFeatureAttributions(
      teamA,
      teamB,
      snapshotA,
      snapshotB,
      venue,
      normalizedComponents,
      disagreementIndex,
      observedSourceCount,
    );
    const keyFactors = featureAttributions.slice(0, 4).map((factor) => factor.description);
    const lowerRankWinProbability =
      teamA.torvikRank > teamB.torvikRank ? winProbabilityA : winProbabilityB;
    const upsetScore = clamp(
      lowerRankWinProbability * 100 * (1 + Math.abs(teamA.torvikRank - teamB.torvikRank) / 96),
      0,
      100,
    );

    const prediction: Prediction = {
      teamAId,
      teamBId,
      date,
      venue,
      expectedMargin,
      winProbabilityA,
      winProbabilityB,
      impliedSpread: formatSpread(teamA, teamB, probabilityToMargin(winProbabilityA)),
      uncertaintyBand: {
        low: expectedMargin - bandRadius,
        high: expectedMargin + bandRadius,
      },
      confidenceTier,
      upsetScore,
      keyFactors,
      featureAttributions,
      components: normalizedComponents,
      diagnostics: {
        sourceCount: normalizedComponents.length,
        observedSourceCount,
        missingSourceKeys: normalizedComponents
          .filter((component) => component.availability === "estimated")
          .map((component) => component.key),
        disagreementIndex,
        seasonPhase: phase,
        calibrationMethod: `${this.learnedArtifact ? "learned annual tournament consensus" : this.latentConsensus.method} + ${this.learnedArtifact?.snapshot_policy ?? "piecewise"} isotonic shrinkage`,
        trainingSeasons: this.modelRun.trainingSeasons,
        sourceCoverageSummary: this.modelRun.sourceCoverage.summary,
      },
      modelRunId: this.modelRun.id,
      generatedAt: this.modelRun.generatedAt,
    };

    return {
      teamA,
      teamB,
      snapshotA,
      snapshotB,
      prediction,
      modelRun: this.modelRun,
    };
  }

  simulateBracket(
    field: BracketEntry[],
    simulations = 4000,
    seed = 2026,
  ): BracketSimulationResult {
    const rng = seededRng(seed);
    const counters = new Map<
      string,
      {
        roundOf32: number;
        sweet16: number;
        elite8: number;
        finalFour: number;
        championship: number;
        champion: number;
      }
    >();
    const finalsCounter = new Map<string, number>();

    field.forEach((entry) => {
      counters.set(entry.teamId, {
        roundOf32: 0,
        sweet16: 0,
        elite8: 0,
        finalFour: 0,
        championship: 0,
        champion: 0,
      });
    });

    for (let iteration = 0; iteration < simulations; iteration += 1) {
      const regionalChampions = BRACKET_REGIONS.map((region) =>
        this.simulateRegion(region, field, counters, rng),
      );

      const semiOne = this.simulateNeutralGame(regionalChampions[0], regionalChampions[1], rng, 5);
      const semiTwo = this.simulateNeutralGame(regionalChampions[2], regionalChampions[3], rng, 5);
      counters.get(semiOne)!.championship += 1;
      counters.get(semiTwo)!.championship += 1;

      const finalists = [semiOne, semiTwo];
      const matchupKey = [...finalists].sort().join("::");
      finalsCounter.set(matchupKey, (finalsCounter.get(matchupKey) ?? 0) + 1);

      const champion = this.simulateNeutralGame(semiOne, semiTwo, rng, 6);
      counters.get(champion)!.champion += 1;
    }

    const roundOdds = Array.from(counters.entries())
      .map(([teamId, counts]) => ({
        teamId,
        roundOf32: counts.roundOf32 / simulations,
        sweet16: counts.sweet16 / simulations,
        elite8: counts.elite8 / simulations,
        finalFour: counts.finalFour / simulations,
        championship: counts.championship / simulations,
        champion: counts.champion / simulations,
      }))
      .sort((left, right) => right.champion - left.champion);

    const titleOdds = roundOdds.slice(0, 16).map(({ teamId, champion }) => ({
      teamId,
      probability: champion,
    }));

    const mostLikelyFinalFour = BRACKET_REGIONS.map((region) => {
      const regionTeams = field
        .filter((entry) => entry.region === region)
        .map((entry) => entry.teamId);
      return roundOdds
        .filter((odds) => regionTeams.includes(odds.teamId))
        .sort((left, right) => right.finalFour - left.finalFour)[0]!.teamId;
    });

    const [mostLikelyMatchupKey, mostLikelyMatchupCount] =
      Array.from(finalsCounter.entries()).sort((left, right) => right[1] - left[1])[0] ?? [
        `${field[0]?.teamId ?? ""}::${field[1]?.teamId ?? ""}`,
        0,
      ];
    const [teamAId, teamBId] = mostLikelyMatchupKey.split("::");

    return {
      season: this.modelRun.season,
      simulations,
      field,
      titleOdds,
      roundOdds,
      mostLikelyFinalFour,
      mostLikelyChampionship: {
        teamAId,
        teamBId,
        probability: mostLikelyMatchupCount / simulations,
      },
      modelRun: this.modelRun,
      generatedAt: new Date().toISOString(),
    };
  }

  private buildLatentConsensusModel(): LatentConsensusModel {
    const rows = this.snapshots.map((snapshot) => this.snapshotSourceStrengths(snapshot));
    const zScoreColumns = SOURCE_KEYS.map((key) => {
      const values = rows.map((row) => row[key]);
      const center = mean(values);
      const spread = standardDeviation(values) || 1;
      return {
        key,
        center,
        spread,
        values: values.map((value) => (value - center) / spread),
      };
    });

    const correlationMatrix = SOURCE_KEYS.map((leftKey, leftIndex) =>
      SOURCE_KEYS.map((rightKey, rightIndex) => {
        const leftValues = zScoreColumns[leftIndex]!.values;
        const rightValues = zScoreColumns[rightIndex]!.values;
        return dot(leftValues, rightValues) / leftValues.length;
      }),
    );

    let pcaVector = powerIteration(correlationMatrix);
    if (pcaVector[0]! < 0) {
      pcaVector = pcaVector.map((value) => value * -1);
    }

    const pcaWeights = l2Normalize(pcaVector.map((value) => Math.abs(value)));
    const equalWeight = 1 / SOURCE_KEYS.length;
    const componentWeights = SOURCE_KEYS.reduce(
      (weights, key, index) => {
        weights[key] = 0.5 * equalWeight + 0.5 * pcaWeights[index]!;
        return weights;
      },
      {} as Record<PredictionComponent["key"], number>,
    );
    const totalWeight = Object.values(componentWeights).reduce((sum, value) => sum + value, 0);
    SOURCE_KEYS.forEach((key) => {
      componentWeights[key] /= totalWeight;
    });

    const teamScores = new Map<string, number>();
    this.snapshots.forEach((snapshot, rowIndex) => {
      const row = SOURCE_KEYS.map(
        (_key, sourceIndex) => zScoreColumns[sourceIndex]!.values[rowIndex]!,
      );
      const zAverage = mean(row);
      const pcaScore = dot(pcaVector, row);
      teamScores.set(snapshot.teamId, 0.5 * zAverage + 0.5 * pcaScore);
    });

    return {
      componentWeights,
      teamScores,
      method: "z-score/PCA latent blend",
    };
  }

  private snapshotSourceStrengths(snapshot: TeamRatingSnapshot) {
    return {
      torvik_efficiency: snapshot.efficiencyMargin,
      torvik_barthag: probabilityToMargin(snapshot.barthag),
      bpi: snapshot.bpi ?? estimatedBpiSignal(snapshot),
      net: rankToPower(snapshot.netRank, fallbackNetPower(snapshot)),
      resume_form: resumeSignal(snapshot) * 0.16,
    } as Record<PredictionComponent["key"], number>;
  }

  private buildConsensusComponents(
    teamA: Team,
    teamB: Team,
    snapshotA: TeamRatingSnapshot,
    snapshotB: TeamRatingSnapshot,
    venue: Venue,
  ): InternalOpinion[] {
    const learnedWeights = this.learnedArtifact?.source_weights;
    const torvikEfficiency = this.torvikEfficiencyComponent(snapshotA, snapshotB, venue);
    const torvikBarthag = this.torvikBarthagComponent(snapshotA, snapshotB, venue);
    const bpi = this.bpiComponent(snapshotA, snapshotB, venue);
    const net = this.netComponent(snapshotA, snapshotB, venue);
    const resumeForm = this.resumeFormComponent(snapshotA, snapshotB, venue);

    return [
      {
        key: "torvik_efficiency",
        name: "Torvik efficiency model",
        description:
          "Additive adjusted-offense versus adjusted-defense matchup translated to game margin via expected tempo.",
        family: "public_rating",
        availability: "observed",
        baseWeight:
          learnedWeights?.torvik_efficiency ?? this.latentConsensus.componentWeights.torvik_efficiency,
        reliability: availabilityReliability("observed"),
        ...torvikEfficiency,
      },
      {
        key: "torvik_barthag",
        name: "Torvik Barthag Log5",
        description:
          "Win-probability expert built from Barthag using Log5, then transformed back into implied margin space.",
        family: "public_rating",
        availability: "observed",
        baseWeight:
          learnedWeights?.torvik_barthag ?? this.latentConsensus.componentWeights.torvik_barthag,
        reliability: availabilityReliability("observed"),
        ...torvikBarthag,
      },
      {
        key: "bpi",
        name: "ESPN BPI",
        description:
          bpi.availability === "observed"
            ? "Observed BPI differential translated directly into projected spread."
            : "BPI was missing for at least one team, so this source is imputed from efficiency and conference context.",
        family: "public_rating",
        baseWeight: learnedWeights?.bpi ?? this.latentConsensus.componentWeights.bpi,
        reliability: availabilityReliability(bpi.availability),
        ...bpi,
      },
      {
        key: "net",
        name: "NET rating curve",
        description:
          net.availability === "observed"
            ? "NET ranks are mapped through a monotonic power curve to estimate neutral-floor strength."
            : "NET was missing for at least one team, so this source is estimated from efficiency and resume signals.",
        family: "public_rank",
        baseWeight: learnedWeights?.net ?? this.latentConsensus.componentWeights.net,
        reliability: availabilityReliability(net.availability),
        ...net,
      },
      {
        key: "resume_form",
        name: "Resume and recent form",
        description:
          "Context prior built from resume score, recent form, conference strength, continuity, and availability proxies.",
        family: "contextual_prior",
        availability: "observed",
        baseWeight:
          learnedWeights?.resume_form ?? this.latentConsensus.componentWeights.resume_form,
        reliability: availabilityReliability("observed"),
        ...resumeForm,
      },
    ];
  }

  private simulateRegion(
    region: BracketRegion,
    field: BracketEntry[],
    counters: Map<
      string,
      {
        roundOf32: number;
        sweet16: number;
        elite8: number;
        finalFour: number;
        championship: number;
        champion: number;
      }
    >,
    rng: () => number,
  ) {
    const regionEntries = field
      .filter((entry) => entry.region === region)
      .sort((left, right) => left.seed - right.seed);

    const roundOf64Winners = ROUND_ONE_PAIRINGS.map(([seedA, seedB]) => {
      const teamAId = regionEntries.find((entry) => entry.seed === seedA)!.teamId;
      const teamBId = regionEntries.find((entry) => entry.seed === seedB)!.teamId;
      const winner = this.simulateNeutralGame(teamAId, teamBId, rng, 1);
      counters.get(winner)!.roundOf32 += 1;
      return winner;
    });

    const sweet16Winners = [
      this.simulateNeutralGame(roundOf64Winners[0], roundOf64Winners[1], rng, 2),
      this.simulateNeutralGame(roundOf64Winners[2], roundOf64Winners[3], rng, 2),
      this.simulateNeutralGame(roundOf64Winners[4], roundOf64Winners[5], rng, 2),
      this.simulateNeutralGame(roundOf64Winners[6], roundOf64Winners[7], rng, 2),
    ];
    sweet16Winners.forEach((winner) => counters.get(winner)!.sweet16 += 1);

    const elite8Winners = [
      this.simulateNeutralGame(sweet16Winners[0], sweet16Winners[1], rng, 3),
      this.simulateNeutralGame(sweet16Winners[2], sweet16Winners[3], rng, 3),
    ];
    elite8Winners.forEach((winner) => counters.get(winner)!.elite8 += 1);

    const regionChampion = this.simulateNeutralGame(elite8Winners[0], elite8Winners[1], rng, 4);
    counters.get(regionChampion)!.finalFour += 1;
    return regionChampion;
  }

  private simulateNeutralGame(teamAId: string, teamBId: string, rng: () => number, tournamentRound = 0) {
    const result = this.predictMatchup(
      teamAId,
      teamBId,
      this.modelRun.generatedAt.slice(0, 10),
      "neutral",
      this.modelRun.season,
      tournamentRound,
    );
    return rng() <= result.prediction.winProbabilityA ? teamAId : teamBId;
  }

  private buildFeatureAttributions(
    teamA: Team,
    teamB: Team,
    snapshotA: TeamRatingSnapshot,
    snapshotB: TeamRatingSnapshot,
    venue: Venue,
    components: PredictionComponent[],
    disagreementIndex: number,
    observedSourceCount: number,
  ): FeatureAttribution[] {
    const dominantSource = [...components].sort(
      (left, right) => Math.abs(right.expectedMargin) - Math.abs(left.expectedMargin),
    )[0]!;
    const favoriteId = dominantSource.expectedMargin >= 0 ? teamA.id : teamB.id;
    const favoriteName = favoriteId === teamA.id ? teamA.shortName : teamB.shortName;

    const rawFactors: FeatureAttribution[] = [
      this.featureAttribution(
        "Crowd consensus",
        dominantSource.expectedMargin * dominantSource.weight,
        teamA,
        teamB,
        `${favoriteName} gets the strongest push from ${dominantSource.name.toLowerCase()}.`,
      ),
      this.featureAttribution(
        "Offense vs defense",
        (snapshotA.offense - snapshotB.offense + snapshotB.defense - snapshotA.defense) * 0.18,
        teamA,
        teamB,
        `${snapshotA.offense - snapshotA.defense > snapshotB.offense - snapshotB.defense ? teamA.shortName : teamB.shortName} owns the cleaner adjusted offense-defense profile entering the game.`,
      ),
      this.featureAttribution(
        "Resume and recency",
        ((snapshotA.resumeScore + snapshotA.recentForm) - (snapshotB.resumeScore + snapshotB.recentForm)) *
          0.08,
        teamA,
        teamB,
        `${snapshotA.resumeScore + snapshotA.recentForm > snapshotB.resumeScore + snapshotB.recentForm ? teamA.shortName : teamB.shortName} has the stronger resume plus recent-form blend.`,
      ),
      this.featureAttribution(
        "Signal agreement",
        (3.4 - disagreementIndex) * (favoriteId === teamA.id ? 1 : -1),
        teamA,
        teamB,
        disagreementIndex < 3
          ? `Public ratings are tightly aligned on ${favoriteName}.`
          : "Public ratings are split enough that the favorite is less secure than the headline spread suggests.",
      ),
      this.featureAttribution(
        "Observed source coverage",
        (observedSourceCount - (components.length - observedSourceCount)) *
          (favoriteId === teamA.id ? 0.8 : -0.8),
        teamA,
        teamB,
        observedSourceCount === components.length
          ? "All public source signals are observed directly for this matchup."
          : "At least one public rating is estimated rather than observed, which slightly lowers certainty.",
      ),
      this.featureAttribution(
        "Venue",
        homeCourtAdjustment(venue),
        teamA,
        teamB,
        venue === "neutral"
          ? "Neutral floor keeps home-court value out of the consensus model."
          : `${venue === "home_a" ? teamA.shortName : teamB.shortName} receives the home-court adjustment.`,
      ),
    ];

    return rawFactors
      .sort((left, right) => Math.abs(right.impact) - Math.abs(left.impact))
      .slice(0, 5);
  }

  private featureAttribution(
    label: string,
    impact: number,
    teamA: Team,
    teamB: Team,
    description: string,
  ): FeatureAttribution {
    return {
      label,
      impact,
      favoredTeamId: impact >= 0 ? teamA.id : teamB.id,
      description,
    };
  }

  private torvikEfficiencyComponent(
    snapshotA: TeamRatingSnapshot,
    snapshotB: TeamRatingSnapshot,
    venue: Venue,
  ) {
    const expectedOffenseA = snapshotA.offense + (snapshotB.defense - D1_AVG_EFFICIENCY);
    const expectedOffenseB = snapshotB.offense + (snapshotA.defense - D1_AVG_EFFICIENCY);
    const possessions = (snapshotA.tempo + snapshotB.tempo) / 2;
    const expectedMargin =
      ((expectedOffenseA - expectedOffenseB) / 100) * possessions +
      homeCourtAdjustment(venue);
    return {
      expectedMargin,
      winProbabilityA: clamp(1 - normalCdf(0, expectedMargin, SIGMA), 0.01, 0.99),
    };
  }

  private torvikBarthagComponent(
    snapshotA: TeamRatingSnapshot,
    snapshotB: TeamRatingSnapshot,
    venue: Venue,
  ) {
    const pA = snapshotA.barthag;
    const pB = snapshotB.barthag;
    const log5Probability = clamp(
      (pA - pA * pB) / (pA + pB - 2 * pA * pB),
      0.01,
      0.99,
    );
    const expectedMargin = probabilityToMargin(log5Probability) + homeCourtAdjustment(venue);
    return {
      expectedMargin,
      winProbabilityA: clamp(1 - normalCdf(0, expectedMargin, SIGMA), 0.01, 0.99),
    };
  }

  private bpiComponent(
    snapshotA: TeamRatingSnapshot,
    snapshotB: TeamRatingSnapshot,
    venue: Venue,
  ) {
    const availability: PredictionComponentAvailability =
      snapshotA.bpi != null && snapshotB.bpi != null ? "observed" : "estimated";
    const ratingA =
      snapshotA.bpi ??
      snapshotA.efficiencyMargin * 0.74 +
        snapshotA.conferenceStrength * 0.035 +
        (snapshotA.recentForm - 50) * 0.06;
    const ratingB =
      snapshotB.bpi ??
      snapshotB.efficiencyMargin * 0.74 +
        snapshotB.conferenceStrength * 0.035 +
        (snapshotB.recentForm - 50) * 0.06;
    const expectedMargin = ratingA - ratingB + homeCourtAdjustment(venue);

    return {
      availability,
      expectedMargin,
      winProbabilityA: clamp(1 - normalCdf(0, expectedMargin, SIGMA), 0.01, 0.99),
    };
  }

  private netComponent(
    snapshotA: TeamRatingSnapshot,
    snapshotB: TeamRatingSnapshot,
    venue: Venue,
  ) {
    const availability: PredictionComponentAvailability =
      snapshotA.netRank != null && snapshotB.netRank != null ? "observed" : "estimated";
    const fallbackA =
      snapshotA.efficiencyMargin * 0.58 +
      snapshotA.resumeScore * 0.14 +
      snapshotA.conferenceStrength * 0.04;
    const fallbackB =
      snapshotB.efficiencyMargin * 0.58 +
      snapshotB.resumeScore * 0.14 +
      snapshotB.conferenceStrength * 0.04;
    const ratingA = rankToPower(snapshotA.netRank, fallbackA);
    const ratingB = rankToPower(snapshotB.netRank, fallbackB);
    const expectedMargin =
      (ratingA - ratingB) * 0.95 + homeCourtAdjustment(venue) * 0.55;

    return {
      availability,
      expectedMargin,
      winProbabilityA: clamp(1 - normalCdf(0, expectedMargin, SIGMA), 0.01, 0.99),
    };
  }

  private resumeFormComponent(
    snapshotA: TeamRatingSnapshot,
    snapshotB: TeamRatingSnapshot,
    venue: Venue,
  ) {
    const signalA =
      snapshotA.resumeScore * 0.41 +
      snapshotA.recentForm * 0.26 +
      snapshotA.conferenceStrength * 0.14 +
      snapshotA.rosterContinuity * 100 * 0.09 +
      snapshotA.availabilityScore * 100 * 0.1;
    const signalB =
      snapshotB.resumeScore * 0.41 +
      snapshotB.recentForm * 0.26 +
      snapshotB.conferenceStrength * 0.14 +
      snapshotB.rosterContinuity * 100 * 0.09 +
      snapshotB.availabilityScore * 100 * 0.1;
    const expectedMargin =
      (signalA - signalB) * 0.16 + homeCourtAdjustment(venue) * 0.4;

    return {
      expectedMargin,
      winProbabilityA: clamp(1 - normalCdf(0, expectedMargin, SIGMA), 0.01, 0.99),
    };
  }

  private learnedExpectedMargin(
    components: PredictionComponent[],
    sourceConsensusMargin: number,
    latentConsensusMargin: number,
    seedDiff: number,
    sameConference: number,
    snapshotA?: TeamRatingSnapshot,
    snapshotB?: TeamRatingSnapshot,
    teamA?: Team,
    teamB?: Team,
    tournamentRound = 0,
  ) {
    if (!this.learnedArtifact) {
      return sourceConsensusMargin * 0.72 + latentConsensusMargin * 0.28;
    }

    const featureMap = this.featureMapFromComponents(
      components,
      sourceConsensusMargin,
      latentConsensusMargin,
      seedDiff,
      sameConference,
      snapshotA,
      snapshotB,
      teamA,
      teamB,
      tournamentRound,
    );

    const scaledFeature = (key: string, value: number) => {
      const mean = this.learnedArtifact!.scaler.means[key] ?? 0;
      const scale = this.learnedArtifact!.scaler.scales[key] ?? 1;
      return (value - mean) / scale;
    };

    return Object.entries(this.learnedArtifact.margin_model.coefficients).reduce(
      (sum, [key, coefficient]) => sum + scaledFeature(key, featureMap[key] ?? 0) * coefficient,
      this.learnedArtifact.margin_model.intercept,
    );
  }

  private calibratedProbability(
    rawProbability: number,
    disagreementIndex: number,
    observedSourceCount: number,
    sourceCount: number,
    components: PredictionComponent[],
    seedDiff: number,
    sameConference: number,
    snapshotA?: TeamRatingSnapshot,
    snapshotB?: TeamRatingSnapshot,
    teamA?: Team,
    teamB?: Team,
    tournamentRound = 0,
  ) {
    const learnedProbability = this.learnedProbability(
      components,
      rawProbability,
      seedDiff,
      sameConference,
      snapshotA,
      snapshotB,
      teamA,
      teamB,
      tournamentRound,
    );
    const generalAnchors = this.learnedArtifact?.calibration.anchors ?? CALIBRATION_ANCHORS;
    const seedGapAnchors = this.learnedArtifact?.calibration.seedGapAnchors;
    const extremeGapAnchors = this.learnedArtifact?.calibration.extremeGapAnchors;
    const generalCalibrated = interpolateCalibration(learnedProbability, generalAnchors);

    let isotonic = generalCalibrated;
    const absSeedDiff = Math.abs(seedDiff);
    // Tier-1 (gap >= 5): blend from general → seed-gap-specific as gap grows 5→8
    if (seedGapAnchors && seedGapAnchors.length > 0 && absSeedDiff >= 5) {
      const seedGapCalibrated = interpolateCalibration(learnedProbability, seedGapAnchors);
      const tier1Weight = clamp((absSeedDiff - 5) / 3, 0, 1);
      isotonic = generalCalibrated * (1 - tier1Weight) + seedGapCalibrated * tier1Weight;
    }
    // Tier-2 (gap >= 8): blend from tier-1 result → extreme-gap-specific as gap grows 8→12
    if (extremeGapAnchors && extremeGapAnchors.length > 0 && absSeedDiff >= 8) {
      const extremeCalibrated = interpolateCalibration(learnedProbability, extremeGapAnchors);
      const tier2Weight = clamp((absSeedDiff - 8) / 4, 0, 1);
      isotonic = isotonic * (1 - tier2Weight) + extremeCalibrated * tier2Weight;
    }

    const observedShare = observedSourceCount / sourceCount;
    const shrink = clamp(0.97 - disagreementIndex * 0.005 - (1 - observedShare) * 0.04, 0.85, 0.97);
    return clamp(0.5 + (isotonic - 0.5) * shrink, 0.01, 0.99);
  }

  private learnedProbability(
    components: PredictionComponent[],
    rawProbability: number,
    seedDiff: number,
    sameConference: number,
    snapshotA?: TeamRatingSnapshot,
    snapshotB?: TeamRatingSnapshot,
    teamA?: Team,
    teamB?: Team,
    tournamentRound = 0,
  ) {
    if (!this.learnedArtifact) {
      return rawProbability;
    }

    const featureMap = this.featureMapFromComponents(
      components,
      components.reduce((sum, component) => sum + component.expectedMargin * component.weight, 0),
      0,
      seedDiff,
      sameConference,
      snapshotA,
      snapshotB,
      teamA,
      teamB,
      tournamentRound,
    );
    const scaledFeature = (key: string, value: number) => {
      const mean = this.learnedArtifact!.scaler.means[key] ?? 0;
      const scale = this.learnedArtifact!.scaler.scales[key] ?? 1;
      return (value - mean) / scale;
    };

    const logit = Object.entries(this.learnedArtifact.win_model.coefficients).reduce(
      (sum, [key, coefficient]) => sum + scaledFeature(key, featureMap[key] ?? 0) * coefficient,
      this.learnedArtifact.win_model.intercept,
    );
    return clamp(1 / (1 + Math.exp(-logit)), 0.01, 0.99);
  }

  private featureMapFromComponents(
    components: PredictionComponent[],
    sourceConsensusMargin: number,
    latentConsensusMargin: number,
    seedDiff: number,
    sameConference: number,
    snapshotA?: TeamRatingSnapshot,
    snapshotB?: TeamRatingSnapshot,
    teamA?: Team,
    teamB?: Team,
    tournamentRound = 0,
  ) {
    const featureMap: Record<string, number> = {
      source_consensus_margin: sourceConsensusMargin,
      latent_consensus_margin: latentConsensusMargin,
      seed_diff: seedDiff,
      same_conference: sameConference,
      observed_source_count: components.filter((component) => component.availability === "observed").length,
    };

    components.forEach((component) => {
      featureMap[component.key] = component.expectedMargin;
      featureMap[`${component.key}_win_prob`] = component.winProbabilityA;
      featureMap[`${component.key}_observed`] = component.availability === "observed" ? 1 : 0;
    });

    if (snapshotA && snapshotB && teamA && teamB) {
      // Fixed field size = 362 (typical D1 count); denominator = 361 to match Python training pipeline
      const getPct = (rank: number | null) => (rank == null ? null : 1 - (rank - 1) / 361);

      const aBpi = getPct(snapshotA.bpiRank);
      const bBpi = getPct(snapshotB.bpiRank);
      const missingBpi = aBpi == null || bBpi == null ? 1 : 0;
      featureMap["missing_bpi_rank"] = missingBpi;
      featureMap["bpi_rank_percentile_diff"] = missingBpi ? 0 : aBpi! - bBpi!;

      const aNet = getPct(snapshotA.netRank);
      const bNet = getPct(snapshotB.netRank);
      const missingNet = aNet == null || bNet == null ? 1 : 0;
      featureMap["missing_net_rank"] = missingNet;
      featureMap["net_rank_percentile_diff"] = missingNet ? 0 : aNet! - bNet!;

      const aMassey = getPct(teamA.torvikRank);
      const bMassey = getPct(teamB.torvikRank);
      const missingMassey = 0;
      featureMap["missing_massey_ordinal_rank"] = missingMassey;
      featureMap["massey_ordinal_rank_percentile_diff"] = aMassey! - bMassey!;

      const kenpomDiff = snapshotA.efficiencyMargin - snapshotB.efficiencyMargin;
      featureMap["missing_kenpom_badj_em"] = 0;
      featureMap["kenpom_badj_em_power_diff"] = kenpomDiff;

      const missingEvanMiya =
        snapshotA.evanmiyaRelativeRating == null || snapshotB.evanmiyaRelativeRating == null ? 1 : 0;
      featureMap["missing_evanmiya_relative_rating"] = missingEvanMiya;
      featureMap["evanmiya_relative_rating_power_diff"] =
        missingEvanMiya === 1 ? 0 : snapshotA.evanmiyaRelativeRating! - snapshotB.evanmiyaRelativeRating!;

      const missingFiveThirtyEight =
        snapshotA.fivethirtyeightPower == null || snapshotB.fivethirtyeightPower == null ? 1 : 0;
      featureMap["missing_fivethirtyeight_power"] = missingFiveThirtyEight;
      featureMap["fivethirtyeight_power_power_diff"] =
        missingFiveThirtyEight === 1 ? 0 : snapshotA.fivethirtyeightPower! - snapshotB.fivethirtyeightPower!;

      featureMap["seed_kenpom_interaction"] = seedDiff * kenpomDiff;
      featureMap["abs_seed_same_conference_interaction"] = Math.abs(seedDiff) * sameConference;

      // Round-aware features (use round ordinal if in tournament context, else 1)
      const roundOrdinal = tournamentRound > 0 ? tournamentRound : 1;
      featureMap["round_seed_interaction"] = roundOrdinal * Math.abs(seedDiff);
      featureMap["round_kenpom_interaction"] = roundOrdinal * kenpomDiff;

      // Seed nonlinearity features
      featureMap["seed_diff_squared"] = seedDiff * seedDiff;
      featureMap["log_abs_seed_diff"] = Math.log1p(Math.abs(seedDiff));

      // Rating disagreement features
      const availableDiffs: number[] = [];
      if (!missingBpi) availableDiffs.push(featureMap["bpi_rank_percentile_diff"]);
      if (!missingNet) availableDiffs.push(featureMap["net_rank_percentile_diff"]);
      if (!missingMassey) availableDiffs.push(featureMap["massey_ordinal_rank_percentile_diff"]);
      if (featureMap["missing_kenpom_badj_em"] === 0) availableDiffs.push(kenpomDiff);
      if (missingEvanMiya === 0) availableDiffs.push(featureMap["evanmiya_relative_rating_power_diff"]);
      if (availableDiffs.length > 1) {
        const diffMean = availableDiffs.reduce((s, v) => s + v, 0) / availableDiffs.length;
        const diffVar = availableDiffs.reduce((s, v) => s + (v - diffMean) ** 2, 0) / availableDiffs.length;
        featureMap["rating_disagreement"] = Math.sqrt(diffVar);
        featureMap["max_min_rating_spread"] = Math.max(...availableDiffs) - Math.min(...availableDiffs);
      } else {
        featureMap["rating_disagreement"] = 0;
        featureMap["max_min_rating_spread"] = 0;
      }

      featureMap["missing_resume_rank_blend"] = 0;
      // NOTE: semantic divergence from Python training pipeline.
      // Python computes this as the average of 7 resume rank percentile diffs.
      // TS uses (resumeScore diff) / 100. The scaler normalizes both, but the
      // optimized feature set excludes this field, so the mismatch does not affect promotion.
      featureMap["resume_rank_blend_percentile_diff"] = (snapshotA.resumeScore - snapshotB.resumeScore) / 100;

      featureMap["available_predictive_rank_count"] = 3 - (missingBpi + missingNet + missingMassey);
      featureMap["available_power_feature_count"] =
        3 - (featureMap["missing_kenpom_badj_em"] + missingEvanMiya + missingFiveThirtyEight);
      featureMap["available_resume_feature_count"] = 1;
    }

    return featureMap;
  }

  private confidenceTier(
    expectedMargin: number,
    disagreementIndex: number,
    winProbabilityA: number,
    observedShare: number,
  ): ConfidenceTier {
    const maxProbability = Math.max(winProbabilityA, 1 - winProbabilityA);

    if (
      Math.abs(expectedMargin) >= 8 &&
      disagreementIndex <= 2.75 &&
      maxProbability >= 0.76 &&
      observedShare >= 0.8
    ) {
      return "high";
    }

    if (
      Math.abs(expectedMargin) >= 4 &&
      disagreementIndex <= 5.1 &&
      maxProbability >= 0.63 &&
      observedShare >= 0.6
    ) {
      return "medium";
    }

    return "low";
  }

  private requireTeam(teamId: string) {
    const team = this.getTeam(teamId);
    if (!team) {
      throw new Error(`Unknown team id: ${teamId}`);
    }
    return team;
  }

  private requireSnapshot(teamId: string) {
    const snapshot = this.getSnapshot(teamId);
    if (!snapshot) {
      throw new Error(`Missing snapshot for team id: ${teamId}`);
    }
    return snapshot;
  }
}
