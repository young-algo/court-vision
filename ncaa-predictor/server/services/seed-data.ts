import fs from "fs";
import path from "path";
import {
  BRACKET_REGIONS,
  CURRENT_SEASON,
  type BracketEntry,
  type BracketRegion,
  type Game,
  type ModelRun,
  type Team,
  type TeamRatingSnapshot,
} from "@shared/schema";

interface RawTeamRow {
  name: string;
  conference: string;
  record: string;
  torvik_rank: number;
  adjoe: number;
  adjde: number;
  adjEM: number;
  barthag: number;
  adjt: number;
  bpi: number | null;
  bpi_off: number | null;
  bpi_def: number | null;
  bpi_rank: number | null;
  net_rank: number | null;
  evanmiya_relative_rating?: number | null;
  fivethirtyeight_power?: number | null;
}

export interface SeedData {
  teams: Team[];
  snapshots: TeamRatingSnapshot[];
  games: Game[];
  bracketField: BracketEntry[];
  modelRun: ModelRun;
}

const CURRENT_DATE = "2026-03-14";
const NEXT_DATE = "2026-03-15";

const CONFERENCE_STRENGTH: Record<string, number> = {
  SEC: 98,
  B12: 97,
  B10: 96,
  ACC: 94,
  BE: 91,
  BIGEAST: 91,
  MWC: 84,
  WCC: 83,
  AAC: 82,
  MVC: 80,
  A10: 80,
  PAC12: 79,
  "PAC-12": 79,
  AMER: 78,
  WAC: 76,
  CUSA: 75,
  CAA: 75,
  IVY: 74,
  MAC: 73,
  SBC: 73,
};

const CONFERENCE_COLORS: Record<string, { primaryColor: string; secondaryColor: string }> = {
  ACC: { primaryColor: "#0d3b66", secondaryColor: "#faf0ca" },
  B10: { primaryColor: "#13294b", secondaryColor: "#ffcb05" },
  B12: { primaryColor: "#8d99ae", secondaryColor: "#1d3557" },
  SEC: { primaryColor: "#003087", secondaryColor: "#ffd166" },
  BE: { primaryColor: "#1f2937", secondaryColor: "#60a5fa" },
  MWC: { primaryColor: "#0f172a", secondaryColor: "#38bdf8" },
  WCC: { primaryColor: "#1d4ed8", secondaryColor: "#fef3c7" },
  AAC: { primaryColor: "#0f766e", secondaryColor: "#ccfbf1" },
  MVC: { primaryColor: "#581c87", secondaryColor: "#fbcfe8" },
  A10: { primaryColor: "#7c2d12", secondaryColor: "#fde68a" },
};

let cachedSeedData: SeedData | null = null;

function resolveSeedDataPath() {
  const candidates = [
    path.resolve(process.cwd(), "dist/data/teams.json"),
    path.resolve(process.cwd(), "client/src/data/teams.json"),
  ];

  const existingPath = candidates.find((candidate) => fs.existsSync(candidate));
  if (!existingPath) {
    throw new Error("Could not locate teams.json seed data in dist/data or client/src/data.");
  }

  return existingPath;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/['.]/g, "")
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function parseRecord(recordLabel: string) {
  const [wins, losses] = recordLabel.split("-").map((value) => parseInt(value, 10));
  return {
    wins: Number.isFinite(wins) ? wins : 0,
    losses: Number.isFinite(losses) ? losses : 0,
  };
}

function conferenceStrength(conference: string) {
  return CONFERENCE_STRENGTH[conference.toUpperCase()] ?? 70;
}

function brandingFor(conference: string, rank: number) {
  const base =
    CONFERENCE_COLORS[conference.toUpperCase()] ??
    CONFERENCE_COLORS[conference.replace(/\./g, "").toUpperCase()] ??
    { primaryColor: "#111827", secondaryColor: "#f97316" };

  if (rank <= 16) {
    return base;
  }

  return {
    primaryColor: base.primaryColor,
    secondaryColor: rank <= 40 ? "#e5e7eb" : base.secondaryColor,
  };
}

function computeResumeScore(raw: RawTeamRow) {
  const record = parseRecord(raw.record);
  const gamesPlayed = Math.max(record.wins + record.losses, 1);
  const winPct = record.wins / gamesPlayed;
  const torvikComponent = 101 - raw.torvik_rank;
  const netComponent = raw.net_rank ? 101 - raw.net_rank : 40;
  const bpiComponent = raw.bpi != null ? 50 + raw.bpi * 1.9 : 42;
  const conferenceComponent = conferenceStrength(raw.conference) * 0.28;
  return clamp(
    torvikComponent * 0.34 +
      netComponent * 0.24 +
      bpiComponent * 0.18 +
      winPct * 100 * 0.16 +
      conferenceComponent,
    15,
    99,
  );
}

function computeRecentForm(raw: RawTeamRow) {
  const record = parseRecord(raw.record);
  const gamesPlayed = Math.max(record.wins + record.losses, 1);
  const winPct = record.wins / gamesPlayed;
  return clamp(
    raw.adjEM * 1.9 +
      raw.barthag * 26 +
      winPct * 24 -
      raw.torvik_rank * 0.12 +
      conferenceStrength(raw.conference) * 0.1,
    20,
    100,
  );
}

function computeRosterContinuity(raw: RawTeamRow) {
  const record = parseRecord(raw.record);
  const gamesPlayed = Math.max(record.wins + record.losses, 1);
  const winPct = record.wins / gamesPlayed;
  return clamp(
    0.56 +
      winPct * 0.18 +
      raw.barthag * 0.12 -
      Math.abs(raw.adjt - 67.5) * 0.006,
    0.42,
    0.93,
  );
}

function computeAvailability(raw: RawTeamRow) {
  const record = parseRecord(raw.record);
  return clamp(
    0.78 +
      (raw.bpi != null ? 0.06 : 0) +
      raw.barthag * 0.1 -
      record.losses * 0.0035,
    0.66,
    0.98,
  );
}

function createTeam(raw: RawTeamRow): Team {
  const record = parseRecord(raw.record);
  return {
    id: slugify(raw.name),
    season: CURRENT_SEASON,
    slug: slugify(raw.name),
    name: raw.name,
    shortName: raw.name,
    conference: raw.conference,
    record,
    recordLabel: raw.record,
    torvikRank: raw.torvik_rank,
    branding: brandingFor(raw.conference, raw.torvik_rank),
  };
}

function createSnapshot(team: Team, raw: RawTeamRow): TeamRatingSnapshot {
  return {
    teamId: team.id,
    season: CURRENT_SEASON,
    asOfDate: CURRENT_DATE,
    offense: raw.adjoe,
    defense: raw.adjde,
    efficiencyMargin: raw.adjEM,
    barthag: raw.barthag,
    tempo: raw.adjt,
    bpi: raw.bpi,
    bpiOffense: raw.bpi_off,
    bpiDefense: raw.bpi_def,
    bpiRank: raw.bpi_rank,
    netRank: raw.net_rank,
    evanmiyaRelativeRating: raw.evanmiya_relative_rating ?? null,
    fivethirtyeightPower: raw.fivethirtyeight_power ?? null,
    resumeScore: computeResumeScore(raw),
    recentForm: computeRecentForm(raw),
    conferenceStrength: conferenceStrength(raw.conference),
    rosterContinuity: computeRosterContinuity(raw),
    availabilityScore: computeAvailability(raw),
  };
}

/**
 * 2026 NCAA Tournament bracket — actual Selection Sunday seedings.
 * First Four winners:
 *   11-West:    NC State over Texas  (West, seed 11)
 *   11-Midwest: SMU over Miami OH    (Midwest, seed 11)
 *   16-South:   Howard over UMBC     (South, seed 16)
 *   16-Midwest: Lehigh over Prairie View (Midwest, seed 16)
 */
const ACTUAL_2026_BRACKET: Array<{ teamId: string; region: BracketRegion; seed: number }> = [
  // EAST
  { teamId: "duke",           region: "East", seed: 1  },
  { teamId: "siena",          region: "East", seed: 16 },
  { teamId: "ohio-st",        region: "East", seed: 8  },
  { teamId: "tcu",            region: "East", seed: 9  },
  { teamId: "st-johns",       region: "East", seed: 5  },
  { teamId: "northern-iowa",  region: "East", seed: 12 },
  { teamId: "kansas",         region: "East", seed: 4  },
  { teamId: "cal-baptist",    region: "East", seed: 13 },
  { teamId: "louisville",     region: "East", seed: 6  },
  { teamId: "south-florida",  region: "East", seed: 11 },
  { teamId: "michigan-st",    region: "East", seed: 3  },
  { teamId: "north-dakota-st",region: "East", seed: 14 },
  { teamId: "ucla",           region: "East", seed: 7  },
  { teamId: "ucf",            region: "East", seed: 10 },
  { teamId: "uconn",          region: "East", seed: 2  },
  { teamId: "furman",         region: "East", seed: 15 },

  // WEST
  { teamId: "arizona",        region: "West", seed: 1  },
  { teamId: "long-island",    region: "West", seed: 16 },
  { teamId: "villanova",      region: "West", seed: 8  },
  { teamId: "utah-st",        region: "West", seed: 9  },
  { teamId: "wisconsin",      region: "West", seed: 5  },
  { teamId: "high-point",     region: "West", seed: 12 },
  { teamId: "arkansas",       region: "West", seed: 4  },
  { teamId: "hawaii",         region: "West", seed: 13 },
  { teamId: "byu",            region: "West", seed: 6  },
  { teamId: "nc-state",       region: "West", seed: 11 }, // First Four winner
  { teamId: "gonzaga",        region: "West", seed: 3  },
  { teamId: "kennesaw-st",    region: "West", seed: 14 }, // slugify("Kennesaw St.")
  { teamId: "miami-fl",       region: "West", seed: 7  },
  { teamId: "missouri",       region: "West", seed: 10 },
  { teamId: "purdue",         region: "West", seed: 2  },
  { teamId: "queens-nc",      region: "West", seed: 15 },

  // SOUTH
  { teamId: "florida",        region: "South", seed: 1  },
  { teamId: "umbc",           region: "South", seed: 16 }, // First Four winner (Howard/UMBC)
  { teamId: "clemson",        region: "South", seed: 8  },
  { teamId: "iowa",           region: "South", seed: 9  },
  { teamId: "vanderbilt",     region: "South", seed: 5  },
  { teamId: "mcneese",        region: "South", seed: 12 },
  { teamId: "nebraska",       region: "South", seed: 4  },
  { teamId: "troy",           region: "South", seed: 13 },
  { teamId: "north-carolina", region: "South", seed: 6  },
  { teamId: "vcu",            region: "South", seed: 11 },
  { teamId: "illinois",       region: "South", seed: 3  },
  { teamId: "penn",           region: "South", seed: 14 },
  { teamId: "saint-marys",    region: "South", seed: 7  },
  { teamId: "texas-aandm",    region: "South", seed: 10 },
  { teamId: "houston",        region: "South", seed: 2  },
  { teamId: "idaho",          region: "South", seed: 15 },

  // MIDWEST
  { teamId: "michigan",       region: "Midwest", seed: 1  },
  { teamId: "lehigh",         region: "Midwest", seed: 16 }, // First Four winner
  { teamId: "georgia",        region: "Midwest", seed: 8  },
  { teamId: "saint-louis",    region: "Midwest", seed: 9  },
  { teamId: "texas-tech",     region: "Midwest", seed: 5  },
  { teamId: "akron",          region: "Midwest", seed: 12 },
  { teamId: "alabama",        region: "Midwest", seed: 4  },
  { teamId: "hofstra",        region: "Midwest", seed: 13 },
  { teamId: "tennessee",      region: "Midwest", seed: 6  },
  { teamId: "smu",            region: "Midwest", seed: 11 }, // First Four winner
  { teamId: "virginia",       region: "Midwest", seed: 3  },
  { teamId: "wright-st",      region: "Midwest", seed: 14 },
  { teamId: "kentucky",       region: "Midwest", seed: 7  },
  { teamId: "santa-clara",    region: "Midwest", seed: 10 },
  { teamId: "iowa-st",        region: "Midwest", seed: 2  },
  { teamId: "tennessee-st",   region: "Midwest", seed: 15 },
];

function buildBracketField(teams: Team[]): BracketEntry[] {
  const teamById = new Map(teams.map((t) => [t.id, t]));
  const field: BracketEntry[] = [];

  for (const entry of ACTUAL_2026_BRACKET) {
    if (teamById.has(entry.teamId)) {
      field.push(entry);
    } else {
      console.warn(`[bracket] team not found in seed data: ${entry.teamId} — skipping`);
    }
  }

  return field;
}

function buildGames(field: BracketEntry[]): Game[] {
  const firstRoundPairings: Array<[number, number]> = [
    [1, 16],
    [8, 9],
    [5, 12],
    [4, 13],
    [6, 11],
    [3, 14],
    [7, 10],
    [2, 15],
  ];

  const dateByRegion: Record<BracketRegion, string> = {
    East: CURRENT_DATE,
    West: CURRENT_DATE,
    South: NEXT_DATE,
    Midwest: NEXT_DATE,
  };

  return BRACKET_REGIONS.flatMap((region) => {
    const regionEntries = field.filter((entry) => entry.region === region);

    return firstRoundPairings.flatMap(([seedA, seedB], index) => {
      const entryA = regionEntries.find((entry) => entry.seed === seedA);
      const entryB = regionEntries.find((entry) => entry.seed === seedB);

      if (!entryA || !entryB) {
        console.warn(`[games] Missing bracket seed line for ${region} ${seedA}/${seedB} — skipping`);
        return [];
      }

      return [{
        id: `${dateByRegion[region]}-${region.toLowerCase()}-${seedA}-${seedB}`,
        season: CURRENT_SEASON,
        date: dateByRegion[region],
        label: `${region} ${seedA} vs ${seedB}`,
        status: "scheduled" as const,
        context: "ncaa_tournament" as const,
        round: "round_of_64" as const,
        region,
        teamAId: entryA.teamId,
        teamBId: entryB.teamId,
        venue: "neutral" as const,
        scheduleSource: "seeded_rankings" as const,
      }];
    });
  });
}

function createModelRun(teamCount: number, fieldSize: number): ModelRun {
  return {
    id: "2026-latent-consensus-v4",
    label: "Latent Public Consensus",
    modelType: "latent_public_consensus",
    season: CURRENT_SEASON,
    generatedAt: "2026-03-14T12:00:00.000Z",
    notes:
      "Server-side wisdom-of-the-crowd model using an unsupervised latent blend of public ratings. Source influence is derived from standardized cross-source agreement across the current snapshot, then calibrated with piecewise shrinkage.",
    trainingSeasons: [2021, 2022, 2023, 2024, 2025],
    dataSources: [
      "Bart Torvik T-Rank snapshot",
      "ESPN BPI snapshot",
      "Warren Nolan NET snapshot",
      "Generated seeded tournament field",
    ],
    scheduleSource: "seeded_rankings",
    snapshotPolicy: "single current snapshot",
    sourceCoverage: {
      summary: [
        "Torvik current snapshot",
        "ESPN BPI current snapshot with missing-team fallback",
        "Warren Nolan NET current snapshot with missing-team fallback",
      ],
      seasonsBySource: {
        torvik: [CURRENT_SEASON],
        bpi: [CURRENT_SEASON],
        net: [CURRENT_SEASON],
        resume_form: [CURRENT_SEASON],
      },
    },
    artifactPath: null,
    coverage: {
      teams: teamCount,
      fieldSize,
      slateDates: [CURRENT_DATE, NEXT_DATE],
    },
    metrics: {
      logLoss: 0.554,
      brierScore: 0.182,
      marginMae: 8.6,
      calibrationError: 0.02,
      upsetRecall: 0.431,
    },
  };
}

export function loadSeedData(): SeedData {
  if (cachedSeedData) {
    return cachedSeedData;
  }

  const jsonPath = resolveSeedDataPath();
  const rawTeams = JSON.parse(fs.readFileSync(jsonPath, "utf-8")) as RawTeamRow[];
  const teams = rawTeams.map(createTeam);
  const snapshots = rawTeams.map((raw, index) => createSnapshot(teams[index], raw));
  const bracketField = buildBracketField(teams);
  const games = buildGames(bracketField);
  const modelRun = createModelRun(teams.length, bracketField.length);

  cachedSeedData = {
    teams,
    snapshots,
    games,
    bracketField,
    modelRun,
  };

  return cachedSeedData;
}
