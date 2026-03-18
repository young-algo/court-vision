/**
 * OddsService — wraps the-odds-api.com v4 endpoints.
 *
 * Two responsibilities:
 *  1. Live/upcoming odds: GET /v4/sports/basketball_ncaab/odds  (current games)
 *  2. Historical odds:    GET /v4/historical/sports/basketball_ncaab/odds  (pre-game snapshots)
 *
 * Results are in-memory cached with a configurable TTL to avoid burning quota on
 * every prediction request.
 *
 * Team name matching uses a multi-stage fuzzy approach:
 *   exact slug → city/school prefix → Dice-coefficient fuzzy fallback
 */

import { env } from "../env";

const ODDS_API_BASE = "https://api.the-odds-api.com/v4";
const SPORT = "basketball_ncaab";

// Cache TTL: 12 h for live odds, 24 h for historical snapshots
const LIVE_TTL_MS = 12 * 60 * 60 * 1000;
const HIST_TTL_MS = 24 * 60 * 60 * 1000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BookmakerOdds {
  bookmaker: string;
  moneylineHome: number | null;  // American odds
  moneylineAway: number | null;
  spreadHome: number | null;     // points (negative = home favored)
  spreadHomePrice: number | null;
}

export interface ConsensusOdds {
  /** Raw API event id */
  eventId: string;
  homeTeam: string;
  awayTeam: string;
  commenceTime: string;
  /** Vig-removed implied win probability for home team (0-1) */
  impliedProbHome: number | null;
  /** Consensus spread from home team perspective (negative = home favored) */
  consensusSpreadHome: number | null;
  /** American moneyline for home team (consensus best line) */
  bestMoneylineHome: number | null;
  /** American moneyline for away team */
  bestMoneylineAway: number | null;
  bookmakerCount: number;
  books: BookmakerOdds[];
  fetchedAt: string;
}

// Normalized pair keyed as `${slugA}::${slugB}` (sorted)
export type OddsMap = Map<string, ConsensusOdds>;

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

interface CacheEntry {
  data: ConsensusOdds[];
  expiresAt: number;
}

let liveCache: CacheEntry | null = null;
const histCache = new Map<string, CacheEntry>();

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/['.]/g, "")
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** Convert American odds to raw implied probability (before vig removal). */
function americanToProb(price: number): number {
  if (price >= 0) return 100 / (price + 100);
  return Math.abs(price) / (Math.abs(price) + 100);
}

/** Remove vig from two raw implied probs so they sum to 1. */
function removeVig(probA: number, probB: number): [number, number] {
  const total = probA + probB;
  if (total <= 0) return [0.5, 0.5];
  return [probA / total, probB / total];
}

/**
 * Dice-coefficient similarity between two strings (case insensitive).
 * Returns 0–1.
 */
function dice(a: string, b: string): number {
  const bigrams = (s: string) => {
    const set = new Set<string>();
    for (let i = 0; i < s.length - 1; i++) set.add(s.slice(i, i + 2));
    return set;
  };
  const aSet = bigrams(a.toLowerCase());
  const bSet = bigrams(b.toLowerCase());
  let intersection = 0;
  aSet.forEach((bg) => { if (bSet.has(bg)) intersection++; });
  const union = aSet.size + bSet.size;
  return union === 0 ? 0 : (2 * intersection) / union;
}

/**
 * Match an odds API team name to one of our team IDs.
 * teamMap: teamId → team name (full)
 */
export function matchTeam(
  apiName: string,
  teamMap: Map<string, string>,
): string | null {
  const apiSlug = slugify(apiName);
  const entries = Array.from(teamMap.entries());

  // 1. Exact slug match
  for (const [id, name] of entries) {
    if (slugify(name) === apiSlug) return id;
  }

  // 2. Our slug is contained in the API slug (handles "Duke Blue Devils" → "duke")
  for (const [id, name] of entries) {
    const ourSlug = slugify(name);
    if (apiSlug.startsWith(ourSlug) || apiSlug.includes(ourSlug)) return id;
  }

  // 3. API slug contained in our slug
  for (const [id, name] of entries) {
    const ourSlug = slugify(name);
    if (ourSlug.includes(apiSlug)) return id;
  }

  // 4. Fuzzy Dice similarity ≥ 0.62
  let bestScore = 0;
  let bestId: string | null = null;
  for (const [id, name] of entries) {
    const score = dice(apiName, name);
    if (score > bestScore) {
      bestScore = score;
      bestId = id;
    }
  }
  if (bestScore >= 0.62) return bestId;

  return null;
}

// ---------------------------------------------------------------------------
// API fetch helpers
// ---------------------------------------------------------------------------

interface RawOutcome {
  name: string;
  price: number;
  point?: number;
}

interface RawMarket {
  key: string;
  outcomes: RawOutcome[];
}

interface RawBookmaker {
  key: string;
  title: string;
  markets: RawMarket[];
}

interface RawEvent {
  id: string;
  commence_time: string;
  home_team: string;
  away_team: string;
  bookmakers: RawBookmaker[];
}

function parseEvent(event: RawEvent): ConsensusOdds {
  const home = event.home_team;
  const away = event.away_team;

  const books: BookmakerOdds[] = [];
  const moneylineHomeProbs: number[] = [];
  const spreadsHome: number[] = [];
  const bestMlHome: number[] = [];
  const bestMlAway: number[] = [];

  for (const bm of event.bookmakers ?? []) {
    let mlHome: number | null = null;
    let mlAway: number | null = null;
    let spreadHome: number | null = null;
    let spreadHomePrice: number | null = null;

    for (const market of bm.markets ?? []) {
      if (market.key === "h2h") {
        const homeO = market.outcomes.find((o) => o.name === home);
        const awayO = market.outcomes.find((o) => o.name === away);
        if (homeO) mlHome = homeO.price;
        if (awayO) mlAway = awayO.price;
      } else if (market.key === "spreads") {
        const homeO = market.outcomes.find((o) => o.name === home);
        if (homeO && homeO.point !== undefined) {
          spreadHome = homeO.point;
          spreadHomePrice = homeO.price;
        }
      }
    }

    if (mlHome !== null && mlAway !== null) {
      const [ph, pa] = removeVig(americanToProb(mlHome), americanToProb(mlAway));
      moneylineHomeProbs.push(ph);
      bestMlHome.push(mlHome);
      bestMlAway.push(mlAway);
    }
    if (spreadHome !== null) spreadsHome.push(spreadHome);

    books.push({
      bookmaker: bm.title,
      moneylineHome: mlHome,
      moneylineAway: mlAway,
      spreadHome,
      spreadHomePrice,
    });
  }

  const impliedProbHome =
    moneylineHomeProbs.length > 0
      ? moneylineHomeProbs.reduce((a, b) => a + b, 0) / moneylineHomeProbs.length
      : null;

  const consensusSpreadHome =
    spreadsHome.length > 0
      ? spreadsHome.reduce((a, b) => a + b, 0) / spreadsHome.length
      : null;

  // Best moneyline = median across books (more robust than mean)
  const median = (arr: number[]) => {
    if (arr.length === 0) return null;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
  };

  return {
    eventId: event.id,
    homeTeam: home,
    awayTeam: away,
    commenceTime: event.commence_time,
    impliedProbHome,
    consensusSpreadHome,
    bestMoneylineHome: median(bestMlHome),
    bestMoneylineAway: median(bestMlAway),
    bookmakerCount: event.bookmakers?.length ?? 0,
    books,
    fetchedAt: new Date().toISOString(),
  };
}

async function fetchFromApi(url: string, params: Record<string, string>): Promise<RawEvent[]> {
  const query = new URLSearchParams({ apiKey: env.ODDS_API_KEY, ...params });
  const fullUrl = `${url}?${query.toString()}`;
  const resp = await fetch(fullUrl);

  const remaining = resp.headers.get("x-requests-remaining");
  const used = resp.headers.get("x-requests-used");
  console.log(`[odds-api] ${resp.status} remaining=${remaining ?? "?"} used=${used ?? "?"}`);

  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`Odds API error ${resp.status}: ${body}`);
  }

  const payload = (await resp.json()) as RawEvent[] | { data: RawEvent[] };
  // Historical endpoint wraps in { data: [...], timestamp, ... }
  return Array.isArray(payload) ? payload : (payload as { data: RawEvent[] }).data ?? [];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Fetch live/upcoming NCAAB odds, with 5-minute cache. */
export async function getLiveOdds(): Promise<ConsensusOdds[]> {
  if (liveCache && Date.now() < liveCache.expiresAt) {
    return liveCache.data;
  }

  const events = await fetchFromApi(`${ODDS_API_BASE}/sports/${SPORT}/odds`, {
    regions: "us",
    markets: "h2h,spreads",
    oddsFormat: "american",
  });

  const data = events.map(parseEvent);
  liveCache = { data, expiresAt: Date.now() + LIVE_TTL_MS };
  return data;
}

/**
 * Fetch historical NCAAB odds snapshot for a specific UTC timestamp.
 * @param dateIso  e.g. "2024-03-21T12:00:00Z"
 */
export async function getHistoricalOdds(dateIso: string): Promise<ConsensusOdds[]> {
  const cached = histCache.get(dateIso);
  if (cached && Date.now() < cached.expiresAt) {
    return cached.data;
  }

  const events = await fetchFromApi(
    `${ODDS_API_BASE}/historical/sports/${SPORT}/odds`,
    {
      regions: "us",
      markets: "h2h,spreads",
      oddsFormat: "american",
      date: dateIso,
    },
  );

  const data = events.map(parseEvent);
  histCache.set(dateIso, { data, expiresAt: Date.now() + HIST_TTL_MS });
  return data;
}

/**
 * Build a Map<teamPairKey, ConsensusOdds> from a list of ConsensusOdds,
 * resolving team names to our canonical IDs.
 *
 * teamPairKey = sorted([teamAId, teamBId]).join("::") for O(1) lookup.
 */
export function buildOddsMap(
  oddsList: ConsensusOdds[],
  teamMap: Map<string, string>,
): OddsMap {
  const map: OddsMap = new Map();
  for (const odds of oddsList) {
    const homeId = matchTeam(odds.homeTeam, teamMap);
    const awayId = matchTeam(odds.awayTeam, teamMap);
    if (homeId && awayId) {
      const key = [homeId, awayId].sort().join("::");
      map.set(key, { ...odds });
    }
  }
  return map;
}

/**
 * Look up odds for a specific matchup. Returns odds oriented so that
 * teamAId is the "home" reference (flipping if needed).
 */
export function getMatchupOdds(
  teamAId: string,
  teamBId: string,
  oddsMap: OddsMap,
): { odds: ConsensusOdds; teamAIsHome: boolean } | null {
  const key = [teamAId, teamBId].sort().join("::");
  const odds = oddsMap.get(key);
  if (!odds) return null;

  const homeId = matchTeam(odds.homeTeam, buildReverseLookup(oddsMap))
  // Determine orientation by checking which team was home in the API event.
  // Since tournament games are neutral, "home" in the API is arbitrary.
  // We expose both orientations via teamAIsHome flag.
  const teamAIsHome =
    odds.homeTeam.toLowerCase().includes(teamAId.replace(/-/g, " ").split(" ")[0] ?? "");

  return { odds, teamAIsHome };
}

function buildReverseLookup(oddsMap: OddsMap): Map<string, string> {
  const map = new Map<string, string>();
  for (const odds of Array.from(oddsMap.values())) {
    map.set(odds.homeTeam, odds.homeTeam);
    map.set(odds.awayTeam, odds.awayTeam);
  }
  return map;
}

/** Convert an American moneyline to a display string like "-350" or "+175". */
export function formatMoneyline(price: number | null): string {
  if (price === null) return "N/A";
  return price >= 0 ? `+${price}` : `${price}`;
}

/** Convert spread to display string like "-6.5" or "+3". */
export function formatSpreadDisplay(spread: number | null): string {
  if (spread === null) return "N/A";
  if (spread === 0) return "PK";
  return spread > 0 ? `+${spread}` : `${spread}`;
}
