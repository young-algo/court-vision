import type {
  BracketSimulationResult,
  GamesResponse,
  MatchupPredictionResponse,
  ModelRunResponse,
  TeamsResponse,
  Venue,
} from "@shared/schema";
import { apiRequest } from "./queryClient";

async function readJson<T>(response: Response): Promise<T> {
  return (await response.json()) as T;
}

export async function fetchTeams(season = 2026) {
  const response = await apiRequest("GET", `/api/teams?season=${season}`);
  return readJson<TeamsResponse>(response);
}

export async function fetchGames(date: string, season = 2026) {
  const response = await apiRequest("GET", `/api/games?date=${date}&season=${season}`);
  return readJson<GamesResponse>(response);
}

export async function fetchMatchupPrediction(
  teamAId: string,
  teamBId: string,
  date: string,
  venue: Venue,
) {
  const params = new URLSearchParams({
    teamAId,
    teamBId,
    date,
    venue,
  });
  const response = await apiRequest("GET", `/api/predictions/matchup?${params.toString()}`);
  return readJson<MatchupPredictionResponse>(response);
}

export async function fetchBracketSimulation(simulations = 4000, seed = 2026) {
  const response = await apiRequest("POST", "/api/predictions/bracket", {
    simulations,
    seed,
  });
  return readJson<BracketSimulationResult>(response);
}

export async function fetchLatestModelRun() {
  const response = await apiRequest("GET", "/api/model-runs/latest");
  return readJson<ModelRunResponse>(response);
}
