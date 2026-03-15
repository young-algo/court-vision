import { type BracketEntry, type Game, type Team } from "@shared/schema";
import {
  buildModelRunFromArtifact,
  loadLearnedTournamentArtifact,
} from "./services/learned-tournament-artifact";
import { PredictionService } from "./services/prediction-service";
import { loadSeedData } from "./services/seed-data";

export interface IStorage {
  listTeams(season?: number): Promise<Team[]>;
  listGamesByDate(date: string): Promise<Game[]>;
  getBracketField(): Promise<BracketEntry[]>;
  getPredictionService(): PredictionService;
}

export class AppStorage implements IStorage {
  private readonly seedData = loadSeedData();
  private readonly learnedArtifact = loadLearnedTournamentArtifact();
  private readonly activeModelRun = this.learnedArtifact
    ? buildModelRunFromArtifact(this.learnedArtifact, this.seedData.modelRun.coverage)
    : this.seedData.modelRun;
  private readonly predictionService = new PredictionService(
    this.seedData.teams,
    this.seedData.snapshots,
    this.activeModelRun,
    this.learnedArtifact,
  );

  async listTeams(_season?: number): Promise<Team[]> {
    return this.seedData.teams;
  }

  async listGamesByDate(date: string): Promise<Game[]> {
    return this.seedData.games.filter((game) => game.date === date);
  }

  async getBracketField(): Promise<BracketEntry[]> {
    return this.seedData.bracketField;
  }

  getPredictionService() {
    return this.predictionService;
  }
}

export const storage = new AppStorage();
