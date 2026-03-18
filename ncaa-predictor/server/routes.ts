import { ZodError } from "zod";
import type { Express, NextFunction, Request, Response } from "express";
import type { Server } from "http";
import {
  bracketRequestSchema,
  gamesQuerySchema,
  matchupQuerySchema,
  teamsQuerySchema,
} from "@shared/schema";
import { logger } from "./logger";
import { storage } from "./storage";
import {
  buildOddsMap,
  formatMoneyline,
  formatSpreadDisplay,
  getLiveOdds,
  matchTeam,
  type ConsensusOdds,
} from "./services/odds-service";

function handleRouteError(error: unknown, next: NextFunction) {
  if (error instanceof ZodError) {
    const validationError = new Error(error.issues.map((issue) => issue.message).join(", "));
    (validationError as Error & { status?: number }).status = 400;
    next(validationError);
    return;
  }

  next(error);
}

/** Refresh the live odds map on the prediction service (best-effort). */
async function refreshOdds(): Promise<void> {
  try {
    const predictionService = storage.getPredictionService();
    const teamMap = predictionService.getTeamNameMap();
    const liveOdds = await getLiveOdds();
    const oddsMap = buildOddsMap(liveOdds, teamMap);
    predictionService.setOddsMap(oddsMap);
    logger.info("odds refreshed", { matchedGames: oddsMap.size, liveEvents: liveOdds.length });
  } catch (err) {
    logger.warn("odds refresh failed (non-fatal)", { error: err instanceof Error ? err.message : String(err) });
  }
}

export async function registerRoutes(
  httpServer: Server,
  app: Express,
): Promise<Server> {
  // Warm up odds on startup and refresh every 12 hours
  void refreshOdds();
  setInterval(() => void refreshOdds(), 12 * 60 * 60 * 1000);

  app.get("/api/teams", async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { season } = teamsQuerySchema.parse(req.query);
      const predictionService = storage.getPredictionService();
      const teams = await storage.listTeams(season);

      res.json({
        season,
        teams,
        modelRun: predictionService.getModelRun(),
      });
    } catch (error) {
      handleRouteError(error, next);
    }
  });

  app.get("/api/games", async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { date, season } = gamesQuerySchema.parse(req.query);
      const games = await storage.listGamesByDate(date);
      const predictionService = storage.getPredictionService();

      res.json({
        season,
        date,
        games: predictionService.projectGames(games),
        modelRun: predictionService.getModelRun(),
      });
    } catch (error) {
      handleRouteError(error, next);
    }
  });

  app.get(
    "/api/predictions/matchup",
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { teamAId, teamBId, season, date, venue } = matchupQuerySchema.parse(req.query);
        const predictionService = storage.getPredictionService();
        const matchup = predictionService.predictMatchup(teamAId, teamBId, date, venue, season);
        res.json(matchup);
      } catch (error) {
        handleRouteError(error, next);
      }
    },
  );

  app.post(
    "/api/predictions/bracket",
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { entries, simulations, seed } = bracketRequestSchema.parse(req.body ?? {});
        const predictionService = storage.getPredictionService();
        const field = entries ?? (await storage.getBracketField());

        res.json(predictionService.simulateBracket(field, simulations, seed));
      } catch (error) {
        handleRouteError(error, next);
      }
    },
  );

  app.get("/api/model-runs/latest", async (_req, res) => {
    res.json({
      modelRun: storage.getPredictionService().getModelRun(),
    });
  });

  app.get("/api/model-runs/calibration", async (_req, res, next) => {
    try {
      const fs = await import("fs");
      const path = await import("path");
      const candidates = [
        path.resolve(process.cwd(), "data/models/tournament-consensus-report.json"),
        path.resolve(process.cwd(), "dist/models/tournament-consensus-report.json"),
      ];
      const reportPath = candidates.find((p) => fs.existsSync(p));
      if (!reportPath) {
        res.json({ calibrationBuckets: {}, pooledMetrics: {} });
        return;
      }
      const report = JSON.parse(fs.readFileSync(reportPath, "utf-8"));
      res.json({
        calibrationBuckets: report.calibrationBuckets ?? {},
        pooledMetrics: report.pooledMetrics ?? {},
        bestCandidate: report.promotion?.bestCandidate ?? null,
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * GET /api/odds
   * Returns live NCAAB odds from the-odds-api, enriched with our canonical
   * team IDs where matchable. Cached for 5 minutes server-side.
   */
  app.get("/api/odds", async (_req: Request, res: Response, next: NextFunction) => {
    try {
      const predictionService = storage.getPredictionService();
      const teamMap = predictionService.getTeamNameMap();
      const liveOdds = await getLiveOdds();

      const enriched = liveOdds.map((odds: ConsensusOdds) => {
        const homeTeamId = matchTeam(odds.homeTeam, teamMap);
        const awayTeamId = matchTeam(odds.awayTeam, teamMap);
        return {
          ...odds,
          homeTeamId,
          awayTeamId,
          moneylineHomeDisplay: formatMoneyline(odds.bestMoneylineHome),
          moneylineAwayDisplay: formatMoneyline(odds.bestMoneylineAway),
          spreadDisplay: formatSpreadDisplay(odds.consensusSpreadHome),
        };
      });

      res.json({ odds: enriched, count: enriched.length, fetchedAt: new Date().toISOString() });
    } catch (error) {
      next(error);
    }
  });

  return httpServer;
}
