import { ZodError } from "zod";
import type { Express, NextFunction, Request, Response } from "express";
import type { Server } from "http";
import {
  bracketRequestSchema,
  gamesQuerySchema,
  matchupQuerySchema,
  teamsQuerySchema,
} from "@shared/schema";
import { storage } from "./storage";

function handleRouteError(error: unknown, next: NextFunction) {
  if (error instanceof ZodError) {
    const validationError = new Error(error.issues.map((issue) => issue.message).join(", "));
    (validationError as Error & { status?: number }).status = 400;
    next(validationError);
    return;
  }

  next(error);
}

export async function registerRoutes(
  httpServer: Server,
  app: Express,
): Promise<Server> {
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

  return httpServer;
}
