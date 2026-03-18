/**
 * Environment variable helpers for the server.
 * Reads from process.env (populated via dotenv in dev, real env in prod).
 */
import { config as loadDotenv } from "dotenv";
import path from "path";

// Load .env from project root in development
loadDotenv({ path: path.resolve(process.cwd(), ".env") });

function required(key: string): string {
  const value = process.env[key];
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value;
}

function optional(key: string, defaultValue = ""): string {
  return process.env[key] ?? defaultValue;
}

export const env = {
  get ODDS_API_KEY(): string {
    return required("ODDS_API_KEY");
  },
  get NODE_ENV(): string {
    return optional("NODE_ENV", "development");
  },
  get PORT(): number {
    return parseInt(optional("PORT", "5100"), 10);
  },
};
