/**
 * OddsDisplay — compact inline widget for showing betting market odds.
 *
 * Renders moneyline and/or spread for a matchup. Used in both the
 * Daily Slate game cards and the Matchup Predictor detail panel.
 */
import { TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { MatchupOdds } from "@shared/schema";

interface OddsDisplayProps {
  odds: MatchupOdds;
  teamAName: string;
  teamBName: string;
  /** If true, renders a compact single-line version for game cards */
  compact?: boolean;
}

function formatMoneyline(price: number | null): string {
  if (price === null) return "–";
  return price >= 0 ? `+${price}` : `${price}`;
}

function formatSpread(spread: number | null): string {
  if (spread === null) return "–";
  if (spread === 0) return "PK";
  return spread > 0 ? `+${spread.toFixed(1)}` : spread.toFixed(1);
}

function impliedProbToPercent(prob: number | null): string {
  if (prob === null) return "–";
  return `${(prob * 100).toFixed(1)}%`;
}

export function OddsDisplay({ odds, teamAName, teamBName, compact = false }: OddsDisplayProps) {
  const hasMoneyline = odds.moneylineA !== null || odds.moneylineB !== null;
  const hasSpread = odds.spreadA !== null;
  const hasOdds = hasMoneyline || hasSpread;

  if (!hasOdds) return null;

  const spreadADisplay = formatSpread(odds.spreadA);
  const spreadBDisplay = odds.spreadA !== null ? formatSpread(-odds.spreadA) : "–";

  if (compact) {
    // Single-row compact version for game cards
    return (
      <div className="flex items-center gap-3 rounded-xl border border-border/60 bg-muted/30 px-3 py-2">
        <div className="flex items-center gap-1.5 text-xs uppercase tracking-[0.14em] text-muted-foreground">
          <TrendingUp className="h-3 w-3 text-emerald-500" />
          <span className="text-emerald-600 dark:text-emerald-400 font-medium">Live odds</span>
        </div>
        <div className="h-3 w-px bg-border" />
        {hasSpread && (
          <>
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-muted-foreground">{teamAName}</span>
              <span className="font-semibold tabular-nums">{spreadADisplay}</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-muted-foreground">{teamBName}</span>
              <span className="font-semibold tabular-nums">{spreadBDisplay}</span>
            </div>
          </>
        )}
        {hasMoneyline && !hasSpread && (
          <>
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-muted-foreground">{teamAName}</span>
              <span className="font-semibold tabular-nums">{formatMoneyline(odds.moneylineA)}</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-muted-foreground">{teamBName}</span>
              <span className="font-semibold tabular-nums">{formatMoneyline(odds.moneylineB)}</span>
            </div>
          </>
        )}
        {odds.bookmakerCount > 0 && (
          <Badge variant="outline" className="ml-auto text-[9px] uppercase tracking-[0.14em] text-muted-foreground">
            {odds.bookmakerCount} books
          </Badge>
        )}
      </div>
    );
  }

  // Full detail version for the matchup predictor
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
        <TrendingUp className="h-3.5 w-3.5 text-emerald-500" />
        Betting market
        {odds.bookmakerCount > 0 && (
          <Badge variant="outline" className="ml-auto text-[9px] uppercase tracking-[0.14em]">
            {odds.bookmakerCount} bookmaker{odds.bookmakerCount !== 1 ? "s" : ""}
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        {hasSpread && (
          <>
            <div className="rounded-2xl bg-muted/45 p-3">
              <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                {teamAName} spread
              </div>
              <div className="mt-1.5 text-xl font-semibold tabular-nums">
                {spreadADisplay}
              </div>
              <div className="mt-0.5 text-xs text-muted-foreground">
                {odds.spreadA !== null && odds.spreadA < 0 ? "favored" : "underdog"}
              </div>
            </div>
            <div className="rounded-2xl bg-muted/45 p-3">
              <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                {teamBName} spread
              </div>
              <div className="mt-1.5 text-xl font-semibold tabular-nums">
                {spreadBDisplay}
              </div>
              <div className="mt-0.5 text-xs text-muted-foreground">
                {odds.spreadA !== null && odds.spreadA > 0 ? "favored" : "underdog"}
              </div>
            </div>
          </>
        )}

        {hasMoneyline && (
          <>
            <div className="rounded-2xl bg-muted/45 p-3">
              <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                {teamAName} moneyline
              </div>
              <div className="mt-1.5 text-xl font-semibold tabular-nums">
                {formatMoneyline(odds.moneylineA)}
              </div>
              {odds.impliedProbA !== null && (
                <div className="mt-0.5 text-xs text-muted-foreground tabular-nums">
                  {impliedProbToPercent(odds.impliedProbA)} implied
                </div>
              )}
            </div>
            <div className="rounded-2xl bg-muted/45 p-3">
              <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                {teamBName} moneyline
              </div>
              <div className="mt-1.5 text-xl font-semibold tabular-nums">
                {formatMoneyline(odds.moneylineB)}
              </div>
              {odds.impliedProbA !== null && (
                <div className="mt-0.5 text-xs text-muted-foreground tabular-nums">
                  {impliedProbToPercent(1 - odds.impliedProbA)} implied
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {odds.impliedProbA !== null && (
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-3 py-2.5">
          <div className="text-xs text-muted-foreground">
            Market-implied win probability (vig-removed):{" "}
            <span className="font-semibold text-foreground">
              {teamAName} {impliedProbToPercent(odds.impliedProbA)}
            </span>
            {" · "}
            <span className="font-semibold text-foreground">
              {teamBName} {impliedProbToPercent(1 - odds.impliedProbA)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
