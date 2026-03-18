import type { BracketRoundOdds, Team } from "@shared/schema";
import { formatPercent } from "@/lib/prediction-display";

interface BracketRegionProps {
  region: string;
  teams: Array<{
    team: Team | undefined;
    seed: number;
    odds: BracketRoundOdds;
  }>;
}

const SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15];

export function BracketRegion({ region, teams }: BracketRegionProps) {
  const ordered = SEED_ORDER.map((seed) => teams.find((t) => t.seed === seed)).filter(Boolean) as BracketRegionProps["teams"];

  if (ordered.length === 0) return null;

  return (
    <div className="rounded-2xl border border-border bg-card p-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
        {region}
      </div>
      <div className="space-y-0.5">
        {ordered.map((entry, i) => {
          const isPairBoundary = i % 2 === 0 && i > 0;
          return (
            <div key={entry.seed}>
              {isPairBoundary && <div className="h-1.5" />}
              <div className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-muted/50 transition-colors">
                <div className="w-5 text-right text-[11px] font-medium text-muted-foreground tabular-nums">
                  {entry.seed}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    {entry.team?.shortName ?? `Seed ${entry.seed}`}
                  </div>
                </div>
                <div className="flex gap-2 text-[11px] tabular-nums text-muted-foreground">
                  <span title="Sweet 16">{formatPercent(entry.odds.sweet16)}</span>
                  <span title="Elite 8" className="hidden sm:inline">{formatPercent(entry.odds.elite8)}</span>
                  <span title="Final Four" className="font-medium text-foreground">{formatPercent(entry.odds.finalFour)}</span>
                </div>
                <div className="w-12 text-right">
                  <div
                    className="inline-block rounded-full px-1.5 py-0.5 text-[10px] font-semibold tabular-nums"
                    style={{
                      backgroundColor: `rgba(249, 115, 22, ${Math.min(entry.odds.champion * 8, 0.3)})`,
                      color: entry.odds.champion > 0.05 ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))",
                    }}
                  >
                    {formatPercent(entry.odds.champion)}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
