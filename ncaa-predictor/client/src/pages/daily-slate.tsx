import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CalendarDays, Flag, Gauge, Sparkles, Trophy } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { ModelStatusCard } from "@/components/model-status-card";
import { fetchGames } from "@/lib/api";
import { formatPercent, formatSigned } from "@/lib/prediction-display";

const DEFAULT_DATE = "2026-03-14";

export default function DailySlatePage() {
  const [date, setDate] = useState(DEFAULT_DATE);

  const gamesQuery = useQuery({
    queryKey: ["games", date],
    queryFn: () => fetchGames(date),
  });

  const games = gamesQuery.data?.games ?? [];

  return (
    <div className="space-y-6">
      <section className="rounded-[28px] border border-primary/15 bg-[linear-gradient(120deg,rgba(249,115,22,0.12),rgba(15,23,42,0.02)_35%,rgba(56,189,248,0.1))] px-6 py-7">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <Badge variant="secondary" className="text-[10px] uppercase tracking-[0.18em]">
              Seeded tournament slate
            </Badge>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight">Daily projection board</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
              This slate is generated from the current seeded field built off the 2026 rating snapshot. The API returns full game cards with matchup projections, confidence, and upset scores for the date you choose.
            </p>
          </div>
          <div className="rounded-2xl border border-border bg-card px-4 py-3 shadow-sm">
            <label className="mb-1 block text-xs uppercase tracking-[0.18em] text-muted-foreground">
              Slate date
            </label>
            <div className="flex items-center gap-2">
              <CalendarDays className="h-4 w-4 text-muted-foreground" />
              <input
                type="date"
                value={date}
                onChange={(event) => setDate(event.target.value)}
                className="bg-transparent text-sm outline-none"
              />
            </div>
          </div>
        </div>
      </section>

      {gamesQuery.data?.modelRun ? <ModelStatusCard modelRun={gamesQuery.data.modelRun} /> : null}

      <div className="grid gap-4">
        {games.map((game) => {
          const favorite = game.prediction.winProbabilityA >= 0.5 ? game.teamA : game.teamB;
          const underdog = game.prediction.winProbabilityA >= 0.5 ? game.teamB : game.teamA;

          return (
            <Card key={game.id} className="border-border/70 bg-card/95 px-5 py-5">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline" className="text-[10px] uppercase tracking-[0.16em]">
                      {game.region}
                    </Badge>
                    <Badge variant="secondary" className="text-[10px] uppercase tracking-[0.16em]">
                      {game.round?.replaceAll("_", " ")}
                    </Badge>
                  </div>
                  <div className="mt-3 text-xl font-semibold tracking-tight">
                    {game.teamA.shortName} vs {game.teamB.shortName}
                  </div>
                  <div className="mt-1 text-sm text-muted-foreground">
                    {game.teamA.recordLabel} • #{game.teamA.torvikRank} vs {game.teamB.recordLabel} • #{game.teamB.torvikRank}
                  </div>
                </div>
                <div className="rounded-2xl bg-muted/50 px-4 py-3 text-right">
                  <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                    Projected spread
                  </div>
                  <div className="mt-2 text-2xl font-semibold">{game.prediction.impliedSpread}</div>
                </div>
              </div>

              <div className="mt-5 grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl bg-muted/45 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                      {game.teamA.shortName}
                    </div>
                    <div className="mt-2 text-2xl font-semibold tabular-nums">
                      {formatPercent(game.prediction.winProbabilityA)}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-muted/45 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                      {game.teamB.shortName}
                    </div>
                    <div className="mt-2 text-2xl font-semibold tabular-nums">
                      {formatPercent(game.prediction.winProbabilityB)}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-muted/45 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                      Upset score
                    </div>
                    <div className="mt-2 text-2xl font-semibold tabular-nums">
                      {game.prediction.upsetScore.toFixed(1)}
                    </div>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-2xl border border-border p-4">
                    <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                      <Gauge className="h-3.5 w-3.5" />
                      Favorite signal
                    </div>
                    <div className="mt-2 text-sm leading-6 text-muted-foreground">
                      {favorite.shortName} leads the projection by{" "}
                      <span className="font-semibold text-foreground">
                        {formatSigned(game.prediction.expectedMargin)}
                      </span>{" "}
                      with a {formatPercent(Math.max(game.prediction.winProbabilityA, game.prediction.winProbabilityB))} win rate.
                    </div>
                  </div>
                  <div className="rounded-2xl border border-border p-4">
                    <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                      <Sparkles className="h-3.5 w-3.5" />
                      Volatility note
                    </div>
                    <div className="mt-2 text-sm leading-6 text-muted-foreground">
                      {underdog.shortName} still carries a non-trivial upset path because the uncertainty band spans{" "}
                      {formatSigned(game.prediction.uncertaintyBand.low)} to{" "}
                      {formatSigned(game.prediction.uncertaintyBand.high)}.
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-5 grid gap-3 md:grid-cols-3">
                {game.prediction.keyFactors.slice(0, 3).map((factor) => (
                  <div key={factor} className="rounded-2xl bg-muted/35 p-4 text-sm leading-6 text-muted-foreground">
                    <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-foreground">
                      <Flag className="h-3.5 w-3.5 text-primary" />
                      Key factor
                    </div>
                    {factor}
                  </div>
                ))}
              </div>
            </Card>
          );
        })}

        {!games.length && !gamesQuery.isLoading ? (
          <Card className="px-5 py-10 text-center">
            <Trophy className="mx-auto h-8 w-8 text-primary" />
            <div className="mt-3 text-lg font-semibold">No games on this seeded slate date</div>
            <div className="mt-2 text-sm text-muted-foreground">
              Try `2026-03-14` or `2026-03-15`, the two seeded tournament opening-round dates available in this build.
            </div>
          </Card>
        ) : null}
      </div>
    </div>
  );
}
