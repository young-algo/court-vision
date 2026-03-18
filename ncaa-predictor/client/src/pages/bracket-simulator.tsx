import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  Crown,
  GitBranchPlus,
  RefreshCcw,
  Spline,
  Trophy,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { BracketRegion } from "@/components/bracket-region";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ModelStatusCard } from "@/components/model-status-card";
import { fetchBracketSimulation, fetchTeams } from "@/lib/api";
import { formatPercent } from "@/lib/prediction-display";
import { BRACKET_REGIONS } from "@shared/schema";

export default function BracketSimulatorPage() {
  const [simulations, setSimulations] = useState(4000);
  const [seed, setSeed] = useState(2026);

  const teamsQuery = useQuery({
    queryKey: ["teams", 2026],
    queryFn: () => fetchTeams(2026),
  });

  const simulationQuery = useQuery({
    queryKey: ["bracket", simulations, seed],
    queryFn: () => fetchBracketSimulation(simulations, seed),
  });

  const teamsById = useMemo(() => {
    return new Map((teamsQuery.data?.teams ?? []).map((team) => [team.id, team]));
  }, [teamsQuery.data?.teams]);

  const titleOdds = simulationQuery.data?.titleOdds ?? [];
  const roundOdds = simulationQuery.data?.roundOdds ?? [];

  return (
    <div className="space-y-6">
      <section className="rounded-[28px] border border-primary/15 bg-[linear-gradient(140deg,rgba(15,23,42,0.04),rgba(249,115,22,0.14),rgba(15,23,42,0.02))] px-6 py-7">
        <div className="flex flex-wrap items-start justify-between gap-5">
          <div className="max-w-2xl">
            <Badge className="bg-primary/10 text-primary hover:bg-primary/10">
              Monte Carlo bracket lab
            </Badge>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight">Bracket simulation from the seeded field</h1>
            <p className="mt-3 text-sm leading-6 text-muted-foreground">
              The simulator runs the current 64-team seeded field through the same backend matchup model used by the matchup and slate APIs. This gives you round advancement odds, title equity, and the most likely championship pairing.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <div className="rounded-2xl border border-border bg-card px-4 py-3 shadow-sm">
              <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                Simulations
              </div>
              <input
                type="number"
                min={100}
                max={20000}
                step={100}
                value={simulations}
                onChange={(event) => setSimulations(Number(event.target.value) || 4000)}
                className="mt-2 w-28 bg-transparent text-lg font-semibold outline-none"
              />
            </div>
            <div className="rounded-2xl border border-border bg-card px-4 py-3 shadow-sm">
              <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                RNG seed
              </div>
              <input
                type="number"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value) || 2026)}
                className="mt-2 w-24 bg-transparent text-lg font-semibold outline-none"
              />
            </div>
            <Button type="button" variant="outline" className="self-end" disabled>
              <RefreshCcw className="mr-2 h-4 w-4" />
              Query refreshes on change
            </Button>
          </div>
        </div>
      </section>

      {simulationQuery.data?.modelRun ? <ModelStatusCard modelRun={simulationQuery.data.modelRun} /> : null}

      {simulationQuery.data ? (
        <>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {BRACKET_REGIONS.map((region) => {
            const regionField = simulationQuery.data!.field.filter((e) => e.region === region);
            const regionTeams = regionField.map((entry) => ({
              team: teamsById.get(entry.teamId),
              seed: entry.seed,
              odds: roundOdds.find((o) => o.teamId === entry.teamId) ?? {
                teamId: entry.teamId, roundOf32: 0, sweet16: 0, elite8: 0, finalFour: 0, championship: 0, champion: 0,
              },
            }));
            return <BracketRegion key={region} region={region} teams={regionTeams} />;
          })}
        </div>
        <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <div className="space-y-6">
            <Card className="px-5 py-5">
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <Crown className="h-3.5 w-3.5" />
                Most likely title paths
              </div>
              <div className="mt-4 space-y-3">
                {titleOdds.slice(0, 8).map((entry, index) => {
                  const team = teamsById.get(entry.teamId);
                  return (
                    <div
                      key={entry.teamId}
                      className="grid grid-cols-[auto_1fr_auto] items-center gap-3 rounded-2xl bg-muted/45 px-4 py-3"
                    >
                      <div className="text-sm font-semibold text-muted-foreground">#{index + 1}</div>
                      <div>
                        <div className="text-sm font-semibold">{team?.shortName ?? entry.teamId}</div>
                        <div className="text-[11px] text-muted-foreground">
                          {team?.conference ?? "Field"}
                        </div>
                      </div>
                      <div className="text-right text-lg font-semibold tabular-nums">
                        {formatPercent(entry.probability)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>

            <Card className="px-5 py-5">
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <GitBranchPlus className="h-3.5 w-3.5" />
                Most likely final four
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {simulationQuery.data.mostLikelyFinalFour.map((teamId) => {
                  const team = teamsById.get(teamId);
                  return (
                    <div key={teamId} className="rounded-2xl border border-border bg-card px-4 py-4">
                      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        Region champion
                      </div>
                      <div className="mt-2 text-lg font-semibold">{team?.shortName ?? teamId}</div>
                      <div className="mt-1 text-sm text-muted-foreground">{team?.conference}</div>
                    </div>
                  );
                })}
              </div>
            </Card>

            <Card className="px-5 py-5">
              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <Spline className="h-3.5 w-3.5" />
                Most likely championship game
              </div>
              <div className="mt-4 rounded-2xl bg-muted/45 px-4 py-4">
                <div className="text-lg font-semibold">
                  {teamsById.get(simulationQuery.data.mostLikelyChampionship.teamAId)?.shortName ??
                    simulationQuery.data.mostLikelyChampionship.teamAId}
                  {" vs "}
                  {teamsById.get(simulationQuery.data.mostLikelyChampionship.teamBId)?.shortName ??
                    simulationQuery.data.mostLikelyChampionship.teamBId}
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  Simulated frequency{" "}
                  <span className="font-semibold text-foreground">
                    {formatPercent(simulationQuery.data.mostLikelyChampionship.probability)}
                  </span>
                </div>
              </div>
            </Card>
          </div>

          <Card className="px-5 py-5">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
              <BarChart3 className="h-3.5 w-3.5" />
              Advancement odds
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                  <tr>
                    <th className="pb-3 pr-4 font-medium">Team</th>
                    <th className="pb-3 pr-4 font-medium">R32</th>
                    <th className="pb-3 pr-4 font-medium">S16</th>
                    <th className="pb-3 pr-4 font-medium">E8</th>
                    <th className="pb-3 pr-4 font-medium">F4</th>
                    <th className="pb-3 pr-4 font-medium">Title game</th>
                    <th className="pb-3 font-medium">Champion</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {roundOdds.slice(0, 16).map((entry) => {
                    const team = teamsById.get(entry.teamId);
                    return (
                      <tr key={entry.teamId}>
                        <td className="py-3 pr-4">
                          <div className="font-semibold">{team?.shortName ?? entry.teamId}</div>
                          <div className="text-xs text-muted-foreground">{team?.conference}</div>
                        </td>
                        <td className="py-3 pr-4 tabular-nums">{formatPercent(entry.roundOf32)}</td>
                        <td className="py-3 pr-4 tabular-nums">{formatPercent(entry.sweet16)}</td>
                        <td className="py-3 pr-4 tabular-nums">{formatPercent(entry.elite8)}</td>
                        <td className="py-3 pr-4 tabular-nums">{formatPercent(entry.finalFour)}</td>
                        <td className="py-3 pr-4 tabular-nums">{formatPercent(entry.championship)}</td>
                        <td className="py-3 font-semibold tabular-nums text-primary">
                          {formatPercent(entry.champion)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="mt-5 rounded-2xl bg-muted/40 px-4 py-4 text-sm text-muted-foreground">
              <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-foreground">
                <Trophy className="h-3.5 w-3.5 text-primary" />
                Build note
              </div>
              This simulator is reproducible because the backend accepts a seed and versioned model run. The field itself is generated from the seeded top-64 snapshot available in the repo today; swapping in a real schedule and selection model later will not require changing the API shape.
            </div>
          </Card>
        </div>
        </>
      ) : null}
    </div>
  );
}
