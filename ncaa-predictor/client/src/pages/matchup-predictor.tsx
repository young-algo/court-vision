import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowRightLeft,
  CalendarDays,
  Gauge,
  Info,
  Layers3,
  Shield,
  Sparkles,
  Target,
  TrendingUp,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ModelStatusCard } from "@/components/model-status-card";
import { TeamSearchSelect } from "@/components/team-search-select";
import { fetchMatchupPrediction, fetchTeams } from "@/lib/api";
import { confidenceCopy, formatPercent, formatSigned } from "@/lib/prediction-display";
import type { Team, Venue } from "@shared/schema";

const DEFAULT_DATE = "2026-03-14";

function VenueSelector({
  venue,
  onChange,
  teamA,
  teamB,
}: {
  venue: Venue;
  onChange: (venue: Venue) => void;
  teamA: Team | null;
  teamB: Team | null;
}) {
  const options: Array<{ value: Venue; label: string }> = [
    { value: "neutral", label: "Neutral court" },
    { value: "home_a", label: teamA ? `${teamA.shortName} home` : "Team A home" },
    { value: "home_b", label: teamB ? `${teamB.shortName} home` : "Team B home" },
  ];

  return (
    <div className="grid grid-cols-3 gap-1 rounded-xl bg-muted p-1">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={`rounded-lg px-3 py-2 text-xs font-medium transition-all ${
            venue === option.value
              ? "bg-card text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function StatRow({
  label,
  valueA,
  valueB,
  icon,
  inverse = false,
}: {
  label: string;
  valueA: string;
  valueB: string;
  icon: ReactNode;
  inverse?: boolean;
}) {
  const aValue = Number(valueA.replace(/[^\d.-]/g, ""));
  const bValue = Number(valueB.replace(/[^\d.-]/g, ""));
  const aWins = Number.isFinite(aValue) && Number.isFinite(bValue) && (inverse ? aValue < bValue : aValue > bValue);
  const bWins = Number.isFinite(aValue) && Number.isFinite(bValue) && (inverse ? bValue < aValue : bValue > aValue);

  return (
    <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 py-2">
      <div className={`text-sm font-medium tabular-nums ${aWins ? "text-primary" : "text-foreground"}`}>
        {valueA}
      </div>
      <div className="flex items-center gap-1 text-xs uppercase tracking-[0.18em] text-muted-foreground">
        {icon}
        {label}
      </div>
      <div className={`text-right text-sm font-medium tabular-nums ${bWins ? "text-primary" : "text-foreground"}`}>
        {valueB}
      </div>
    </div>
  );
}

export default function MatchupPredictor() {
  const [teamA, setTeamA] = useState<Team | null>(null);
  const [teamB, setTeamB] = useState<Team | null>(null);
  const [venue, setVenue] = useState<Venue>("neutral");
  const [date, setDate] = useState(DEFAULT_DATE);

  const teamsQuery = useQuery({
    queryKey: ["teams", 2026],
    queryFn: () => fetchTeams(2026),
  });

  const teams = teamsQuery.data?.teams ?? [];

  useEffect(() => {
    if (!teamA && teams[0]) {
      setTeamA(teams[0]);
    }

    if (!teamB && teams[1]) {
      setTeamB(teams[1]);
    }
  }, [teamA, teamB, teams]);

  const matchupQuery = useQuery({
    queryKey: ["matchup", teamA?.id, teamB?.id, date, venue],
    queryFn: () => fetchMatchupPrediction(teamA!.id, teamB!.id, date, venue),
    enabled: Boolean(teamA && teamB),
  });

  const prediction = matchupQuery.data;
  const modelRun = prediction?.modelRun ?? teamsQuery.data?.modelRun;

  const favorite = useMemo(() => {
    if (!prediction) return null;
    return prediction.prediction.expectedMargin >= 0 ? prediction.teamA : prediction.teamB;
  }, [prediction]);

  return (
    <div className="space-y-6">
      <section className="overflow-hidden rounded-[28px] border border-primary/15 bg-[radial-gradient(circle_at_top_left,_rgba(249,115,22,0.18),_transparent_36%),linear-gradient(135deg,rgba(15,23,42,0.96),rgba(30,41,59,0.88))] px-6 py-7 text-white shadow-2xl">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="max-w-2xl">
            <Badge className="border-white/15 bg-white/10 text-white hover:bg-white/10">
              API-backed matchup lab
            </Badge>
            <h1 className="mt-4 text-3xl font-semibold tracking-tight sm:text-4xl">
              Matchup projections with a server-side ensemble.
            </h1>
            <p className="mt-3 max-w-xl text-sm leading-6 text-white/78 sm:text-base">
              This view now pulls typed predictions from the backend instead of bundling fixed rankings into the browser. The current slate and bracket field are seeded from the available 2026 snapshot while the rest of the platform architecture moves server-side.
            </p>
          </div>
          {modelRun ? (
            <div className="grid min-w-[240px] grid-cols-2 gap-3 text-sm">
              <div className="rounded-2xl border border-white/10 bg-white/8 p-4">
                <div className="text-white/60">Teams covered</div>
                <div className="mt-2 text-2xl font-semibold">{modelRun.coverage.teams}</div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/8 p-4">
                <div className="text-white/60">Field size</div>
                <div className="mt-2 text-2xl font-semibold">{modelRun.coverage.fieldSize}</div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/8 p-4">
                <div className="text-white/60">Log loss</div>
                <div className="mt-2 text-2xl font-semibold">{modelRun.metrics.logLoss.toFixed(3)}</div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/8 p-4">
                <div className="text-white/60">Margin MAE</div>
                <div className="mt-2 text-2xl font-semibold">{modelRun.metrics.marginMae.toFixed(1)}</div>
              </div>
            </div>
          ) : null}
        </div>
      </section>

      <Card className="border-border/70 bg-card/95 px-5 py-5">
        <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr]">
          <TeamSearchSelect
            label="Team A"
            teams={teams}
            selectedTeam={teamA}
            otherTeam={teamB}
            onSelect={setTeamA}
            side="left"
          />
          <div className="flex items-end justify-center">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="mb-0.5"
              onClick={() => {
                setTeamA(teamB);
                setTeamB(teamA);
                setVenue((currentVenue) => {
                  if (currentVenue === "home_a") return "home_b";
                  if (currentVenue === "home_b") return "home_a";
                  return currentVenue;
                });
              }}
              disabled={!teamA || !teamB}
            >
              <ArrowRightLeft className="h-4 w-4" />
            </Button>
          </div>
          <TeamSearchSelect
            label="Team B"
            teams={teams}
            selectedTeam={teamB}
            otherTeam={teamA}
            onSelect={setTeamB}
            side="right"
          />
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_1.6fr]">
          <div>
            <label className="mb-1.5 block text-xs font-medium uppercase tracking-[0.24em] text-muted-foreground">
              Projection date
            </label>
            <div className="flex items-center gap-2 rounded-xl border border-border bg-card px-3 py-2.5">
              <CalendarDays className="h-4 w-4 text-muted-foreground" />
              <input
                data-testid="projection-date"
                type="date"
                value={date}
                onChange={(event) => setDate(event.target.value)}
                className="w-full bg-transparent text-sm outline-none"
              />
            </div>
          </div>
          <div>
            <label className="mb-1.5 block text-xs font-medium uppercase tracking-[0.24em] text-muted-foreground">
              Venue
            </label>
            <VenueSelector venue={venue} onChange={setVenue} teamA={teamA} teamB={teamB} />
          </div>
        </div>
      </Card>

      {matchupQuery.isLoading ? (
        <Card className="px-5 py-8 text-sm text-muted-foreground">Generating projection...</Card>
      ) : null}

      {prediction ? (
        <div className="grid gap-6 xl:grid-cols-[1.3fr_0.9fr]">
          <div className="space-y-6">
            <Card className="overflow-hidden border-primary/20">
              <div className="bg-gradient-to-r from-primary/10 via-primary/5 to-transparent px-6 py-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-primary">
                      <Sparkles className="h-3.5 w-3.5" />
                      Projected outcome
                    </div>
                    <div className="mt-3 text-3xl font-semibold tracking-tight">
                      {prediction.prediction.impliedSpread}
                    </div>
                    <div className="mt-2 text-sm text-muted-foreground">
                      Favorite: {favorite?.shortName} • uncertainty band{" "}
                      {formatSigned(prediction.prediction.uncertaintyBand.low)} to{" "}
                      {formatSigned(prediction.prediction.uncertaintyBand.high)}
                    </div>
                  </div>
                  <Badge
                    variant={
                      prediction.prediction.confidenceTier === "high"
                        ? "default"
                        : prediction.prediction.confidenceTier === "medium"
                          ? "secondary"
                          : "outline"
                    }
                    className="text-[10px] uppercase tracking-[0.18em]"
                  >
                    {prediction.prediction.confidenceTier} confidence
                  </Badge>
                </div>
              </div>

              <div className="space-y-5 px-6 py-5">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl bg-muted/45 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                      {prediction.teamA.shortName}
                    </div>
                    <div className="mt-2 text-4xl font-semibold tabular-nums">
                      {formatPercent(prediction.prediction.winProbabilityA)}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-muted/45 p-4 text-right">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                      {prediction.teamB.shortName}
                    </div>
                    <div className="mt-2 text-4xl font-semibold tabular-nums">
                      {formatPercent(prediction.prediction.winProbabilityB)}
                    </div>
                  </div>
                </div>

                <div className="overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-3 rounded-full bg-primary transition-all"
                    style={{ width: `${prediction.prediction.winProbabilityA * 100}%` }}
                  />
                </div>

                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl bg-muted/45 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                      Expected margin
                    </div>
                    <div className="mt-2 text-2xl font-semibold tabular-nums">
                      {formatSigned(prediction.prediction.expectedMargin)}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-muted/45 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                      Upset score
                    </div>
                    <div className="mt-2 text-2xl font-semibold tabular-nums">
                      {prediction.prediction.upsetScore.toFixed(1)}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-muted/45 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                      Model note
                    </div>
                    <div className="mt-2 text-sm leading-5 text-muted-foreground">
                      {confidenceCopy(prediction.prediction.confidenceTier)}
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-border bg-card px-4 py-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        Crowd diagnostics
                      </div>
                      <div className="mt-2 text-sm text-muted-foreground">
                        {prediction.prediction.diagnostics.observedSourceCount}/
                        {prediction.prediction.diagnostics.sourceCount} public signals observed • disagreement{" "}
                        {prediction.prediction.diagnostics.disagreementIndex.toFixed(2)}
                      </div>
                    </div>
                    <Badge variant="outline" className="text-[10px] uppercase tracking-[0.16em]">
                      {prediction.prediction.diagnostics.calibrationMethod}
                    </Badge>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="px-5 py-4">
              <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <Gauge className="h-3.5 w-3.5" />
                Rating snapshot
              </div>
              <div className="divide-y divide-border">
                <StatRow
                  label="AdjEM"
                  valueA={formatSigned(prediction.snapshotA.efficiencyMargin)}
                  valueB={formatSigned(prediction.snapshotB.efficiencyMargin)}
                  icon={<Zap className="h-3 w-3" />}
                />
                <StatRow
                  label="Adj Off"
                  valueA={prediction.snapshotA.offense.toFixed(1)}
                  valueB={prediction.snapshotB.offense.toFixed(1)}
                  icon={<TrendingUp className="h-3 w-3" />}
                />
                <StatRow
                  label="Adj Def"
                  valueA={prediction.snapshotA.defense.toFixed(1)}
                  valueB={prediction.snapshotB.defense.toFixed(1)}
                  icon={<Shield className="h-3 w-3" />}
                  inverse
                />
                <StatRow
                  label="Tempo"
                  valueA={prediction.snapshotA.tempo.toFixed(1)}
                  valueB={prediction.snapshotB.tempo.toFixed(1)}
                  icon={<Activity className="h-3 w-3" />}
                />
                <StatRow
                  label="Resume"
                  valueA={prediction.snapshotA.resumeScore.toFixed(1)}
                  valueB={prediction.snapshotB.resumeScore.toFixed(1)}
                  icon={<Layers3 className="h-3 w-3" />}
                />
                <StatRow
                  label="Recent"
                  valueA={prediction.snapshotA.recentForm.toFixed(1)}
                  valueB={prediction.snapshotB.recentForm.toFixed(1)}
                  icon={<Target className="h-3 w-3" />}
                />
              </div>
            </Card>
          </div>

          <div className="space-y-6">
            <Card className="px-5 py-4">
              <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <Layers3 className="h-3.5 w-3.5" />
                Crowd consensus breakdown
              </div>
              <div className="space-y-3">
                {prediction.prediction.components.map((component) => (
                  <div key={component.key} className="rounded-2xl bg-muted/45 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <div className="text-sm font-semibold">{component.name}</div>
                          <Badge variant="outline" className="text-[10px] uppercase tracking-[0.16em]">
                            {component.availability}
                          </Badge>
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          Weight {(component.weight * 100).toFixed(0)}% • {component.family.replace("_", " ")}
                        </div>
                        <div className="mt-2 max-w-sm text-xs leading-5 text-muted-foreground">
                          {component.description}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-semibold tabular-nums">
                          {formatSigned(component.expectedMargin)}
                        </div>
                        <div className="text-xs text-muted-foreground tabular-nums">
                          {formatPercent(component.winProbabilityA)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card className="px-5 py-4">
              <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <Info className="h-3.5 w-3.5" />
                Key factors
              </div>
              <div className="space-y-3">
                {prediction.prediction.featureAttributions.map((factor) => {
                  const favoredTeam =
                    factor.favoredTeamId === prediction.teamA.id ? prediction.teamA : prediction.teamB;
                  return (
                    <div key={factor.label} className="rounded-2xl border border-border bg-card p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-sm font-semibold">{factor.label}</div>
                        <Badge variant="outline" className="text-[10px] uppercase tracking-[0.16em]">
                          {favoredTeam.shortName}
                        </Badge>
                      </div>
                      <div className="mt-2 text-sm leading-6 text-muted-foreground">
                        {factor.description}
                      </div>
                      <div className="mt-2 text-xs font-medium tabular-nums text-primary">
                        Impact {formatSigned(factor.impact)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        </div>
      ) : null}

      {modelRun ? <ModelStatusCard modelRun={modelRun} /> : null}
    </div>
  );
}
