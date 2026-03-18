import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  LineChart,
  Line,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

interface PooledMetrics {
  logLoss: number;
  brierScore: number;
  marginMae: number;
  calibrationError: number;
  upsetRecall: number;
}

interface CalibrationResponse {
  calibrationBuckets: Record<string, Array<{ bucket: number; count: number; actualRate: number; probabilityRate: number }>>;
  pooledMetrics: Record<string, PooledMetrics>;
  bestCandidate: string | null;
}

async function fetchCalibration(): Promise<CalibrationResponse> {
  const res = await fetch("/api/model-runs/calibration");
  if (!res.ok) throw new Error("Failed to fetch calibration data");
  return res.json();
}

const MODEL_DISPLAY: Record<string, string> = {
  ensemble_blend: "Ensemble Blend",
  enhanced_candidate_stack: "Enhanced (Distilled)",
  gbm_distilled_candidate_stack: "GBM Distilled",
  direct_gbm: "Direct GBM",
  optimized_candidate_stack: "Optimized Linear",
  candidate_stack: "Full Candidate Stack",
  reduced_candidate_stack: "Reduced Stack",
  seed_only_logit: "Seed-Only Baseline",
  equal_weight_consensus: "Equal-Weight Consensus",
  best_single_source_power: "Best Single Source",
};

const CHART_MODELS = [
  "ensemble_blend",
  "enhanced_candidate_stack",
  "gbm_distilled_candidate_stack",
  "optimized_candidate_stack",
  "seed_only_logit",
  "equal_weight_consensus",
];

export default function BacktestExplorerPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["calibration"],
    queryFn: fetchCalibration,
  });

  if (isLoading || !data) {
    return (
      <div className="space-y-6">
        <section className="rounded-[28px] border border-primary/15 bg-[linear-gradient(135deg,rgba(15,23,42,0.04),rgba(124,58,237,0.12))] px-6 py-7">
          <Badge variant="secondary" className="text-[10px] uppercase tracking-[0.18em]">
            Model diagnostics
          </Badge>
          <h1 className="mt-3 text-3xl font-semibold tracking-tight">Backtest Explorer</h1>
          <p className="mt-3 text-sm text-muted-foreground">Loading...</p>
        </section>
      </div>
    );
  }

  const models = Object.entries(data.pooledMetrics)
    .filter(([name]) => CHART_MODELS.includes(name))
    .sort((a, b) => a[1].logLoss - b[1].logLoss);

  const barData = models.map(([name, metrics]) => ({
    model: MODEL_DISPLAY[name] ?? name,
    logLoss: Number(metrics.logLoss.toFixed(4)),
    brierScore: Number(metrics.brierScore.toFixed(4)),
  }));

  const allModels = Object.entries(data.pooledMetrics)
    .sort((a, b) => a[1].logLoss - b[1].logLoss);

  const bestModel = data.bestCandidate ?? allModels[0]?.[0] ?? "";
  const bestBuckets = data.calibrationBuckets[bestModel] ?? [];
  const calibrationData = bestBuckets.map((b) => ({
    predicted: Math.round(b.probabilityRate * 100),
    actual: Math.round(b.actualRate * 100),
    count: b.count,
  }));

  return (
    <div className="space-y-6">
      <section className="rounded-[28px] border border-primary/15 bg-[linear-gradient(135deg,rgba(15,23,42,0.04),rgba(124,58,237,0.12))] px-6 py-7">
        <Badge variant="secondary" className="text-[10px] uppercase tracking-[0.18em]">
          Model diagnostics
        </Badge>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight">Backtest Explorer</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
          Leave-one-season-out (LOSO) backtest results across 9 tournament seasons (2016-2025, excluding 2020).
          Pooled metrics evaluate each model on holdout seasons it never trained on.
        </p>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card className="px-5 py-4">
          <div className="mb-3 text-xs uppercase tracking-[0.18em] text-muted-foreground">
            Log Loss comparison (lower is better)
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} layout="vertical" margin={{ left: 120, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis type="number" domain={[0.54, 0.62]} tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="model" tick={{ fontSize: 11 }} width={110} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Bar dataKey="logLoss" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card className="px-5 py-4">
          <div className="mb-3 text-xs uppercase tracking-[0.18em] text-muted-foreground">
            Calibration reliability ({MODEL_DISPLAY[bestModel] ?? bestModel})
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={calibrationData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis dataKey="predicted" tick={{ fontSize: 10 }} label={{ value: "Predicted %", position: "insideBottom", offset: -5, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 10 }} label={{ value: "Actual %", angle: -90, position: "insideLeft", fontSize: 11 }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke="hsl(var(--muted-foreground))" strokeDasharray="5 5" strokeOpacity={0.4} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Line type="monotone" dataKey="actual" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 4 }} name="Actual win rate" />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card className="px-5 py-4">
        <div className="mb-3 text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Full model comparison (pooled LOSO holdout)
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-[0.12em] text-muted-foreground">
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 px-3 text-right">Log Loss</th>
                <th className="py-2 px-3 text-right">Brier</th>
                <th className="py-2 px-3 text-right">MAE</th>
                <th className="py-2 px-3 text-right">ECE</th>
                <th className="py-2 px-3 text-right">Upset Recall</th>
              </tr>
            </thead>
            <tbody>
              {allModels.map(([name, m], i) => (
                <tr key={name} className={`border-b border-border/50 ${name === bestModel ? "bg-primary/5 font-medium" : ""}`}>
                  <td className="py-2 pr-4">
                    <div className="flex items-center gap-2">
                      {MODEL_DISPLAY[name] ?? name}
                      {name === bestModel && (
                        <Badge variant="default" className="text-[9px] uppercase tracking-[0.16em]">best</Badge>
                      )}
                    </div>
                  </td>
                  <td className="py-2 px-3 text-right tabular-nums">{m.logLoss.toFixed(4)}</td>
                  <td className="py-2 px-3 text-right tabular-nums">{m.brierScore.toFixed(4)}</td>
                  <td className="py-2 px-3 text-right tabular-nums">{m.marginMae.toFixed(2)}</td>
                  <td className="py-2 px-3 text-right tabular-nums">{(m.calibrationError * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-right tabular-nums">{(m.upsetRecall * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
