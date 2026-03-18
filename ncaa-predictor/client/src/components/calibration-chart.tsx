import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card } from "@/components/ui/card";

interface CalibrationBucket {
  bucket: number;
  count: number;
  actualRate: number;
  probabilityRate: number;
}

interface CalibrationResponse {
  calibrationBuckets: Record<string, CalibrationBucket[]>;
  pooledMetrics: Record<string, { logLoss: number; brierScore: number; calibrationError: number }>;
  bestCandidate: string | null;
}

async function fetchCalibration(): Promise<CalibrationResponse> {
  const res = await fetch("/api/model-runs/calibration");
  if (!res.ok) throw new Error("Failed to fetch calibration data");
  return res.json();
}

export function CalibrationChart() {
  const { data, isLoading } = useQuery({
    queryKey: ["calibration"],
    queryFn: fetchCalibration,
  });

  if (isLoading || !data) {
    return null;
  }

  const bestCandidate = data.bestCandidate ?? "enhanced_candidate_stack";
  const buckets = data.calibrationBuckets[bestCandidate];
  if (!buckets || buckets.length === 0) {
    return null;
  }

  const chartData = buckets.map((b) => ({
    predicted: Math.round(b.probabilityRate * 100),
    actual: Math.round(b.actualRate * 100),
    count: b.count,
  }));

  const metrics = data.pooledMetrics[bestCandidate];

  return (
    <Card className="px-5 py-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Calibration reliability diagram
        </div>
        {metrics && (
          <div className="flex gap-3 text-xs text-muted-foreground">
            <span>ECE {(metrics.calibrationError * 100).toFixed(1)}%</span>
            <span>Brier {metrics.brierScore.toFixed(3)}</span>
          </div>
        )}
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis
            dataKey="predicted"
            label={{ value: "Predicted %", position: "insideBottom", offset: -5, fontSize: 11 }}
            tick={{ fontSize: 10 }}
          />
          <YAxis
            label={{ value: "Actual %", angle: -90, position: "insideLeft", fontSize: 11 }}
            tick={{ fontSize: 10 }}
          />
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
            stroke="hsl(var(--muted-foreground))"
            strokeDasharray="5 5"
            strokeOpacity={0.4}
          />
          <Tooltip
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
            formatter={(value: number, name: string) => [
              `${value}%`,
              name === "actual" ? "Actual win rate" : "Predicted probability",
            ]}
            labelFormatter={(label) => `Predicted: ${label}%`}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={{ r: 4, fill: "hsl(var(--primary))" }}
            name="Actual win rate"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-2 text-center text-xs text-muted-foreground">
        Closer to the diagonal = better calibrated. Based on pooled LOSO holdout predictions.
      </div>
    </Card>
  );
}
