import { Activity, Target, TrendingUp } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { ModelRun } from "@shared/schema";

interface ModelStatusCardProps {
  modelRun: ModelRun;
}

export function ModelStatusCard({ modelRun }: ModelStatusCardProps) {
  return (
    <Card className="border-primary/15 bg-card/95 px-5 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">{modelRun.label}</h3>
            <Badge variant="secondary" className="text-[10px] uppercase tracking-[0.16em]">
              {modelRun.scheduleSource.replace("_", " ")}
            </Badge>
          </div>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
            {modelRun.notes}
          </p>
        </div>
        <div className="grid min-w-[220px] grid-cols-2 gap-2 text-xs">
          <div className="rounded-xl bg-muted/50 p-3">
            <div className="text-muted-foreground">Log loss</div>
            <div className="mt-1 text-lg font-semibold tabular-nums">
              {modelRun.metrics.logLoss.toFixed(3)}
            </div>
          </div>
          <div className="rounded-xl bg-muted/50 p-3">
            <div className="text-muted-foreground">Margin MAE</div>
            <div className="mt-1 text-lg font-semibold tabular-nums">
              {modelRun.metrics.marginMae.toFixed(1)}
            </div>
          </div>
          <div className="rounded-xl bg-muted/50 p-3">
            <div className="flex items-center gap-1 text-muted-foreground">
              <TrendingUp className="h-3.5 w-3.5" />
              Upset recall
            </div>
            <div className="mt-1 text-lg font-semibold tabular-nums">
              {(modelRun.metrics.upsetRecall * 100).toFixed(1)}%
            </div>
          </div>
          <div className="rounded-xl bg-muted/50 p-3">
            <div className="flex items-center gap-1 text-muted-foreground">
              <Target className="h-3.5 w-3.5" />
              Calibration err.
            </div>
            <div className="mt-1 text-lg font-semibold tabular-nums">
              {modelRun.metrics.calibrationError.toFixed(3)}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
