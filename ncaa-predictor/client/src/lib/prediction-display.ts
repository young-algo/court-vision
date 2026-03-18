import type { MatchupOdds, Prediction, Team } from "@shared/schema";

export function formatSigned(value: number, digits = 1) {
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function formatPercent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Minimal shape needed by matchupInsight — satisfied by both MatchupPredictionResponse and GameProjection. */
interface InsightInput {
  teamA: Pick<Team, "shortName">;
  teamB: Pick<Team, "shortName">;
  prediction: Prediction;
  odds: MatchupOdds | null;
}

/**
 * Build a short, matchup-specific insight sentence from the prediction data.
 * Picks the single most interesting observation rather than a generic confidence blurb.
 */
export function matchupInsight(pred: InsightInput): string {
  const p = pred.prediction;
  const comps = p.components;
  const favName = p.expectedMargin >= 0 ? pred.teamA.shortName : pred.teamB.shortName;
  const dogName = p.expectedMargin >= 0 ? pred.teamB.shortName : pred.teamA.shortName;
  const absMargin = Math.abs(p.expectedMargin);
  const disagreement = p.diagnostics.disagreementIndex;

  // Find the component that deviates most from the consensus margin
  const margins = comps.map((c) => c.expectedMargin);
  const avgMargin = margins.reduce((a, b) => a + b, 0) / margins.length;
  const outlier = comps.reduce((best, c) =>
    Math.abs(c.expectedMargin - avgMargin) > Math.abs(best.expectedMargin - avgMargin) ? c : best
  );
  const outlierDelta = outlier.expectedMargin - avgMargin;

  // Check if betting odds exist and disagree with the model
  const odds = pred.odds;
  if (odds && odds.spreadA !== null) {
    const modelSpread = -p.expectedMargin; // negative = teamA favored
    const oddsSpread = odds.spreadA;
    const lineDiff = Math.abs(modelSpread - oddsSpread);
    if (lineDiff >= 3) {
      const side = oddsSpread < modelSpread ? favName : dogName;
      return `Market line differs from model by ${lineDiff.toFixed(1)} points — books are higher on ${side}.`;
    }
  }

  // High disagreement among sources
  if (disagreement > 4) {
    return `Sources disagree sharply (σ ${disagreement.toFixed(1)}). ${outlier.name.split(" ")[0]} is the outlier at ${formatSigned(outlier.expectedMargin)}.`;
  }

  // Very tight game — essentially a toss-up
  if (absMargin < 1.5) {
    const strongest = comps.reduce((a, b) => Math.abs(a.expectedMargin) > Math.abs(b.expectedMargin) ? a : b);
    return `True toss-up. The only edge comes from ${strongest.name.split(" ")[0]} (${formatSigned(strongest.expectedMargin)}).`;
  }

  // One source is a clear outlier pulling the line
  if (Math.abs(outlierDelta) > 4) {
    const dir = outlierDelta > 0 ? `bullish on ${pred.teamA.shortName}` : `bullish on ${pred.teamB.shortName}`;
    return `${outlier.name.split(" ")[0]} is the outlier — ${dir} at ${formatSigned(outlier.expectedMargin)} vs consensus ${formatSigned(avgMargin)}.`;
  }

  // All sources agree, large margin — dominant favorite
  if (disagreement < 2 && absMargin >= 10) {
    return `All sources align on ${favName} by double digits. Low variance, high confidence pick.`;
  }

  // Sources agree, moderate margin
  if (disagreement < 2.5) {
    const best = comps.reduce((a, b) => Math.abs(a.expectedMargin) > Math.abs(b.expectedMargin) ? a : b);
    return `Tight source agreement. ${best.name.split(" ")[0]} gives ${favName} the strongest edge at ${formatSigned(best.expectedMargin)}.`;
  }

  // Default: highlight the key driver
  const topAttrib = p.featureAttributions[0];
  if (topAttrib) {
    return `${topAttrib.label}: ${topAttrib.description}`;
  }

  return `${favName} projects as a ${absMargin.toFixed(0)}-point favorite with moderate source agreement.`;
}
