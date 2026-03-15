import type { ConfidenceTier } from "@shared/schema";

export function formatSigned(value: number, digits = 1) {
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function formatPercent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function confidenceCopy(confidence: ConfidenceTier) {
  if (confidence === "high") {
    return "Model agreement is tight and the projected edge is meaningful.";
  }

  if (confidence === "medium") {
    return "The edge is real, but model disagreement still leaves upset room.";
  }

  return "This projects as volatile; treat the favorite as vulnerable.";
}
