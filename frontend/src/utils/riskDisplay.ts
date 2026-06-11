export type RiskDisplaySource = {
  risk_level?: unknown;
  risk_score?: unknown;
  service_band?: unknown;
  service_band_label?: unknown;
  service_band_percent?: unknown;
};

const serviceBandLabels: Record<string, string> = {
  LOW: "낮음",
  ATTENTION: "관심 필요",
  CAUTION: "주의",
  HIGH_CAUTION: "높은 주의",
};

const serviceBandPercents: Record<string, number> = {
  LOW: 25,
  ATTENTION: 45,
  CAUTION: 65,
  HIGH_CAUTION: 80,
};

const legacyRiskLabels: Record<string, string> = {
  LOW: "낮음",
  MEDIUM: "주의",
  HIGH: "높은 주의",
};

const legacyRiskPercents: Record<string, number> = {
  LOW: 25,
  MEDIUM: 55,
  HIGH: 80,
};

const riskColors: Record<string, string> = {
  LOW: "#1D9E75",
  ATTENTION: "#2563EB",
  CAUTION: "#F59E0B",
  HIGH_CAUTION: "#EF4444",
  MEDIUM: "#F59E0B",
  HIGH: "#EF4444",
};

function normalizeKey(value: unknown): string {
  return String(value ?? "").trim().toUpperCase();
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value)));
}

export function getDisplayRiskBand(result: RiskDisplaySource | null | undefined): string {
  const riskLevel = normalizeKey(result?.risk_level);
  if (riskLevel in serviceBandLabels || riskLevel in legacyRiskLabels) {
    return riskLevel;
  }
  const serviceBand = normalizeKey(result?.service_band);
  if (serviceBand in serviceBandLabels) {
    return serviceBand;
  }
  return riskLevel;
}

export function getDisplayRiskLabel(result: RiskDisplaySource | null | undefined): string {
  const riskLevel = normalizeKey(result?.risk_level);
  if (riskLevel in serviceBandLabels || riskLevel in legacyRiskLabels) {
    return serviceBandLabels[riskLevel] ?? legacyRiskLabels[riskLevel];
  }

  const explicitLabel = String(result?.service_band_label ?? "").trim();
  if (explicitLabel) {
    return explicitLabel;
  }

  const band = getDisplayRiskBand(result);
  return serviceBandLabels[band] ?? legacyRiskLabels[band] ?? (band || "-");
}

export function getDisplayRiskPercent(result: RiskDisplaySource | null | undefined): number {
  const riskLevel = normalizeKey(result?.risk_level);
  if (riskLevel in serviceBandPercents) {
    return serviceBandPercents[riskLevel];
  }
  if (riskLevel in legacyRiskPercents) {
    return legacyRiskPercents[riskLevel];
  }

  const explicitPercent = Number(result?.service_band_percent);
  if (Number.isFinite(explicitPercent)) {
    return clampPercent(explicitPercent);
  }

  const band = getDisplayRiskBand(result);
  if (band in serviceBandPercents) {
    return serviceBandPercents[band];
  }

  const score = Number(result?.risk_score);
  if (Number.isFinite(score)) {
    return clampPercent(score <= 1 ? score * 100 : score);
  }

  return legacyRiskPercents[band] ?? 0;
}

export function getRiskClassName(result: RiskDisplaySource | null | undefined): string {
  const band = getDisplayRiskBand(result).toLowerCase().replaceAll("_", "-");
  return band ? `risk-${band}` : "risk-low";
}

export function getRiskColor(result: RiskDisplaySource | null | undefined): string {
  return riskColors[getDisplayRiskBand(result)] ?? riskColors.LOW;
}

export function getDisplayRiskScoreLabel(result: RiskDisplaySource | null | undefined): string {
  return `${getDisplayRiskPercent(result)}/100`;
}
