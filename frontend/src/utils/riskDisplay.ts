export type RiskDisplaySource = {
  risk_level?: unknown;
  risk_score?: unknown;
  service_band?: unknown;
  service_band_label?: unknown;
  service_band_percent?: unknown;
};

export type AnalysisSourceDisplaySource = {
  result_source?: unknown;
  x2_available?: unknown;
  x2_stage_label?: unknown;
};

export type AnalysisResultLike = {
  analysis_type?: unknown;
  analysis_mode?: unknown;
  analyzedAt?: unknown;
  analyzed_at?: unknown;
  createdAt?: unknown;
  created_at?: unknown;
  date?: unknown;
  id?: unknown;
};

export type RiskStageKey = "LOW" | "ATTENTION" | "CAUTION" | "HIGH_CAUTION";
export type AnalysisDisplayMode = "BASIC" | "PRECISION";

export const riskStageOrder: RiskStageKey[] = ["LOW", "ATTENTION", "CAUTION", "HIGH_CAUTION"];
export const basicAnalysisTypes = ["HYPERTENSION", "DIABETES", "DYSLIPIDEMIA", "OBESITY"] as const;
export const precisionAnalysisTypes = [
  "HYPERTENSION",
  "DIABETES",
  "DYSLIPIDEMIA",
  "OBESITY",
  "ABDOMINAL_OBESITY",
  "FATTY_LIVER",
  "ANEMIA",
  "LIVER_FUNCTION",
  "KIDNEY_FUNCTION",
  "CHRONIC_KIDNEY_DISEASE",
] as const;

export const analysisTypeLabels: Record<string, string> = {
  HYPERTENSION: "고혈압",
  DIABETES: "당뇨",
  DYSLIPIDEMIA: "콜레스테롤·중성지방",
  OBESITY: "비만",
  ABDOMINAL_OBESITY: "복부비만",
  FATTY_LIVER: "지방간",
  ANEMIA: "빈혈",
  LIVER_FUNCTION: "간기능",
  KIDNEY_FUNCTION: "신장기능",
  CHRONIC_KIDNEY_DISEASE: "만성콩팥병",
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
  ATTENTION: "#D99A3D",
  CAUTION: "#EA580C",
  HIGH_CAUTION: "#EF4444",
  MEDIUM: "#EA580C",
  HIGH: "#EF4444",
};

const analysisSourceLabels: Record<string, string> = {
  X2_RULE: "검진 수치 반영",
  BASIC_FALLBACK: "간편분석 유지",
};

const x2UnavailableReasons: Record<string, string> = {
  HYPERTENSION: "혈압 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  DIABETES: "공복혈당 또는 당화혈색소 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  DYSLIPIDEMIA: "콜레스테롤·중성지방 관련 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  OBESITY: "BMI 또는 키/몸무게 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  ABDOMINAL_OBESITY: "허리둘레 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  FATTY_LIVER: "AST/ALT 등 간 관련 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  ANEMIA: "혈색소 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  LIVER_FUNCTION: "AST/ALT/감마GTP 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  KIDNEY_FUNCTION: "요단백/크레아티닌/eGFR 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
  CHRONIC_KIDNEY_DISEASE: "eGFR 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.",
};

export type AnalysisDisplaySlot<T extends AnalysisResultLike> = {
  analysisType: string;
  diseaseName: string;
  isUnavailable: boolean;
  result: T | null;
  unavailableReason: string | null;
};

function normalizeKey(value: unknown): string {
  return String(value ?? "").trim().toUpperCase();
}

function getItemTimestamp(item: AnalysisResultLike): number {
  const candidates = [item.analyzedAt, item.analyzed_at, item.createdAt, item.created_at, item.date];
  for (const candidate of candidates) {
    const parsed = Date.parse(String(candidate ?? ""));
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
}

function getItemId(item: AnalysisResultLike): number {
  const parsed = Number(item.id);
  return Number.isFinite(parsed) ? parsed : 0;
}

function isNewerAnalysisItem<T extends AnalysisResultLike>(
  candidate: T,
  candidateIndex: number,
  current: T,
  currentIndex: number,
): boolean {
  const candidateTimestamp = getItemTimestamp(candidate);
  const currentTimestamp = getItemTimestamp(current);
  if (candidateTimestamp !== currentTimestamp) {
    return candidateTimestamp > currentTimestamp;
  }
  const candidateId = getItemId(candidate);
  const currentId = getItemId(current);
  if (candidateId !== currentId) {
    return candidateId > currentId;
  }
  return candidateIndex > currentIndex;
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value)));
}

export function getAnalysisTypeLabel(value: unknown, fallback = "질환"): string {
  const key = normalizeKey(value);
  return analysisTypeLabels[key] ?? fallback;
}

export function isKnownAnalysisType(value: unknown): boolean {
  return normalizeKey(value) in analysisTypeLabels;
}

export function getLatestResultsByAnalysisType<T extends AnalysisResultLike>(items: T[]): T[] {
  const latestByType = new Map<string, { index: number; item: T }>();
  items.forEach((item, index) => {
    const key = normalizeKey(item.analysis_type);
    if (!key || !(key in analysisTypeLabels)) {
      return;
    }
    const current = latestByType.get(key);
    if (!current || isNewerAnalysisItem(item, index, current.item, current.index)) {
      latestByType.set(key, { index, item });
    }
  });
  return Array.from(latestByType.values()).map(({ item }) => item);
}

export function getExpectedAnalysisTypesByMode(mode: unknown): string[] {
  return normalizeKey(mode) === "PRECISION" ? [...precisionAnalysisTypes] : [...basicAnalysisTypes];
}

export function getLatestAnalysisMode<T extends AnalysisResultLike>(
  items: T[],
  fallback: AnalysisDisplayMode = "BASIC",
): AnalysisDisplayMode {
  const latest = items.reduce<{ index: number; item: T } | null>((current, item, index) => {
    if (!current || isNewerAnalysisItem(item, index, current.item, current.index)) {
      return { index, item };
    }
    return current;
  }, null);
  return normalizeKey(latest?.item.analysis_mode) === "PRECISION" ? "PRECISION" : fallback;
}

export function getUnavailableAnalysisReason(analysisType: unknown, mode: unknown): string {
  const key = normalizeKey(analysisType);
  if (normalizeKey(mode) !== "PRECISION") {
    return "이번 간편분석 결과에 포함되지 않았습니다.";
  }
  return x2UnavailableReasons[key] ?? "필요한 검진 수치가 없어 이번 정밀분석에서 판정하지 않았습니다.";
}

export function mergeResultsWithExpectedAnalysisTypes<T extends AnalysisResultLike>(
  items: T[],
  mode: unknown,
): AnalysisDisplaySlot<T>[] {
  const latestResults = getLatestResultsByAnalysisType(items);
  const latestByType = new Map(latestResults.map((item) => [normalizeKey(item.analysis_type), item]));
  return getExpectedAnalysisTypesByMode(mode).map((analysisType) => {
    const result = latestByType.get(analysisType) ?? null;
    return {
      analysisType,
      diseaseName: getAnalysisTypeLabel(analysisType),
      isUnavailable: result === null,
      result,
      unavailableReason: result === null ? getUnavailableAnalysisReason(analysisType, mode) : null,
    };
  });
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

export function getCanonicalRiskStage(result: RiskDisplaySource | null | undefined): RiskStageKey {
  const band = getDisplayRiskBand(result);
  if (band === "ATTENTION" || band === "CAUTION" || band === "HIGH_CAUTION") {
    return band;
  }
  if (band === "MEDIUM") {
    return "CAUTION";
  }
  if (band === "HIGH") {
    return "HIGH_CAUTION";
  }
  const explicitPercent = Number(result?.service_band_percent);
  const score = Number(result?.risk_score);
  const percent = Number.isFinite(explicitPercent) ? explicitPercent : score <= 1 ? score * 100 : score;
  if (Number.isFinite(percent)) {
    if (percent >= 75) {
      return "HIGH_CAUTION";
    }
    if (percent >= 60) {
      return "CAUTION";
    }
    if (percent >= 40) {
      return "ATTENTION";
    }
  }
  return "LOW";
}

export function getRiskStageLabel(stage: RiskStageKey): string {
  return serviceBandLabels[stage];
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

export function getAnalysisSourceBadgeLabel(result: AnalysisSourceDisplaySource | null | undefined): string | null {
  const source = normalizeKey(result?.result_source);
  return analysisSourceLabels[source] ?? null;
}

export function getX2StageSummary(result: AnalysisSourceDisplaySource | null | undefined): string | null {
  const source = normalizeKey(result?.result_source);
  const stageLabel = String(result?.x2_stage_label ?? "").trim();
  if (source === "X2_RULE" && stageLabel) {
    return `검진 수치 기준: ${stageLabel}`;
  }
  if (source === "BASIC_FALLBACK" && result?.x2_available === false) {
    return "검진 수치가 부족해 간편분석 결과를 유지했습니다.";
  }
  return null;
}
