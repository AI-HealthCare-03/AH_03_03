import { type CSSProperties, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  DashboardAnalysisResult,
  DashboardSummary,
  getDashboardChallenges,
  getDashboardDiets,
  getDashboardHealth,
  getDashboardMedications,
  getDashboardSummary,
  getDashboardTrends,
} from "../api/dashboard";

type DashboardData = Record<string, unknown>;
type HealthRecord = Record<string, unknown>;
type TrendItem = Record<string, unknown>;
type AnyRecord = Record<string, unknown>;

type ChartPoint = {
  date: string;
  label: string;
  value: number;
  displayValue?: string;
};

type ChartSeries = {
  key: string;
  label: string;
  color: string;
  points: ChartPoint[];
};

const challengeStatusLabels: Record<string, string> = {
  ACTIVE: "진행 중",
  IN_PROGRESS: "진행 중",
  JOINED: "진행 중",
  COMPLETED: "완료",
  GIVEN_UP: "포기",
  CANCELED: "포기",
  CANCELLED: "포기",
  PENDING: "대기",
};

const diseaseChartColors: Record<string, string> = {
  HYPERTENSION: "var(--chart-hypertension)",
  DIABETES: "var(--chart-diabetes)",
  DYSLIPIDEMIA: "var(--chart-dyslipidemia)",
  OBESITY: "var(--chart-obesity)",
};

const analysisTypeLabels: Record<string, string> = {
  DIABETES: "당뇨",
  HYPERTENSION: "고혈압",
  DYSLIPIDEMIA: "이상지질혈증",
  OBESITY: "비만",
};

const riskLevelLabels: Record<string, string> = {
  LOW: "낮음",
  MEDIUM: "관리 필요",
  HIGH: "높음",
};

const periodOptions = [
  { label: "1주일", value: "week" },
  { label: "1개월", value: "month" },
  { label: "3개월", value: "quarter" },
  { label: "6개월", value: "year" },
  { label: "전체", value: "all" },
];

function latestValue(items: Record<string, unknown>[] | undefined, fallback = "-"): string {
  const item = items?.[0];
  if (!item) {
    return fallback;
  }
  const value = item.value ?? item.systolic;
  return value === undefined || value === null ? fallback : String(value);
}

function averageValue(items: Record<string, unknown>[] | undefined): string {
  const values = (items ?? []).map((item) => Number(item.value)).filter((value) => Number.isFinite(value));
  if (values.length === 0) {
    return "-";
  }
  return `${Math.round(values.reduce((sum, value) => sum + value, 0) / values.length)}%`;
}

function getTrendValue(item: TrendItem, key = "value"): number | null {
  const value = Number(item[key]);
  return Number.isFinite(value) ? value : null;
}

function toChartPoint(item: TrendItem, value: number, suffix = ""): ChartPoint {
  const date = String(item.date ?? item.created_at ?? item.measured_at ?? "");
  const label = date ? formatDate(date) : "-";
  return {
    date,
    label,
    value,
    displayValue: `${Math.round(value * 10) / 10}${suffix}`,
  };
}

function makeValueSeries(
  items: TrendItem[] | undefined,
  label: string,
  color: string,
  options: { key?: string; suffix?: string } = {},
): ChartSeries {
  const key = options.key ?? "value";
  const suffix = options.suffix ?? "";
  return {
    key: label,
    label,
    color,
    points: (items ?? [])
      .map((item) => {
        const value = getTrendValue(item, key);
        return value === null ? null : toChartPoint(item, value, suffix);
      })
      .filter((point): point is ChartPoint => Boolean(point))
      .sort((a, b) => a.date.localeCompare(b.date)),
  };
}

function makeMedicationAdherenceSeries(records: AnyRecord[]): ChartSeries {
  const grouped = new Map<string, { total: number; taken: number }>();
  records.forEach((record) => {
    const rawDate = String(record.scheduled_at ?? record.taken_at ?? record.created_at ?? "");
    if (!rawDate) return;
    const date = rawDate.slice(0, 10);
    const current = grouped.get(date) ?? { total: 0, taken: 0 };
    current.total += 1;
    if (record.is_taken) current.taken += 1;
    grouped.set(date, current);
  });
  return {
    key: "복약 수행률",
    label: "복약 수행률",
    color: "var(--chart-overall)",
    points: Array.from(grouped.entries())
      .map(([date, value]) => ({
        date,
        label: formatDate(date),
        value: value.total > 0 ? Math.round((value.taken / value.total) * 100) : 0,
        displayValue: `${value.taken}/${value.total}회`,
      }))
      .sort((a, b) => a.date.localeCompare(b.date)),
  };
}

function EmptyChartState() {
  return (
    <div className="empty-state chart-empty-state">
      <strong>아직 추적할 데이터가 충분하지 않습니다.</strong>
      <p>건강정보를 2회 이상 입력하거나 검진표를 등록하면 변화 추이를 확인할 수 있습니다.</p>
      <div className="button-row">
        <Link className="button secondary" to="/health">
          건강정보 입력하기
        </Link>
        <Link className="button secondary" to="/ocr/exam">
          검진표 등록하기
        </Link>
      </div>
    </div>
  );
}

function LineChart({ series }: { series: ChartSeries[] }) {
  const visibleSeries = series
    .map((item) => ({ ...item, points: item.points.filter((point) => Number.isFinite(point.value)) }))
    .filter((item) => item.points.length > 0);
  const allPoints = visibleSeries.flatMap((item) => item.points);
  const uniqueDates = Array.from(new Set(allPoints.map((point) => point.date || point.label))).sort();
  const hasEnoughData = uniqueDates.length >= 2 || visibleSeries.some((item) => item.points.length >= 2);

  if (!hasEnoughData) {
    return <EmptyChartState />;
  }

  const values = allPoints.map((point) => point.value);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const rangePadding = rawMin === rawMax ? Math.max(rawMax * 0.1, 1) : (rawMax - rawMin) * 0.12;
  const min = Math.max(0, rawMin - rangePadding);
  const max = rawMax + rangePadding;
  const width = 640;
  const height = 240;
  const padding = { top: 20, right: 24, bottom: 34, left: 44 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const xFor = (date: string, fallbackIndex: number) => {
    const index = uniqueDates.indexOf(date);
    const safeIndex = index >= 0 ? index : fallbackIndex;
    return padding.left + (uniqueDates.length <= 1 ? innerWidth / 2 : (safeIndex / (uniqueDates.length - 1)) * innerWidth);
  };
  const yFor = (value: number) => padding.top + (1 - (value - min) / Math.max(max - min, 1)) * innerHeight;
  const gridValues = [max, (max + min) / 2, min];

  return (
    <div className="line-chart-card">
      <div className="line-chart-wrap">
        <svg aria-label="추적 꺾은선 그래프" className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img">
          {gridValues.map((value) => {
            const y = yFor(value);
            return (
              <g key={value}>
                <line className="line-chart-grid" x1={padding.left} x2={width - padding.right} y1={y} y2={y} />
                <text className="line-chart-axis" x={8} y={y + 4}>
                  {Math.round(value)}
                </text>
              </g>
            );
          })}
          {uniqueDates.length > 0 && (
            <>
              <text className="line-chart-axis" x={padding.left} y={height - 8}>
                {formatDate(uniqueDates[0])}
              </text>
              <text className="line-chart-axis" textAnchor="end" x={width - padding.right} y={height - 8}>
                {formatDate(uniqueDates[uniqueDates.length - 1])}
              </text>
            </>
          )}
          {visibleSeries.map((item) => {
            const d = item.points
              .map((point, index) => `${index === 0 ? "M" : "L"} ${xFor(point.date || point.label, index)} ${yFor(point.value)}`)
              .join(" ");
            return (
              <g key={item.key}>
                <path className="line-chart-path" d={d} style={{ "--series-color": item.color } as CSSProperties} />
                {item.points.map((point, index) => (
                  <circle
                    className="line-chart-point"
                    cx={xFor(point.date || point.label, index)}
                    cy={yFor(point.value)}
                    key={`${item.key}-${point.date}-${index}`}
                    r="3.5"
                    style={{ "--series-color": item.color } as CSSProperties}
                  >
                    <title>
                      {item.label} · {point.label} · {point.displayValue ?? Math.round(point.value)}
                    </title>
                  </circle>
                ))}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="line-chart-legend">
        {visibleSeries.map((item) => (
          <span key={item.key}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function toNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatCompactValue(value: unknown, fallback = "아직 기록 없음"): string {
  if (value === undefined || value === null || value === "") {
    return fallback;
  }
  return String(value);
}

function formatRiskLevel(value: unknown): string {
  const key = String(value ?? "").toUpperCase();
  return riskLevelLabels[key] ?? (key || "-");
}

function formatModelLabel(result: DashboardAnalysisResult): string | null {
  const modelName = result.model_name ? String(result.model_name) : "";
  const modelVersion = result.model_version ? String(result.model_version) : "";
  if (!modelName && !modelVersion) {
    return null;
  }
  if (modelName.toLowerCase() === "catboost") {
    return modelVersion ? `CatBoost · ${modelVersion}` : "CatBoost";
  }
  if (modelName.toLowerCase() === "rule_based") {
    return modelVersion ? `rule_based · ${modelVersion}` : "rule_based";
  }
  return modelVersion ? `${modelName} · ${modelVersion}` : modelName || modelVersion;
}

function formatRiskScore(value: unknown): string {
  const score = Number(value);
  if (!Number.isFinite(score)) {
    return "";
  }
  return `${Math.round(score <= 1 ? score * 100 : score)}/100`;
}

function getSeriesDelta(series: ChartSeries[]): number | null {
  const points = series.flatMap((item) => item.points).filter((point) => Number.isFinite(point.value));
  if (points.length < 2) {
    return null;
  }
  const sorted = [...points].sort((a, b) => a.date.localeCompare(b.date));
  return Math.round((sorted[sorted.length - 1].value - sorted[sorted.length - 2].value) * 10) / 10;
}

function MiniSparkline({ series }: { series: ChartSeries[] }) {
  const points = series
    .flatMap((item) => item.points)
    .filter((point) => Number.isFinite(point.value))
    .slice(-6);
  if (points.length < 2) {
    return <div className="metric-sparkline-empty">기록 없음</div>;
  }
  const values = points.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1);
  const d = points
    .map((point, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * 92 + 4;
      const y = 38 - ((point.value - min) / range) * 28;
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
  return (
    <svg aria-hidden="true" className="metric-sparkline" viewBox="0 0 100 44">
      <path d={d} />
    </svg>
  );
}

function MiniGauge({ value }: { value: number | null }) {
  const percent = Math.max(0, Math.min(value ?? 0, 100));
  return (
    <div className="metric-gauge" style={{ "--gauge-value": `${percent}%` } as CSSProperties}>
      <span>{value === null ? "-" : `${Math.round(percent)}%`}</span>
    </div>
  );
}

function DashboardSummaryCard({
  label,
  value,
  unit,
  delta,
  series,
  gauge,
}: {
  label: string;
  value: string;
  unit?: string;
  delta?: number | null;
  series?: ChartSeries[];
  gauge?: number | null;
}) {
  const hasDelta = delta !== undefined && delta !== null && Number.isFinite(delta);
  return (
    <article className="metric-summary-card">
      <div className="metric-summary-card__head">
        <span>{label}</span>
        {hasDelta ? (
          <em className={delta > 0 ? "up" : delta < 0 ? "down" : ""}>
            {delta > 0 ? "▲" : delta < 0 ? "▼" : "━"} {Math.abs(delta)}
          </em>
        ) : (
          <em>변화 기록 없음</em>
        )}
      </div>
      <strong>
        {value}
        {unit ? <small>{unit}</small> : null}
      </strong>
      {gauge !== undefined ? <MiniGauge value={gauge} /> : <MiniSparkline series={series ?? []} />}
    </article>
  );
}

const analysisMetrics = [
  {
    key: "blood",
    label: "혈당/혈압 변화",
    description: "최근 입력된 건강 기록을 기준으로 혈당과 혈압 변화를 확인합니다.",
    trendKeys: ["glucose", "blood_pressure"],
    ready: true,
  },
  {
    key: "weight",
    label: "체중/BMI 변화",
    description: "최근 체중과 BMI 변화를 확인합니다.",
    trendKeys: ["weight", "bmi"],
    ready: true,
  },
  {
    key: "exercise",
    label: "운동 수행률",
    description: "걷기와 근력운동 입력값을 기준으로 활동 상태를 확인합니다.",
    trendKeys: ["exercise"],
    ready: true,
  },
  {
    key: "diet",
    label: "식단 점수 변화",
    description: "최근 식단 기록의 점수 변화를 확인합니다.",
    trendKeys: ["diet_score"],
    ready: true,
  },
  {
    key: "medication",
    label: "복약/영양제 수행률",
    description: "등록된 복약/영양제 기록을 기준으로 수행 상태를 확인합니다.",
    trendKeys: ["medication"],
    ready: true,
  },
  {
    key: "sleep",
    label: "수면 변화",
    description: "수면 기록을 2회 이상 입력하면 변화 추이를 확인할 수 있습니다.",
    trendKeys: ["sleep"],
    ready: false,
  },
  {
    key: "water",
    label: "수분 섭취 수행률",
    description: "수분 섭취 기록이 쌓이면 수행률을 확인할 수 있습니다.",
    trendKeys: ["water"],
    ready: false,
  },
] as const;

function formatDate(value: unknown): string {
  if (!value) return "-";
  const d = new Date(String(value));
  if (Number.isNaN(d.getTime())) return "-";
  const y = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}.${mo}.${day}`;
}

function getChallengeIcon(category: unknown): string {
  const key = String(category ?? "").toUpperCase();
  if (key.includes("DIET") || key.includes("식")) return "🥗";
  if (key.includes("MEDICATION") || key.includes("복약")) return "💊";
  if (key.includes("WALK") || key.includes("EXERCISE") || key.includes("운동")) return "🚶";
  if (key.includes("WATER") || key.includes("수분")) return "💧";
  return "✅";
}

function getChallengeTitle(challenge: AnyRecord): string {
  const nested = challenge.challenge as AnyRecord | undefined;
  return String(nested?.title ?? challenge.title ?? "추천 챌린지");
}

function getChallengeDuration(challenge: AnyRecord): string {
  const nested = challenge.challenge as AnyRecord | undefined;
  const days = nested?.duration_days ?? challenge.duration_days;
  return days ? `${String(days)}일` : "기간 확인";
}

function getChallengeProgress(challenge: AnyRecord): number {
  const value = Number(challenge.progress ?? challenge.progress_rate ?? 0);
  return Number.isFinite(value) ? Math.max(0, Math.min(value, 100)) : 0;
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<Partial<DashboardSummary>>({});
  const [healthSection, setHealthSection] = useState<DashboardData>({});
  const [challengeSection, setChallengeSection] = useState<DashboardData>({});
  const [dietSection, setDietSection] = useState<DashboardData>({});
  const [medicationSection, setMedicationSection] = useState<DashboardData>({});
  const [trends, setTrends] = useState<Record<string, Record<string, unknown>[]>>({});
  const [activeMetricKey, setActiveMetricKey] = useState<(typeof analysisMetrics)[number]["key"]>("blood");
  const [selectedPeriod, setSelectedPeriod] = useState("week");

  useEffect(() => {
    const load = async () => {
      const [summaryResult, trendsResult, healthResult, challengeResult, dietResult, medicationResult] =
        await Promise.allSettled([
          getDashboardSummary<DashboardSummary>(),
          getDashboardTrends<Record<string, Record<string, unknown>[]>>(selectedPeriod),
          getDashboardHealth<DashboardData>(),
          getDashboardChallenges<DashboardData>(),
          getDashboardDiets<DashboardData>(),
          getDashboardMedications<DashboardData>(),
        ]);
      if (summaryResult.status === "fulfilled") setSummary(summaryResult.value);
      if (trendsResult.status === "fulfilled") setTrends(trendsResult.value);
      if (healthResult.status === "fulfilled") setHealthSection(healthResult.value);
      if (challengeResult.status === "fulfilled") setChallengeSection(challengeResult.value);
      if (dietResult.status === "fulfilled") setDietSection(dietResult.value);
      if (medicationResult.status === "fulfilled") setMedicationSection(medicationResult.value);
    };
    void load().catch(() => undefined);
  }, [selectedPeriod]);

  const latest = (healthSection.latest_health_record ?? summary.latest_health_record ?? {}) as HealthRecord;
  const latestAnalysisResults = Array.isArray(summary.latest_analysis_results)
    ? (summary.latest_analysis_results as DashboardAnalysisResult[])
    : [];
  const diseaseAnalysisResults = latestAnalysisResults.filter((result) =>
    Boolean(analysisTypeLabels[String(result.analysis_type)]),
  );
  const dashboardDiets = Array.isArray(dietSection.recent_diet_records)
    ? (dietSection.recent_diet_records as AnyRecord[])
    : [];
  const dashboardMedicationRecords = Array.isArray(medicationSection.recent_medication_records)
    ? (medicationSection.recent_medication_records as AnyRecord[])
    : [];
  const dashboardChallenges = Array.isArray(challengeSection.user_challenges)
    ? (challengeSection.user_challenges as AnyRecord[])
    : [];
  const challengeRate = averageValue(trends.challenge_completion_rate);
  const dietScore = dashboardDiets[0]?.diet_score ? String(dashboardDiets[0].diet_score) : latestValue(trends.diet_score);
  const bloodPressureSeries = [
    makeValueSeries(trends.blood_pressure, "수축기 혈압", diseaseChartColors.HYPERTENSION, {
      key: "systolic",
      suffix: " mmHg",
    }),
    makeValueSeries(trends.blood_pressure, "이완기 혈압", "var(--chart-hypertension-soft-line)", {
      key: "diastolic",
      suffix: " mmHg",
    }),
  ];
  const glucoseSeries = [
    makeValueSeries(trends.glucose, "공복혈당", diseaseChartColors.DIABETES, { suffix: " mg/dL" }),
  ];
  const weightSeries = [
    makeValueSeries(trends.weight, "체중", diseaseChartColors.OBESITY, { suffix: " kg" }),
    makeValueSeries(trends.bmi, "BMI", "var(--chart-obesity-soft-line)"),
  ];
  const lifestyleSeries = [
    makeValueSeries(trends.diet_score, "식단 점수", diseaseChartColors.DIABETES, { suffix: "점" }),
    makeValueSeries(trends.challenge_completion_rate, "챌린지 수행률", diseaseChartColors.OBESITY, { suffix: "%" }),
    makeMedicationAdherenceSeries(dashboardMedicationRecords),
  ];
  const medicationAdherenceSeries = makeMedicationAdherenceSeries(dashboardMedicationRecords);
  const medicationAdherence = medicationAdherenceSeries.points.at(-1)?.value ?? null;
  const sleepValue = toNumber(latest.sleep_hours);
  const summaryCards = [
    {
      label: "혈당",
      value: formatCompactValue(latest.fasting_glucose),
      unit: latest.fasting_glucose ? "mg/dL" : "",
      delta: getSeriesDelta(glucoseSeries),
      series: glucoseSeries,
    },
    {
      label: "혈압",
      value:
        latest.systolic_bp || latest.diastolic_bp
          ? `${String(latest.systolic_bp ?? "-")}/${String(latest.diastolic_bp ?? "-")}`
          : "아직 기록 없음",
      unit: latest.systolic_bp || latest.diastolic_bp ? "mmHg" : "",
      delta: getSeriesDelta(bloodPressureSeries.slice(0, 1)),
      series: bloodPressureSeries,
    },
    {
      label: "체중",
      value: formatCompactValue(latest.weight_kg),
      unit: latest.weight_kg ? "kg" : "",
      delta: getSeriesDelta(weightSeries.slice(0, 1)),
      series: weightSeries,
    },
    {
      label: "챌린지 수행률",
      value: challengeRate,
      delta: getSeriesDelta([lifestyleSeries[1]]),
      gauge: toNumber(challengeRate.replace("%", "")),
    },
    {
      label: "식단 점수",
      value: dietScore === "-" ? "아직 기록 없음" : `${dietScore}점`,
      delta: getSeriesDelta([lifestyleSeries[0]]),
      gauge: toNumber(dietScore),
    },
    {
      label: "복약/영양제 수행률",
      value: medicationAdherence === null ? "아직 기록 없음" : `${Math.round(medicationAdherence)}%`,
      delta: getSeriesDelta([medicationAdherenceSeries]),
      gauge: medicationAdherence,
    },
    {
      label: "수면 시간",
      value: sleepValue === null ? "아직 기록 없음" : String(sleepValue),
      unit: sleepValue === null ? "" : "hr",
      delta: null,
      gauge: sleepValue === null ? null : Math.min(Math.round((sleepValue / 8) * 100), 100),
    },
  ];
  const activeMetric = analysisMetrics.find((metric) => metric.key === activeMetricKey) ?? analysisMetrics[0];
  const activeMetricSeries =
    activeMetricKey === "blood"
      ? [...glucoseSeries, ...bloodPressureSeries]
      : activeMetricKey === "weight"
        ? weightSeries
        : activeMetricKey === "diet"
          ? [lifestyleSeries[0]]
          : activeMetricKey === "exercise"
            ? [lifestyleSeries[1]]
            : activeMetricKey === "medication"
              ? [medicationAdherenceSeries]
              : [];
  const activeChartCards =
    activeMetricKey === "blood"
      ? [
          {
            title: "혈당",
            unit: "mg/dL",
            value: formatCompactValue(latest.fasting_glucose, "-"),
            delta: getSeriesDelta(glucoseSeries),
            series: glucoseSeries,
          },
          {
            title: "혈압",
            unit: "mmHg",
            value:
              latest.systolic_bp || latest.diastolic_bp
                ? `${String(latest.systolic_bp ?? "-")}/${String(latest.diastolic_bp ?? "-")}`
                : "-",
            delta: getSeriesDelta(bloodPressureSeries.slice(0, 1)),
            series: bloodPressureSeries,
          },
        ]
      : activeMetricKey === "weight"
        ? [
            {
              title: "체중 / BMI",
              unit: "",
              value: latest.weight_kg ? `${String(latest.weight_kg)}kg` : "-",
              delta: getSeriesDelta(weightSeries.slice(0, 1)),
              series: weightSeries,
            },
          ]
        : activeMetricKey === "diet"
          ? [
              {
                title: "식단 점수",
                unit: "점",
                value: dietScore === "-" ? "-" : dietScore,
                delta: getSeriesDelta([lifestyleSeries[0]]),
                series: [lifestyleSeries[0]],
              },
            ]
          : activeMetricKey === "exercise"
            ? [
                {
                  title: "챌린지 수행률",
                  unit: "%",
                  value: challengeRate,
                  delta: getSeriesDelta([lifestyleSeries[1]]),
                  series: [lifestyleSeries[1]],
                },
              ]
            : activeMetricKey === "medication"
              ? [
                  {
                    title: "복약 수행률",
                    unit: "%",
                    value: medicationAdherence === null ? "-" : String(Math.round(medicationAdherence)),
                    delta: getSeriesDelta([medicationAdherenceSeries]),
                    series: [medicationAdherenceSeries],
                  },
                ]
              : [
                  {
                    title: activeMetric.label,
                    unit: "",
                    value: "-",
                    delta: null,
                    series: activeMetricSeries,
                  },
                ];
  const aiComment =
    String((summary as AnyRecord).ai_comment ?? "").trim() ||
    (latestAnalysisResults.length > 0
      ? "최근 분석 결과와 건강 지표 변화를 함께 보면서 생활습관을 꾸준히 조정해보세요."
      : "아직 분석 결과가 없습니다. 건강정보를 입력하고 간편 분석을 실행하면 맞춤 코멘트를 확인할 수 있습니다.");
  const latestDiet = dashboardDiets[0];
  const dietSummary = String(
    latestDiet?.summary ??
      latestDiet?.analysis_summary ??
      latestDiet?.recommendation ??
      latestDiet?.description ??
      "",
  ).trim();
  const dietPoints = [
    latestDiet?.recommended_points,
    latestDiet?.warning_points,
    latestDiet?.recommendation,
    dietSummary,
  ]
    .filter((item) => typeof item === "string" && item.trim().length > 0)
    .map((item) => String(item).trim())
    .slice(0, 3);
  const recommendedChallenges = dashboardChallenges.slice(0, 4);

  return (
    <div className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <h1>추적 대시보드</h1>
          <p>혈당, 혈압, 체중, 식단, 챌린지 변화를 한 번에 확인합니다.</p>
        </div>
        <div className="dashboard-period-tabs" aria-label="조회 기간">
          {periodOptions.map((period) => (
            <button
              className={selectedPeriod === period.value ? "active" : ""}
              key={period.value}
              onClick={() => setSelectedPeriod(period.value)}
              type="button"
            >
              {period.label}
            </button>
          ))}
        </div>
      </header>
      <div className="metric-summary-grid">
        {summaryCards.map((item) => (
          <DashboardSummaryCard
            delta={item.delta}
            gauge={item.gauge}
            key={item.label}
            label={item.label}
            series={item.series}
            unit={item.unit}
            value={item.value}
          />
        ))}
      </div>
      <section className="dashboard-analysis-layout">
        <aside className="analysis-tab-list">
          <h2>분석 항목 선택</h2>
          <div>
            {analysisMetrics.map((item) => (
              <button
                className={`analysis-tab-button${activeMetricKey === item.key ? " active" : ""}`}
                disabled={!item.ready}
                key={item.key}
                onClick={() => setActiveMetricKey(item.key)}
                type="button"
              >
                <span>{item.label}</span>
                {!item.ready ? <em>데이터 필요</em> : null}
              </button>
            ))}
          </div>
        </aside>
        <div className="dashboard-analysis-main">
          <section className="analysis-chart-card">
            <div className="dashboard-chart-heading">
              <div>
                <h2>{activeMetric.label} 요약</h2>
                <p>{activeMetric.description}</p>
              </div>
              <span>{formatDate(new Date())}</span>
            </div>
            <div className="dashboard-chart-panels">
              {activeChartCards.map((item) => (
                <article className="dashboard-chart-panel" key={item.title}>
                  <div className="dashboard-chart-panel-head">
                    <strong>{item.title}</strong>
                    <span>{item.unit}</span>
                  </div>
                  <LineChart series={item.series} />
                  <div className="dashboard-chart-stat">
                    <span>현재</span>
                    <strong>
                      {item.value}
                      {item.unit && item.value !== "-" ? <small>{item.unit}</small> : null}
                    </strong>
                    {item.delta !== null && item.delta !== undefined ? <em>{item.delta > 0 ? "▲" : item.delta < 0 ? "▼" : "━"} {Math.abs(item.delta)}</em> : null}
                  </div>
                </article>
              ))}
            </div>
          </section>
          <div className="dashboard-helper-grid">
            <section className="card ai-comment-card">
              <div className="card-header">
                <h2>AI 코멘트</h2>
              </div>
              <p>{aiComment}</p>
              {latestAnalysisResults.length === 0 ? (
                <Link className="button secondary compact-button" to="/analysis">
                  간편 분석 실행하기
                </Link>
              ) : null}
            </section>
            <section className="card">
              <div className="card-header">
                <h2>최근 분석 결과</h2>
              </div>
              {diseaseAnalysisResults.length > 0 ? (
                <div className="card-list">
                  {diseaseAnalysisResults.map((result) => {
                    const model = formatModelLabel(result);
                    return (
                      <div className="mini-card" key={result.id}>
                        <div className="record-row">
                          <div>
                            <strong>{analysisTypeLabels[String(result.analysis_type)]}</strong>
                            <p className="muted">
                              {result.analysis_mode === "PRECISION" ? "정밀 분석" : "간편 분석"}
                              {model ? ` · ${model}` : ""}
                            </p>
                          </div>
                          <span className={`badge risk-${String(result.risk_level ?? "").toLowerCase()}`}>
                            {formatRiskLevel(result.risk_level)}
                          </span>
                        </div>
                        <span className="badge badge-reference">{formatRiskScore(result.risk_score)}</span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="empty-state">
                  <strong>최근 분석 결과가 없습니다.</strong>
                  <p>건강정보를 입력하고 분석을 실행하면 질환별 결과가 표시됩니다.</p>
                  <Link className="button secondary compact-button" to="/analysis">
                    분석 실행하기
                  </Link>
                </div>
              )}
            </section>
            <section className="card health-tip-card">
              <div className="card-header">
                <h2>건강 팁</h2>
              </div>
              <ul className="dashboard-tip-list">
                <li>식후 10분 걷기처럼 부담이 적은 습관을 먼저 반복해보세요.</li>
                <li>아침 건강 지표를 같은 시간대에 기록하면 변화 흐름을 보기 좋습니다.</li>
                <li>혈압이나 혈당 수치가 평소와 다르면 무리하지 말고 필요한 경우 의료진과 상담하세요.</li>
              </ul>
            </section>
          </div>
        </div>
      </section>
      <section className="card diet-summary-section">
        <div className="card-header">
          <div>
            <h2>식단 추천 결과 요약</h2>
            <p>최근 식단 기록을 기준으로 식단 점수와 추천 포인트를 확인합니다.</p>
          </div>
          <div className="card-actions">
            <Link className="button secondary compact-button" to="/diets">
              식단 업로드하기
            </Link>
          </div>
        </div>
        {latestDiet ? (
          <div className="dashboard-diet-summary">
            <div className="dashboard-diet-score">
              <MiniGauge value={toNumber(latestDiet.diet_score)} />
              <strong>{latestDiet.diet_score ? `${String(latestDiet.diet_score)}점` : "점수 미산정"}</strong>
            </div>
            <div className="dashboard-diet-copy">
              <span>추천 식단 요약</span>
              <p>{dietSummary || "최근 식단 기록을 기준으로 식사 구성을 점검해보세요."}</p>
            </div>
            <div className="dashboard-diet-points">
              <span>추천 포인트</span>
              {dietPoints.length > 0 ? (
                <ul>
                  {dietPoints.map((point) => (
                    <li key={point}>{point}</li>
                  ))}
                </ul>
              ) : (
                <p>식단을 더 기록하면 추천 포인트를 확인할 수 있습니다.</p>
              )}
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <strong>아직 식단 분석 결과가 없습니다.</strong>
            <p>식단 이미지를 업로드하면 식단 점수와 추천 포인트를 확인할 수 있습니다.</p>
            <Link className="button secondary compact-button" to="/diets">
              식단 업로드하기
            </Link>
          </div>
        )}
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <h2>추천 챌린지</h2>
            <p>최근 참여 현황을 바탕으로 이어가기 좋은 챌린지를 확인합니다.</p>
          </div>
          <div className="card-actions">
            <Link className="button secondary compact-button" to="/challenges">
              전체 보기
            </Link>
          </div>
        </div>
        {recommendedChallenges.length > 0 ? (
          <div className="challenge-compact-grid">
            {recommendedChallenges.map((challenge) => {
              const nested = challenge.challenge as AnyRecord | undefined;
              const status = String(challenge.status ?? "").toUpperCase();
              const progress = getChallengeProgress(challenge);
              return (
                <article className="challenge-compact-card" key={String(challenge.id)}>
                  <div className="challenge-compact-icon">{getChallengeIcon(nested?.category ?? challenge.category)}</div>
                  <div>
                    <strong>{getChallengeTitle(challenge)}</strong>
                    <p>
                      {getChallengeDuration(challenge)} ·{" "}
                      {challengeStatusLabels[status] ?? (status ? status : "참여 가능")}
                    </p>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${progress}%` }} />
                  </div>
                  <div className="challenge-compact-footer">
                    <span>{Math.round(progress)}%</span>
                    <Link className="button secondary compact-button" to={`/challenges/${String(challenge.challenge_id ?? nested?.id ?? "")}`}>
                      {status === "ACTIVE" || status === "IN_PROGRESS" || status === "JOINED" ? "진행중" : "참여하기"}
                    </Link>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <div className="empty-state">
            <strong>추천할 챌린지 기록이 아직 없습니다.</strong>
            <p>관심 있는 챌린지에 참여하면 진행률과 추천 흐름을 이곳에서 확인할 수 있습니다.</p>
            <Link className="button secondary compact-button" to="/challenges">
              챌린지 둘러보기
            </Link>
          </div>
        )}
      </section>
    </div>
  );
}
