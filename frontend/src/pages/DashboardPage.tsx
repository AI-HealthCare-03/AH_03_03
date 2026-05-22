import { type CSSProperties, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  DashboardAnalysisResult,
  DashboardRiskFactor,
  DashboardSummary,
  getDashboardChallenges,
  getDashboardDiets,
  getDashboardHealth,
  getDashboardMedications,
  getDashboardSummary,
  getDashboardTrends,
} from "../api/dashboard";
import { listExams, listMeasurements } from "../api/exams";
import Card from "../components/Card";

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

const diseaseChartColors: Record<string, string> = {
  HYPERTENSION: "var(--chart-hypertension)",
  DIABETES: "var(--chart-diabetes)",
  DYSLIPIDEMIA: "var(--chart-dyslipidemia)",
  OBESITY: "var(--chart-obesity)",
  OVERALL: "var(--chart-overall)",
};

function getModeLabel(mode: unknown): string {
  return String(mode ?? "BASIC").toUpperCase() === "PRECISION" ? "정밀" : "간편";
}

function getRiskScorePercent(value: unknown): number {
  const score = Number(value);
  if (!Number.isFinite(score)) {
    return 0;
  }
  return Math.round(score <= 1 ? score * 100 : score);
}

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
      <p>건강정보를 2회 이상 입력하거나 검진표 OCR을 추가하면 변화 추이를 확인할 수 있습니다.</p>
      <div className="button-row">
        <Link className="button secondary" to="/health">
          건강정보 입력하기
        </Link>
        <Link className="button secondary" to="/ocr/exam">
          검진표 OCR 추가하기
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

function getExamMeasurement(measurements: AnyRecord[], key: string): AnyRecord | null {
  return measurements.find((m) => String(m.measurement_key) === key) ?? null;
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<Partial<DashboardSummary>>({});
  const [healthSection, setHealthSection] = useState<DashboardData>({});
  const [challengeSection, setChallengeSection] = useState<DashboardData>({});
  const [dietSection, setDietSection] = useState<DashboardData>({});
  const [medicationSection, setMedicationSection] = useState<DashboardData>({});
  const [trends, setTrends] = useState<Record<string, Record<string, unknown>[]>>({});
  const [activeMetricKey, setActiveMetricKey] = useState<(typeof analysisMetrics)[number]["key"]>("blood");
  const [latestExam, setLatestExam] = useState<AnyRecord | null>(null);
  const [examMeasurements, setExamMeasurements] = useState<AnyRecord[]>([]);

  useEffect(() => {
    const load = async () => {
      const [summaryResult, trendsResult, healthResult, challengeResult, dietResult, medicationResult, examList] =
        await Promise.allSettled([
        getDashboardSummary<DashboardSummary>(),
        getDashboardTrends<Record<string, Record<string, unknown>[]>>("week"),
        getDashboardHealth<DashboardData>(),
        getDashboardChallenges<DashboardData>(),
        getDashboardDiets<DashboardData>(),
        getDashboardMedications<DashboardData>(),
        listExams<AnyRecord[]>({ limit: 1 }),
      ]);
      if (summaryResult.status === "fulfilled") setSummary(summaryResult.value);
      if (trendsResult.status === "fulfilled") setTrends(trendsResult.value);
      if (healthResult.status === "fulfilled") setHealthSection(healthResult.value);
      if (challengeResult.status === "fulfilled") setChallengeSection(challengeResult.value);
      if (dietResult.status === "fulfilled") setDietSection(dietResult.value);
      if (medicationResult.status === "fulfilled") setMedicationSection(medicationResult.value);
      const examRecords = examList.status === "fulfilled" && Array.isArray(examList.value) ? examList.value : [];
      const firstExam = examRecords[0] && typeof examRecords[0] === "object" ? (examRecords[0] as AnyRecord) : null;
      setLatestExam(firstExam);
      if (firstExam?.id) {
        try {
          const measurements = await listMeasurements(Number(firstExam.id));
          setExamMeasurements(measurements as unknown as AnyRecord[]);
        } catch {
          setExamMeasurements([]);
        }
      }
    };
    void load().catch(() => undefined);
  }, []);

  const latest = (healthSection.latest_health_record ?? summary.latest_health_record ?? {}) as HealthRecord;
  const latestAnalysisResults = Array.isArray(summary.latest_analysis_results)
    ? (summary.latest_analysis_results as DashboardAnalysisResult[])
    : [];
  const topRiskFactors = Array.isArray(summary.top_risk_factors)
    ? (summary.top_risk_factors as DashboardRiskFactor[])
    : [];
  const dashboardDiets = Array.isArray(dietSection.recent_diet_records)
    ? (dietSection.recent_diet_records as AnyRecord[])
    : [];
  const dashboardMedications = Array.isArray(medicationSection.active_medications)
    ? (medicationSection.active_medications as AnyRecord[])
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
  const riskTrendSeries = ["DIABETES", "HYPERTENSION", "DYSLIPIDEMIA", "OBESITY"].map((analysisType) => ({
    key: analysisType,
    label: analysisTypeLabels[analysisType],
    color: diseaseChartColors[analysisType],
    points: latestAnalysisResults
      .filter((result) => result.analysis_type === analysisType)
      .map((result) => {
        const value = getRiskScorePercent(result.risk_score);
        const date = result.analyzed_at ?? result.created_at;
        return {
          date,
          label: formatDate(date),
          value,
          displayValue: `${value}/100`,
        };
      })
      .sort((a, b) => a.date.localeCompare(b.date)),
  }));
  const metrics = [
    ["혈당", latest.fasting_glucose ?? "-"],
    ["혈압", `${String(latest.systolic_bp ?? "-")}/${String(latest.diastolic_bp ?? "-")}`],
    ["체중", latest.weight_kg ?? "-"],
    ["종합 위험도", summary.overall_risk_level ? riskLevelLabels[String(summary.overall_risk_level)] : "-"],
    ["챌린지 수행률", challengeRate],
    ["식단 점수", dietScore],
    ["복약/영양제", `${String(dashboardMedications.length || summary.active_medication_count || 0)}개`],
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
            : [lifestyleSeries[2]];

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>추적 대시보드</h1>
          <p>혈당, 혈압, 체중, 식단, 챌린지 변화를 한 번에 확인합니다.</p>
        </div>
      </div>
      <div className="metric-grid">
        {metrics.map(([label, value]) => (
          <div className="metric-card" key={String(label)}>
            <span>{String(label)}</span>
            <strong>{String(value)}</strong>
          </div>
        ))}
      </div>
      <div className="dashboard-grid">
        <Card title="분석 항목">
          <div className="card-list">
            {analysisMetrics.map((item) => (
              <button
                className={activeMetricKey === item.key ? "filter-tab active" : "filter-tab"}
                disabled={!item.ready}
                key={item.key}
                onClick={() => setActiveMetricKey(item.key)}
                type="button"
              >
                {item.label}
                {!item.ready && <span className="badge badge-reference">데이터 필요</span>}
              </button>
            ))}
          </div>
        </Card>
        <Card title={activeMetric.label}>
          <p className="muted">{activeMetric.description}</p>
          <LineChart series={activeMetricSeries} />
        </Card>
      </div>
      <div className="page-grid">
        <Card title="만성질환 위험도 추이">
          <p className="muted">질환별 위험도 분석 결과가 2회 이상 쌓이면 변화 추이를 확인할 수 있습니다.</p>
          <LineChart series={riskTrendSeries} />
        </Card>
        <Card title="검진/건강 수치 변화">
          <p className="muted">혈압, 공복혈당, 체중과 BMI 변화를 함께 확인합니다.</p>
          <LineChart series={[...bloodPressureSeries, ...glucoseSeries, ...weightSeries]} />
        </Card>
        <Card title="생활관리 점수 변화">
          <p className="muted">식단 점수, 챌린지 수행률, 복약 수행률을 추적합니다.</p>
          <LineChart series={lifestyleSeries} />
        </Card>
      </div>
      <div className="page-grid">
        <Card title="최근 만성질환 위험도">
          {latestAnalysisResults.length > 0 ? (
            <div className="card-list">
              {latestAnalysisResults.map((result) => {
                const level = String(result.risk_level ?? "").toUpperCase();
                const score = getRiskScorePercent(result.risk_score);
                const color = diseaseChartColors[result.analysis_type] ?? diseaseChartColors.OVERALL;
                return (
                  <div className="mini-card" key={String(result.id)}>
                    <div className="button-row">
                      <strong>{analysisTypeLabels[result.analysis_type] ?? result.analysis_type}</strong>
                      <span className="badge badge-reference">{getModeLabel(result.analysis_mode)}</span>
                      <span className={`badge risk-${level.toLowerCase()}`}>{riskLevelLabels[level] ?? level}</span>
                    </div>
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ background: color, width: `${score}%` }} />
                    </div>
                    <p className="muted">
                      {score}/100 · {formatDate(result.analyzed_at)}
                    </p>
                    {result.summary && <p>{result.summary}</p>}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="state-box">최근 분석 결과가 없습니다. 건강 분석을 실행하면 질환별 위험도가 표시됩니다.</div>
          )}
        </Card>
        <Card title="주요 위험요인">
          {topRiskFactors.length > 0 ? (
            <div className="card-list">
              {topRiskFactors.map((factor) => (
                <div className="mini-card" key={`${factor.analysis_result_id}-${factor.factor_key}`}>
                  <div className="button-row">
                    <strong>{factor.factor_name}</strong>
                    <span className="badge badge-reference">{analysisTypeLabels[factor.analysis_type] ?? factor.analysis_type}</span>
                    <span className="badge badge-reference">{getModeLabel(factor.analysis_mode)}</span>
                  </div>
                  <p className="muted">
                    {factor.factor_value ? `값: ${factor.factor_value}` : "기록된 값 없음"}
                    {factor.contribution_score !== null && factor.contribution_score !== undefined
                      ? ` · 영향도 ${getRiskScorePercent(factor.contribution_score)}/100`
                      : ""}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <div className="state-box">최근 분석 위험요인이 없습니다.</div>
          )}
        </Card>
      </div>
      <div className="page-grid">
        <Card title="AI 코멘트">
          <p>최근 입력된 건강 지표를 기준으로 혈당, 혈압, 체중 변화를 함께 확인해보세요.</p>
        </Card>
        <Card title="건강 팁">
          <p>식후 10분 걷기와 저녁 나트륨 줄이기는 혈당과 혈압 관리에 도움이 될 수 있습니다.</p>
        </Card>
        <Card title="추천 챌린지">
          {dashboardChallenges.length > 0 ? (
            <div className="card-list">
              {dashboardChallenges.slice(0, 3).map((challenge) => (
                <div className="mini-card" key={String(challenge.id)}>
                  <strong>{String((challenge.challenge as AnyRecord | undefined)?.title ?? "참여 중인 챌린지")}</strong>
                  <p className="muted">{String(challenge.status ?? "진행 중")}</p>
                </div>
              ))}
            </div>
          ) : (
            <p>식후 산책, 물 마시기, 혈압 기록하기 같은 기본 챌린지를 먼저 시작해보세요.</p>
          )}
        </Card>
        <Card title="복약 수행 요약">
          {dashboardMedicationRecords.length > 0 ? (
            <div className="card-list">
              {dashboardMedicationRecords.slice(0, 3).map((record) => (
                <div className="mini-card" key={String(record.id)}>
                  <strong>{record.is_taken ? "복용 완료" : "복용 대기"}</strong>
                  <p className="muted">{String(record.scheduled_at ?? record.created_at ?? "-")}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="state-box">최근 복약 기록이 없습니다.</div>
          )}
        </Card>
        <Card
          title="최근 검진표"
          actions={
            <Link className="button secondary" to="/ocr/exam">
              검진표 입력
            </Link>
          }
        >
          {latestExam ? (
            <div className="metric-grid">
              <div>
                <span>검진일</span>
                <strong>{formatDate(latestExam.exam_date ?? latestExam.uploaded_at)}</strong>
              </div>
              <div>
                <span>혈압</span>
                <strong>
                  {(() => {
                    const sys = getExamMeasurement(examMeasurements, "systolic_bp");
                    const dia = getExamMeasurement(examMeasurements, "diastolic_bp");
                    if (sys?.value && dia?.value) return `${String(sys.value)}/${String(dia.value)} mmHg`;
                    return "-";
                  })()}
                </strong>
              </div>
              <div>
                <span>공복혈당</span>
                <strong>
                  {(() => {
                    const m = getExamMeasurement(examMeasurements, "fasting_glucose");
                    return m?.value ? `${String(m.value)} mg/dL` : "-";
                  })()}
                </strong>
              </div>
              <div>
                <span>총콜레스테롤</span>
                <strong>
                  {(() => {
                    const m = getExamMeasurement(examMeasurements, "total_cholesterol");
                    return m?.value ? `${String(m.value)} mg/dL` : "-";
                  })()}
                </strong>
              </div>
            </div>
          ) : (
            <div className="state-box">등록된 검진표가 없습니다. 검진표를 업로드하면 주요 수치를 자동으로 정리합니다.</div>
          )}
        </Card>
      </div>
    </div>
  );
}
