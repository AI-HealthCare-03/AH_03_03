import { useEffect, useState } from "react";
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

function Bars({ items }: { items: Record<string, unknown>[] }) {
  return (
    <div className="bars">
      {items.slice(0, 8).map((item, index) => {
        const value = Number(item.value ?? item.systolic ?? 0);
        const label =
          item.systolic || item.diastolic
            ? `${String(item.systolic ?? "-")}/${String(item.diastolic ?? "-")}`
            : value || "-";
        return (
          <div className="bar-row" key={`${String(item.date)}-${index}`}>
            <span>{String(item.date)}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${Math.min(value, 100)}%` }} />
            </div>
            <strong>{label}</strong>
          </div>
        );
      })}
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

type AnyRecord = Record<string, unknown>;

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
  const activeTrendItems = activeMetric.trendKeys.flatMap((key) => trends[key] ?? []);

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
                {!item.ready && <span className="badge badge-reference">준비 중</span>}
              </button>
            ))}
          </div>
        </Card>
        <Card title={activeMetric.label}>
          <p className="muted">{activeMetric.description}</p>
          {activeTrendItems.length > 0 ? (
            <Bars items={activeTrendItems} />
          ) : (
            <div className="state-box">표시할 추적 데이터가 없습니다.</div>
          )}
        </Card>
      </div>
      <div className="page-grid">
        <Card title="만성질환 위험도">
          {latestAnalysisResults.length > 0 ? (
            <div className="card-list">
              {latestAnalysisResults.map((result) => {
                const level = String(result.risk_level ?? "").toUpperCase();
                const score = getRiskScorePercent(result.risk_score);
                return (
                  <div className="mini-card" key={String(result.id)}>
                    <div className="button-row">
                      <strong>{analysisTypeLabels[result.analysis_type] ?? result.analysis_type}</strong>
                      <span className="badge badge-reference">{getModeLabel(result.analysis_mode)}</span>
                      <span className={`badge risk-${level.toLowerCase()}`}>{riskLevelLabels[level] ?? level}</span>
                    </div>
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: `${score}%` }} />
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
        <Card title="혈당 추이">
          <Bars items={trends.glucose ?? []} />
        </Card>
        <Card title="혈압 추이">
          <Bars items={trends.blood_pressure ?? []} />
        </Card>
        <Card title="AI 코멘트">
          <p>최근 입력된 건강 지표를 기준으로 혈당, 혈압, 체중 변화를 함께 확인해보세요.</p>
        </Card>
        <Card title="건강 팁">
          <p>식후 10분 걷기와 저녁 나트륨 줄이기는 혈당과 혈압 관리에 도움이 될 수 있습니다.</p>
        </Card>
        <Card title="식단 추천 결과 요약">
          <Bars items={trends.diet_score ?? []} />
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
