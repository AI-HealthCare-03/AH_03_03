import { type CSSProperties, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { getLatestAnalysisResults } from "../api/analysis";
import { listChallengeRecommendations, listChallenges, listMyChallenges } from "../api/challenges";
import { listExams, listMeasurements } from "../api/exams";
import { getAnalysisReadiness, getLatestHealthRecord } from "../api/health";
import { getMainSummary } from "../api/main";
import { listMedications } from "../api/medications";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

type MainData = Record<string, unknown>;
type AnyRecord = Record<string, unknown>;

const publicFallback = {
  service_title: "HealthCare",
  service_description: "AI 기반 건강 분석, 검진표 OCR, 식단 분석, 챌린지 기능을 한 곳에서 제공합니다.",
};

const landingFeatures = [
  {
    icon: "🧭",
    title: "AI 위험도 분석",
    description: "건강정보를 기반으로 당뇨, 고혈압, 이상지질혈증 위험도를 확인합니다.",
    redirect: "/analysis",
  },
  {
    icon: "📄",
    title: "검진표 OCR 입력",
    description: "검진표 이미지나 PDF에서 주요 건강 수치를 빠르게 입력합니다.",
    redirect: "/ocr/exam",
  },
  {
    icon: "🥗",
    title: "식단 이미지 분석",
    description: "식단 사진을 기록하고 영양 요약과 개선 포인트를 확인합니다.",
    redirect: "/diets",
  },
  {
    icon: "🚶",
    title: "맞춤 챌린지",
    description: "위험도와 생활습관에 맞춘 작은 건강 습관을 실천합니다.",
    redirect: "/challenges",
  },
  {
    icon: "💊",
    title: "복약/영양제 관리",
    description: "복약 정보와 기록을 한 곳에서 관리합니다.",
    redirect: "/medications",
  },
  {
    icon: "💬",
    title: "AI 건강 상담",
    description: "건강 분석, 식단, 운동, 복약 관련 질문을 편하게 남깁니다.",
    redirect: "/chatbot",
  },
];

const landingPersonas = [
  {
    id: "exam",
    icon: "📋",
    title: "검진 결과가 걱정되는 직장인",
    quote: "검진표는 받았는데 수치가 뭘 의미하는지 모르겠어요.",
    features: ["검진표 OCR", "위험도 분석", "AI 코멘트"],
    flow: [
      { icon: "📄", title: "검진표 업로드", description: "촬영하거나 파일로 올립니다." },
      { icon: "📊", title: "위험도 분석", description: "주요 질환 위험도를 확인합니다." },
      { icon: "✅", title: "챌린지 추천", description: "생활습관 액션을 이어갑니다." },
    ],
  },
  {
    id: "habit",
    icon: "🚶",
    title: "생활습관을 바꾸고 싶은 사용자",
    quote: "혈당, 혈압, 체중 관리를 위해 식단과 운동 습관을 만들고 싶어요.",
    features: ["식단 분석", "챌린지", "추적 대시보드"],
    flow: [
      { icon: "🥗", title: "식단 기록", description: "사진과 기본 정보를 입력합니다." },
      { icon: "📈", title: "변화 추적", description: "건강 점수와 추이를 봅니다." },
      { icon: "🚶", title: "챌린지 실천", description: "추천 습관을 시작합니다." },
    ],
  },
  {
    id: "record",
    icon: "💊",
    title: "복약/건강기록을 관리하는 사용자",
    quote: "복약, 영양제, 건강기록을 놓치지 않고 관리하고 싶어요.",
    features: ["복약 OCR", "복약 기록", "알림", "AI 상담"],
    flow: [
      { icon: "💊", title: "복약 입력", description: "처방전과 약봉투를 정리합니다." },
      { icon: "🔔", title: "알림 확인", description: "기록과 알림을 관리합니다." },
      { icon: "🤖", title: "AI 상담", description: "궁금한 점을 이어서 묻습니다." },
    ],
  },
];

const serviceFlow = [
  { icon: "📄", label: "검진표 OCR" },
  { icon: "📊", label: "위험도 분석" },
  { icon: "🤖", label: "AI 상담" },
  { icon: "✅", label: "맞춤 챌린지" },
];

const previewMetrics = [
  ["오늘의 건강 점수", "82점"],
  ["당뇨 위험도", "관리 필요"],
  ["고혈압 위험도", "낮음"],
  ["추천 챌린지", "식후 10분 걷기"],
  ["최근 검진일", "2026.05.20"],
];

const landingPreview = {
  healthScore: 82,
  diabetesRiskScore: 58,
  hypertensionRiskScore: 32,
  challengeProgress: 72,
};

const riskLabelMap: Record<string, string> = {
  LOW: "낮음",
  MEDIUM: "관리 필요",
  HIGH: "높음",
};

const analysisTypeLabels: Record<string, string> = {
  DIABETES: "당뇨",
  HYPERTENSION: "고혈압",
  DYSLIPIDEMIA: "이상지질혈증",
  OBESITY: "비만",
};

const categoryLabel: Record<string, string> = {
  BLOOD_PRESSURE: "혈압 관리",
  BLOOD_SUGAR: "혈당 관리",
  BLOOD_GLUCOSE: "혈당 관리",
  DIET: "식단",
  EXERCISE: "운동",
  MEDICATION: "복약",
  HABIT: "생활습관",
  WATER: "수분섭취",
  SLEEP: "수면",
  COMMON: "공통",
  WEIGHT: "체중 관리",
};

function asRecord(value: unknown): AnyRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as AnyRecord) : {};
}

function formatRisk(value: unknown): string {
  const raw = String(value ?? "").toUpperCase();
  return riskLabelMap[raw] ?? (raw || "-");
}

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

function formatOptional(value: unknown, suffix = ""): string {
  if (value === undefined || value === null || value === "") {
    return "아직 기록 없음";
  }
  return `${String(value)}${suffix}`;
}

function formatBloodPressure(record: AnyRecord): string {
  if (!record.systolic_bp && !record.diastolic_bp) {
    return "아직 기록 없음";
  }
  return `${String(record.systolic_bp ?? "-")}/${String(record.diastolic_bp ?? "-")} mmHg`;
}

function getCategoryLabel(value: unknown): string {
  const category = String(value ?? "COMMON").toUpperCase();
  return categoryLabel[category] ?? "공통";
}

function getChallengeTitle(challenge: AnyRecord): string {
  const nested = asRecord(challenge.challenge);
  return String(challenge.title ?? nested.title ?? "추천 챌린지");
}

function getChallengeDuration(challenge: AnyRecord): string {
  const nested = asRecord(challenge.challenge);
  const duration = challenge.duration_days ?? nested.duration_days;
  return duration ? `${String(duration)}일` : "기간 안내";
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(Math.round(value), 100));
}

function toPercentStyle(value: number): CSSProperties {
  return { width: `${clampPercent(value)}%` };
}

function toGaugeStyle(value: number): CSSProperties {
  return { "--gauge-value": `${clampPercent(value)}%` } as CSSProperties;
}

function getChallengeProgress(challenge: AnyRecord): number {
  const explicit = Number(challenge.progress ?? challenge.progress_rate ?? challenge.completion_rate);
  if (Number.isFinite(explicit)) {
    return clampPercent(explicit <= 1 ? explicit * 100 : explicit);
  }
  const completedDays = Number(challenge.completed_days ?? challenge.completed_count);
  const durationDays = Number(challenge.duration_days ?? asRecord(challenge.challenge).duration_days);
  if (Number.isFinite(completedDays) && Number.isFinite(durationDays) && durationDays > 0) {
    return clampPercent((completedDays / durationDays) * 100);
  }
  const status = String(challenge.status ?? "").toUpperCase();
  if (status === "COMPLETED") return 100;
  if (["ACTIVE", "IN_PROGRESS", "JOINED"].includes(status)) return 40;
  return 0;
}

function getAveragePercent(values: number[]): number {
  const valid = values.filter((value) => Number.isFinite(value));
  if (valid.length === 0) {
    return 0;
  }
  return clampPercent(valid.reduce((sum, value) => sum + value, 0) / valid.length);
}

export default function MainPage() {
  const { backendUser, isAuthenticated } = useAuth();
  const [selectedPersonaId, setSelectedPersonaId] = useState(landingPersonas[0].id);
  const [data, setData] = useState<MainData>(publicFallback);
  const [analysisResults, setAnalysisResults] = useState<AnyRecord[]>([]);
  const [latestHealthRecord, setLatestHealthRecord] = useState<AnyRecord>({});
  const [readiness, setReadiness] = useState<AnyRecord>({});
  const [challenges, setChallenges] = useState<AnyRecord[]>([]);
  const [challengeRecommendations, setChallengeRecommendations] = useState<AnyRecord[]>([]);
  const [myChallenges, setMyChallenges] = useState<AnyRecord[]>([]);
  const [medications, setMedications] = useState<AnyRecord[]>([]);
  const [sectionError, setSectionError] = useState("");
  const [latestExam, setLatestExam] = useState<AnyRecord | null>(null);
  const [examMeasurements, setExamMeasurements] = useState<AnyRecord[]>([]);

  useEffect(() => {
    const load = async () => {
      if (isAuthenticated) {
        const [
          summary,
          latestResults,
          latestHealth,
          healthReadiness,
          challengeList,
          recommendationList,
          myChallengeList,
          medicationList,
          examList,
        ] = await Promise.allSettled([
            getMainSummary<MainData>(),
            getLatestAnalysisResults<AnyRecord[]>(),
            getLatestHealthRecord<AnyRecord>(),
            getAnalysisReadiness<AnyRecord>(),
            listChallenges<AnyRecord[]>({ limit: 20 }),
            listChallengeRecommendations<AnyRecord[]>({ limit: 3 }),
            listMyChallenges<AnyRecord[]>({ limit: 3 }),
            listMedications<AnyRecord[]>(),
            listExams<AnyRecord[]>({ limit: 1 }),
          ]);

        setData(summary.status === "fulfilled" ? summary.value : publicFallback);
        setAnalysisResults(latestResults.status === "fulfilled" && Array.isArray(latestResults.value) ? latestResults.value : []);
        setLatestHealthRecord(latestHealth.status === "fulfilled" ? asRecord(latestHealth.value) : {});
        setReadiness(healthReadiness.status === "fulfilled" ? asRecord(healthReadiness.value) : {});
        setChallenges(challengeList.status === "fulfilled" && Array.isArray(challengeList.value) ? challengeList.value : []);
        setChallengeRecommendations(
          recommendationList.status === "fulfilled" && Array.isArray(recommendationList.value)
            ? recommendationList.value
            : [],
        );
        setMyChallenges(
          myChallengeList.status === "fulfilled" && Array.isArray(myChallengeList.value) ? myChallengeList.value : [],
        );
        setMedications(medicationList.status === "fulfilled" && Array.isArray(medicationList.value) ? medicationList.value : []);
        setSectionError(summary.status === "rejected" ? "일부 요약 정보를 불러오지 못했습니다. 각 기능은 바로 이용할 수 있습니다." : "");

        const examRecords = examList.status === "fulfilled" && Array.isArray(examList.value) ? examList.value : [];
        const firstExam = examRecords[0] ? asRecord(examRecords[0]) : null;
        setLatestExam(firstExam);
        if (firstExam?.id) {
          try {
            const measurements = await listMeasurements(Number(firstExam.id));
            setExamMeasurements(measurements as unknown as AnyRecord[]);
          } catch {
            setExamMeasurements([]);
          }
        }
        return;
      }

      try {
        setData(publicFallback);
        setAnalysisResults([]);
        setLatestHealthRecord({});
        setReadiness({});
        setChallenges([]);
        setChallengeRecommendations([]);
        setMyChallenges([]);
        setMedications([]);
        setSectionError("");
      } catch {
        setData(publicFallback);
        setAnalysisResults([]);
      }
    };
    void load();
  }, [isAuthenticated]);

  if (isAuthenticated) {
    const latestHealth = Object.keys(latestHealthRecord).length > 0 ? latestHealthRecord : asRecord(data.latest_health_summary);
    const latestAnalysis = asRecord(data.latest_analysis_summary);
    const dashboardSummary = asRecord(data.dashboard_summary);
    const recentRecords = asRecord(data.recent_records);
    const recentDietRecords = Array.isArray(recentRecords.diet_records) ? (recentRecords.diet_records as AnyRecord[]) : [];
    const latestDietScore = recentDietRecords[0]?.diet_score ?? null;
    const effectiveAnalysisResults =
      analysisResults.length > 0 ? analysisResults : latestAnalysis.analysis_type ? [latestAnalysis] : [];
    const hasAnalysisResults = effectiveAnalysisResults.length > 0;
    const displayAnalysisResults = effectiveAnalysisResults.filter((result) =>
      Boolean(analysisTypeLabels[String(result.analysis_type ?? "")]),
    );
    const basicReady = readiness.basic_ready ?? readiness.is_ready;
    const todayCards = [
      {
        title: basicReady === false ? "기본 건강정보 입력" : "건강정보 확인",
        description:
          basicReady === false
            ? "기본 건강정보를 입력하면 건강 위험도 분석을 실행할 수 있습니다."
            : "저장된 건강정보를 확인하고 필요한 항목을 보완해보세요.",
        buttonLabel: "건강정보 입력하기",
        to: "/health",
      },
      {
        title: effectiveAnalysisResults.length > 0 ? "최근 분석 결과 확인" : "건강 위험도 분석",
        description:
          effectiveAnalysisResults.length > 0
            ? "최근 분석 결과를 확인하고 다음 관리 행동을 정리해보세요."
            : "건강정보 입력 후 위험도 분석을 실행해보세요.",
        buttonLabel: effectiveAnalysisResults.length > 0 ? "분석 결과 보기" : "분석하러 가기",
        to: effectiveAnalysisResults.length > 0 ? "/analysis/history" : "/analysis",
      },
      {
        title: challenges.length > 0 || myChallenges.length > 0 ? "추천 챌린지 시작" : "챌린지 둘러보기",
        description:
          myChallenges.length > 0
            ? "진행 중인 챌린지를 이어가고 오늘 실천을 기록해보세요."
            : "추천 챌린지를 시작하고 작은 건강 습관을 만들어보세요.",
        buttonLabel: "챌린지 보기",
        to: "/challenges",
      },
      {
        title: medications.length > 0 ? "오늘 복약 기록 확인" : "복약/영양제 등록",
        description:
          medications.length > 0
            ? "오늘 복약 기록을 확인하고 필요한 메모를 남겨보세요."
            : "복약 정보를 등록하면 건강 관리 흐름을 함께 확인할 수 있습니다.",
        buttonLabel: "복약 관리",
        to: "/medications",
      },
    ];
    // Extract safe numeric values from health records
    const glucoseVal = latestHealthRecord.fasting_glucose != null ? Number(latestHealthRecord.fasting_glucose) : null;
    const systolicVal = latestHealthRecord.systolic_bp != null ? Number(latestHealthRecord.systolic_bp) : null;
    const diastolicVal = latestHealthRecord.diastolic_bp != null ? Number(latestHealthRecord.diastolic_bp) : null;
    const weightVal = latestHealthRecord.weight_kg != null ? Number(latestHealthRecord.weight_kg) : null;
    const bmiVal = latestHealthRecord.bmi != null ? Number(latestHealthRecord.bmi) : null;

    // Diet score bars (last 5, oldest → newest left → right)
    const dietScoreBars = recentDietRecords
      .slice(0, 5)
      .reverse()
      .map((r) => Number(r.diet_score ?? 0));

    const challengeCount = myChallenges.length;
    const challengeRate = getAveragePercent(myChallenges.map(getChallengeProgress));
    const medicationActiveCount = medications.filter((item) => item.is_active !== false).length;
    const medicationRate = medications.length > 0 ? clampPercent((medicationActiveCount / medications.length) * 100) : 0;
    const RING_R = 28;
    const RING_C = 2 * Math.PI * RING_R;
    const ringOffset = RING_C * (1 - challengeRate / 100);

    const recommendationChallengeIds = challengeRecommendations
      .map((recommendation) => Number(recommendation.challenge_id))
      .filter((id) => Number.isFinite(id));
    const recommendedChallenges =
      recommendationChallengeIds.length > 0
        ? recommendationChallengeIds
            .map((id) => challenges.find((challenge) => Number(challenge.id) === id))
            .filter((challenge): challenge is AnyRecord => Boolean(challenge))
            .slice(0, 3)
        : challenges.slice(0, 3);
    return (
      <div className="page-stack">
        <div className="main-dashboard-hero">
          <div>
            <h1>안녕하세요, {backendUser?.nickname ?? backendUser?.name ?? "회원"}님</h1>
            <p>오늘의 건강 관리 상태를 확인해보세요.</p>
          </div>
          <div className="home-action-panel">
            <Link className="home-action-card" to="/analysis">
              <span className="home-action-card__icon">🧭</span>
              <span>
                <strong className="home-action-card__title">건강위험도 분석하기</strong>
                <em className="home-action-card__description">간편 분석으로 현재 건강정보 기반 질환별 결과를 확인합니다.</em>
              </span>
            </Link>
            <Link className="home-action-card" to="/diets">
              <span className="home-action-card__icon">🥗</span>
              <span>
                <strong className="home-action-card__title">식단 이미지 분석하기</strong>
                <em className="home-action-card__description">식단 이미지를 추가하면 식단 분석 결과를 확인할 수 있습니다.</em>
              </span>
            </Link>
            <div className="home-action-links">
              <Link to="/health">건강정보 입력</Link>
              <Link to="/ocr/exam">검진표 OCR 추가</Link>
            </div>
          </div>
        </div>
        {sectionError ? <div className="empty-state">{sectionError}</div> : null}

        <section className="main-dashboard-section">
          <div className="section-heading compact">
            <h2>오늘 할 일</h2>
            <p>지금 바로 이어갈 수 있는 건강 관리 행동입니다.</p>
          </div>
          <div className="todo-card-grid">
            {todayCards.map((task) => (
              <Link className="todo-card" key={task.title} to={task.to}>
                <span className="badge badge-reference">오늘</span>
                <strong>{task.title}</strong>
                <p>{task.description}</p>
                <em>{task.buttonLabel}</em>
              </Link>
            ))}
          </div>
        </section>

        <section className="main-dashboard-section">
          <div className="section-heading compact">
            <h2>건강 추적 요약</h2>
            <p>최근 기록된 건강 지표를 시각적으로 확인합니다.</p>
          </div>
          <div className="viz-card-grid">
            {/* 혈당 / 혈압 card */}
            <div className="viz-card">
              <div className="viz-card-row">
                <span className="viz-card-label">혈당 / 혈압</span>
                <Link className="muted" style={{ fontSize: 12 }} to="/ocr/exam">입력하기 →</Link>
              </div>
              {glucoseVal != null || systolicVal != null ? (
                <>
                  {glucoseVal != null && (
                    <div>
                      <div className="viz-card-value">{glucoseVal} <span className="viz-card-sub">mg/dL</span></div>
                      <div className="viz-card-sub">공복혈당</div>
                      <div style={{ marginTop: 8 }}>
                        <div className="viz-progress-label">
                          <span style={{ fontSize: 12, color: "var(--color-muted)" }}>0</span>
                          <span style={{ fontSize: 12, color: "var(--color-muted)" }}>200</span>
                        </div>
                        <div className="viz-progress-bar">
                          <div
                            className="viz-progress-fill"
                            style={{
                              width: `${Math.min(100, (glucoseVal / 200) * 100)}%`,
                              background: glucoseVal >= 126 ? "var(--color-danger)" : glucoseVal >= 100 ? "var(--chart-gold)" : "var(--chart-green)",
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                  {systolicVal != null && (
                    <div className="viz-stat-row">
                      <span>혈압</span>
                      <strong>{systolicVal}/{diastolicVal ?? "-"} <span className="viz-card-sub">mmHg</span></strong>
                    </div>
                  )}
                  {weightVal != null && (
                    <div className="viz-stat-row">
                      <span>체중 / BMI</span>
                      <strong>{weightVal}kg {bmiVal != null ? `/ ${bmiVal}` : ""}</strong>
                    </div>
                  )}
                </>
              ) : (
                <div className="viz-empty">
                  아직 건강 기록이 없습니다.<br />
                  <Link to="/health" style={{ color: "var(--color-primary)", fontWeight: 900 }}>건강정보 입력하기</Link>
                </div>
              )}
            </div>

            {/* 식단 점수 card */}
            <div className="viz-card">
              <div className="viz-card-row">
                <span className="viz-card-label">식단 점수</span>
                <Link className="muted" style={{ fontSize: 12 }} to="/diets">식단 분석 →</Link>
              </div>
              {latestDietScore != null ? (
                <>
                  <div>
                    <div className="viz-card-value">{String(latestDietScore)}<span className="viz-card-sub">점</span></div>
                    <div className="viz-card-sub">최근 식단 분석 결과</div>
                  </div>
                  {dietScoreBars.length > 0 ? (
                    <div className="viz-mini-bars">
                      {dietScoreBars.map((score, i) => (
                        <div
                          className={`viz-mini-bar${score >= 80 ? "" : score >= 60 ? " warn" : " danger"}`}
                          key={i}
                          style={{ height: `${Math.max(8, score)}%` }}
                        />
                      ))}
                    </div>
                  ) : (
                    <div className="viz-empty">추세 데이터가 부족합니다.</div>
                  )}
                </>
              ) : (
                <div className="viz-empty">
                  아직 식단 분석 기록이 없습니다.<br />
                  <Link to="/diets" style={{ color: "var(--color-primary)", fontWeight: 900 }}>식단 분석하기</Link>
                </div>
              )}
            </div>

            {/* 챌린지 / 복약 card */}
            <div className="viz-card">
              <span className="viz-card-label">챌린지 / 복약</span>
              <div className="viz-ring-container">
                <svg className="viz-ring-svg" viewBox="0 0 64 64">
                  <circle className="viz-ring-track" cx="32" cy="32" r={RING_R} />
                  <circle
                    className="viz-ring-fill"
                    cx="32"
                    cy="32"
                    r={RING_R}
                    strokeDasharray={RING_C}
                    strokeDashoffset={ringOffset}
                  />
                </svg>
                <div>
                  <div className="viz-ring-value">{challengeCount}개</div>
                  <div className="viz-ring-label">참여 중인 챌린지</div>
                  {challengeCount === 0 && (
                    <Link to="/challenges" style={{ fontSize: 13, color: "var(--color-primary)", fontWeight: 900 }}>챌린지 보기</Link>
                  )}
                </div>
              </div>
              <div className="viz-stat-row">
                <span>복약/영양제</span>
                <strong>
                  {medications.length > 0 ? `${medicationActiveCount}/${medications.length}개 복용 중` : (
                    <Link to="/medications" style={{ color: "var(--color-primary)", fontSize: 13 }}>등록하기</Link>
                  )}
                </strong>
              </div>
              {medications.length > 0 && (
                <div className="progress-bar">
                  <div className="progress-fill" style={toPercentStyle(medicationRate)} />
                </div>
              )}
              {latestExam && (
                <div className="viz-stat-row">
                  <span>최근 검진표</span>
                  <strong>{formatDate(latestExam.exam_date ?? latestExam.uploaded_at)}</strong>
                </div>
              )}
            </div>
          </div>
        </section>

        <div className="main-dashboard-grid">
          <Card title="추천 챌린지">
            {recommendedChallenges.length > 0 ? (
              <div className="compact-list">
                {recommendedChallenges.map((challenge) => (
                  <div className="compact-list-item" key={String(challenge.id ?? challenge.title)}>
                    <div>
                      <strong>{getChallengeTitle(challenge)}</strong>
                      <p>
                        {getCategoryLabel(challenge.category)} · {getChallengeDuration(challenge)}
                      </p>
                    </div>
                    <Link className="button secondary" to="/challenges">
                      참여하기
                    </Link>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <p>추천 챌린지가 없습니다. 전체 챌린지를 확인해보세요.</p>
                <Link className="button secondary" to="/challenges">
                  챌린지 보기
                </Link>
              </div>
            )}
          </Card>

          <Card title="AI 코멘트">
            <p>
              {String(
                data.ai_comment ??
                  "최근 입력된 건강정보를 바탕으로 생활습관을 꾸준히 기록해보세요.",
              )}
            </p>
            <div className="button-row">
              <Link className="button" to="/chatbot">
                AI에게 질문하기
              </Link>
              <Link className="button secondary" to="/analysis/history">
                분석 이력 보기
              </Link>
            </div>
          </Card>
        </div>

        <Card title={hasAnalysisResults ? "최근 분석 결과" : "간편 분석 시작"}>
          {hasAnalysisResults ? (
            <>
              {displayAnalysisResults.length > 0 ? (
                <div className="metric-grid">
                  {displayAnalysisResults.map((result) => (
                    <div key={String(result.id ?? result.analysis_type)}>
                      <span>{analysisTypeLabels[String(result.analysis_type)]}</span>
                      <strong>{formatRisk(result.risk_level)}</strong>
                    </div>
                  ))}
                  <div>
                    <span>최근 분석일</span>
                    <strong>{formatDate(displayAnalysisResults[0]?.analyzed_at ?? displayAnalysisResults[0]?.created_at)}</strong>
                  </div>
                </div>
              ) : (
                <div className="empty-state analysis-empty-state">
                  <strong>최근 분석 결과가 없습니다.</strong>
                  <p>질환별 분석을 실행하면 당뇨, 고혈압, 이상지질혈증, 비만 결과가 표시됩니다.</p>
                </div>
              )}
            </>
          ) : (
            <div className="empty-state analysis-empty-state">
              <strong>아직 분석 결과가 없습니다.</strong>
              <p>건강정보를 입력하고 간편 분석을 실행하면 현재 건강정보 기반 질환별 분석 결과를 볼 수 있습니다.</p>
              <p>검진표를 추가하면 정밀 분석에 활용할 수 있습니다.</p>
              <div className="button-row">
                <Link className="button" to="/analysis">
                  간편 분석 실행하기
                </Link>
                <Link className="button secondary" to="/health">
                  건강정보 입력하기
                </Link>
                <Link className="button secondary" to="/ocr/exam">
                  검진표 OCR 추가
                </Link>
              </div>
            </div>
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
            <div className="empty-state">
              <p>등록된 검진표가 없습니다. 검진표를 업로드하면 주요 수치를 자동으로 정리합니다.</p>
            </div>
          )}
        </Card>
      </div>
    );
  }

  const selectedPersona = landingPersonas.find((persona) => persona.id === selectedPersonaId) ?? landingPersonas[0];

  return (
    <div className="landing-page">
      <section className="hero-panel">
        <div>
          <span className="eyebrow">HealthCare</span>
          <h1>건강검진표부터 생활습관까지, 내 건강 위험도를 쉽게 확인하세요</h1>
          <p>{String(data.service_description)}</p>
          <div className="button-row">
            <Link className="button" to="/signup">
              무료로 시작하기
            </Link>
            <Link className="button secondary" to="/login">
              로그인
            </Link>
          </div>
        </div>
        <div className="mobile-health-card">
          <div className="hero-card-topline">
            <span className="badge badge-reference">예시 화면</span>
            <div className="mini-gauge" style={toGaugeStyle(landingPreview.healthScore)}>
              <span>{landingPreview.healthScore}</span>
            </div>
          </div>
          <div className="sparkline-row">
            <div>
              <span>혈당</span>
              <strong>108 mg/dL</strong>
            </div>
            <div className="sparkline sparkline-glucose" aria-hidden="true">
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>
          </div>
          <div className="sparkline-row">
            <div>
              <span>혈압</span>
              <strong>132/84</strong>
            </div>
            <div className="sparkline sparkline-pressure" aria-hidden="true">
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>
          </div>
          <div>
            <div className="progress-label">
              <span>챌린지 수행률</span>
              <strong>{landingPreview.challengeProgress}%</strong>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={toPercentStyle(landingPreview.challengeProgress)} />
            </div>
          </div>
          <p className="muted">건강 점수와 주요 지표를 한눈에 확인합니다.</p>
        </div>
      </section>

      <section className="landing-section">
        <div className="section-heading">
          <h2>내 상황에 맞는 시작 방법</h2>
          <p>가장 가까운 상황을 선택하면 어떤 흐름으로 서비스를 쓰면 좋은지 보여드립니다.</p>
        </div>
        <div className="persona-grid">
          {landingPersonas.map((persona) => (
            <button
              className={`persona-card ${persona.id === selectedPersona.id ? "active" : ""}`}
              key={persona.id}
              type="button"
              onClick={() => setSelectedPersonaId(persona.id)}
            >
              <span className="persona-icon">{persona.icon}</span>
              <strong>{persona.title}</strong>
              <p>{persona.quote}</p>
              <em>{persona.features.join(" · ")}</em>
            </button>
          ))}
        </div>
      </section>

      <section className="persona-flow-panel">
        <div>
          <span className="eyebrow">추천 흐름</span>
          <h2>{selectedPersona.title}</h2>
          <p>{selectedPersona.quote}</p>
        </div>
        <div className="persona-timeline">
          {selectedPersona.flow.map((step, index) => (
            <div className="persona-timeline-item" key={step.title}>
              <span className="timeline-icon">{step.icon}</span>
              <div>
                <strong>{step.title}</strong>
                <p>{step.description}</p>
              </div>
              {index < selectedPersona.flow.length - 1 ? <em aria-hidden="true">→</em> : null}
            </div>
          ))}
        </div>
        <div className="button-row">
          <Link className="button" to="/signup">
            내 건강 분석 시작하기
          </Link>
          <Link className="button secondary" to="/login">
            로그인 후 이용하기
          </Link>
        </div>
      </section>

      <section className="landing-section" id="features">
        <div className="section-heading">
          <h2>기능 미리보기</h2>
          <p>아이콘을 눌러 관심 기능을 확인하고 로그인 후 바로 이어갈 수 있습니다.</p>
        </div>
        <div className="service-flow-strip">
          {serviceFlow.map((item, index) => (
            <div className="service-flow-item" key={item.label}>
              <span>{item.icon}</span>
              <strong>{item.label}</strong>
              {index < serviceFlow.length - 1 ? <em aria-hidden="true">→</em> : null}
            </div>
          ))}
        </div>
        <div className="landing-feature-grid">
          {landingFeatures.map((feature) => (
            <Link
              className="landing-feature-card"
              key={feature.title}
              to={`/login?redirect=${encodeURIComponent(feature.redirect)}`}
            >
              <span className="landing-feature-icon">{feature.icon}</span>
              <strong>{feature.title}</strong>
              <p>{feature.description}</p>
              <em className="badge badge-reference">로그인 후 이용</em>
            </Link>
          ))}
        </div>
      </section>
      <section className="landing-section">
        <Card title="예시 대시보드 미리보기">
          <div className="preview-dashboard-grid">
            <div className="preview-score-card">
              <div className="score-gauge" style={toGaugeStyle(landingPreview.healthScore)}>
                <span>{landingPreview.healthScore}</span>
              </div>
              <strong>오늘의 건강 점수</strong>
              <p>최근 기록 기반 예시 화면입니다.</p>
            </div>
            <div className="preview-bars">
              <div className="preview-bar-row">
                <span>당뇨 위험도</span>
                <strong>관리 필요</strong>
                <div className="progress-bar">
                  <div className="progress-fill warning-fill" style={toPercentStyle(landingPreview.diabetesRiskScore)} />
                </div>
              </div>
              <div className="preview-bar-row">
                <span>고혈압 위험도</span>
                <strong>낮음</strong>
                <div className="progress-bar">
                  <div className="progress-fill success-fill" style={toPercentStyle(landingPreview.hypertensionRiskScore)} />
                </div>
              </div>
              <div className="preview-bar-row">
                <span>챌린지 진행률</span>
                <strong>{landingPreview.challengeProgress}%</strong>
                <div className="progress-bar">
                  <div className="progress-fill" style={toPercentStyle(landingPreview.challengeProgress)} />
                </div>
              </div>
            </div>
            <div className="preview-record-card">
              <span className="badge badge-reference">최근 기록</span>
              <strong>{previewMetrics[4][1]}</strong>
              <p>검진표 OCR과 직접 입력 기록을 함께 관리합니다.</p>
            </div>
          </div>
        </Card>
      </section>
      <section className="landing-cta">
        <div>
          <h2>내 건강 데이터를 입력하고 맞춤 분석을 받아보세요.</h2>
          <p>회원가입 후 기본 건강정보를 입력하면 건강 위험도 분석과 맞춤 챌린지를 바로 확인할 수 있습니다.</p>
        </div>
        <div className="button-row">
          <Link className="button" to="/signup">
            회원가입하고 시작하기
          </Link>
          <Link className="button secondary" to="/faqs">
            FAQ 보기
          </Link>
        </div>
      </section>
    </div>
  );
}
