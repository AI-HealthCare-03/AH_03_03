import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { getLatestAnalysisResults } from "../api/analysis";
import { getMainSummary, getPublicMain } from "../api/main";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

type MainData = Record<string, unknown>;
type AnyRecord = Record<string, unknown>;

const publicFallback = {
  service_title: "HealthCare",
  service_description: "AI 기반 분석과 맞춤 챌린지로 나의 건강 상태를 관리해보세요.",
  sample_health_cards: ["AI 위험도 분석", "헬스케어 코치", "쉬운 기록", "추적 대시보드", "가족 관리"],
  sample_challenges: ["혈당", "혈압", "체중", "챌린지 수행률", "식단 점수"],
  locked_features: ["개인 대시보드", "맞춤 분석", "챌린지 참여"],
};

const riskLabelMap: Record<string, string> = {
  LOW: "낮음",
  MEDIUM: "관리 필요",
  HIGH: "높음",
};

function asRecord(value: unknown): AnyRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as AnyRecord) : {};
}

function asRecordArray(value: unknown): AnyRecord[] {
  return Array.isArray(value) ? value.filter((item): item is AnyRecord => Boolean(item) && typeof item === "object") : [];
}

function formatRisk(value: unknown): string {
  const raw = String(value ?? "").toUpperCase();
  return riskLabelMap[raw] ?? (raw || "-");
}

function formatDate(value: unknown): string {
  if (!value) {
    return "-";
  }
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString("ko-KR");
}

function buildOverallRisk(levels: string[]): string {
  if (levels.includes("HIGH")) {
    return "높음";
  }
  if (levels.includes("MEDIUM")) {
    return "관리 필요";
  }
  if (levels.length > 0 && levels.every((level) => level === "LOW")) {
    return "낮음";
  }
  return "-";
}

export default function MainPage() {
  const { backendUser, isAuthenticated } = useAuth();
  const [data, setData] = useState<MainData>(publicFallback);
  const [analysisResults, setAnalysisResults] = useState<AnyRecord[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        if (isAuthenticated) {
          const [summary, latestResults] = await Promise.all([
            getMainSummary<MainData>(),
            getLatestAnalysisResults<AnyRecord[]>(),
          ]);
          setData(summary);
          setAnalysisResults(latestResults);
          return;
        }
        setData(await getPublicMain<MainData>());
        setAnalysisResults([]);
      } catch {
        setData(publicFallback);
        setAnalysisResults([]);
      }
    };
    void load();
  }, [isAuthenticated]);

  if (isAuthenticated) {
    const latestHealth = asRecord(data.latest_health_summary);
    const latestAnalysis = asRecord(data.latest_analysis_summary);
    const dashboardSummary = asRecord(data.dashboard_summary);
    const activeChallengeSummary = asRecord(data.active_challenge_summary);
    const todayTasks = asRecordArray(data.today_tasks);
    const recentRecords = asRecord(data.recent_records);
    const recentHealthRecords = asRecordArray(recentRecords.health_records);
    const effectiveAnalysisResults =
      analysisResults.length > 0 ? analysisResults : latestAnalysis.analysis_type ? [latestAnalysis] : [];
    const resultByType = new Map(
      effectiveAnalysisResults.map((result) => [
        String(result.analysis_type ?? ""),
        String(result.risk_level ?? "").toUpperCase(),
      ]),
    );
    const riskLevels =
      effectiveAnalysisResults.length > 0
        ? effectiveAnalysisResults.map((result) => String(result.risk_level ?? "").toUpperCase()).filter(Boolean)
        : [];
    const taskFallback = [
      { title: latestHealth.id ? "건강정보 확인" : "건강정보 입력", url: "/health" },
      { title: "식단 기록", url: "/diets" },
      { title: activeChallengeSummary.my_challenge_count ? "챌린지 수행" : "챌린지 참여", url: "/challenges" },
      { title: dashboardSummary.active_medication_count ? "복약 기록" : "복약/영양제 등록", url: "/medications" },
    ];
    const taskCards =
      todayTasks.length > 0
        ? todayTasks.slice(0, 4).map((task) => ({
            title: String(task.title ?? "오늘 할 일"),
            url: task.task_type === "MEDICATION" ? "/medications" : "/challenges",
          }))
        : taskFallback;
    const miniCards = [
      ["혈당", latestHealth.fasting_glucose ? `${String(latestHealth.fasting_glucose)} mg/dL` : "-"],
      [
        "혈압",
        latestHealth.systolic_bp || latestHealth.diastolic_bp
          ? `${String(latestHealth.systolic_bp ?? "-")}/${String(latestHealth.diastolic_bp ?? "-")} mmHg`
          : "-",
      ],
      ["체중", latestHealth.weight_kg ? `${String(latestHealth.weight_kg)} kg` : "-"],
      ["챌린지", `${String(activeChallengeSummary.my_challenge_count ?? 0)}개 진행`],
      ["최근 기록", `${recentHealthRecords.length}건`],
    ];

    return (
      <div className="page-stack">
        <div className="page-header">
          <div>
            <h1>안녕하세요, {backendUser?.nickname ?? backendUser?.name ?? "회원"}님</h1>
            <p>오늘의 건강 기록과 위험도 요약을 확인해보세요.</p>
          </div>
          <Link className="button" to="/health">
            건강 분석 시작하기
          </Link>
        </div>
        <div className="metric-grid">
          {taskCards.map((task) => (
            <Link className="stat-card" key={task.title} to={task.url}>
              <span>오늘 할 일</span>
              <strong>{task.title}</strong>
            </Link>
          ))}
        </div>
        <div className="page-grid">
          <Card title="위험도 요약">
            <div className="metric-grid">
              <div>
                <span>당뇨 위험도</span>
                <strong>{formatRisk(resultByType.get("DIABETES"))}</strong>
              </div>
              <div>
                <span>고혈압 위험도</span>
                <strong>{formatRisk(resultByType.get("HYPERTENSION"))}</strong>
              </div>
              <div>
                <span>종합 위험도</span>
                <strong>{buildOverallRisk(riskLevels)}</strong>
              </div>
              <div>
                <span>최근 분석일</span>
                <strong>{formatDate(latestAnalysis.analyzed_at ?? latestAnalysis.created_at)}</strong>
              </div>
            </div>
          </Card>
          <Card title="AI 건강 코멘트">
            <p>{String(data.ai_comment ?? "혈당과 혈압을 함께 추적하면 생활습관 변화 효과를 더 쉽게 볼 수 있어요.")}</p>
          </Card>
          <Card title="추천 액션">
            <div className="button-row">
              <Link className="button secondary" to="/challenges">
                식후 10분 걷기
              </Link>
              <Link className="button secondary" to="/diets">
                식단 이미지 분석
              </Link>
              <Link className="button secondary" to="/dashboard">
                추적 대시보드
              </Link>
            </div>
          </Card>
          <Card title="추적 미니 카드">
            <div className="metric-grid">
              {miniCards.map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <section className="hero-panel">
        <div>
          <span className="eyebrow">HealthCare MVP</span>
          <h1>당뇨와 고혈압 위험도를 쉽게 확인하고 건강한 습관을 만들어가세요</h1>
          <p>{String(data.service_description)}</p>
          <div className="button-row">
            <Link className="button" to="/signup">
              건강 분석 시작하기
            </Link>
            <Link className="button secondary" to="/login">
              로그인 후 이용 가능
            </Link>
          </div>
        </div>
        <div className="mobile-health-card">
          <span className="badge risk-medium">오늘의 건강 요약</span>
          <strong>혈당 108 mg/dL</strong>
          <strong>혈압 132/84 mmHg</strong>
          <strong>챌린지 수행률 72%</strong>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: "72%" }} />
          </div>
          <p className="muted">모바일 건강 카드 예시입니다.</p>
        </div>
      </section>
      <div className="metric-grid">
        {publicFallback.sample_health_cards.map((feature) => (
          <div className="metric-card card" key={feature}>
            <span>주요 기능</span>
            <strong>{feature}</strong>
            <p className="muted">로그인 후 이용 가능</p>
          </div>
        ))}
      </div>
      <Card title="예시 추적 대시보드">
        <div className="metric-grid">
          {publicFallback.sample_challenges.map((label) => (
            <div key={label}>
              <span>{label}</span>
              <strong>-</strong>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
