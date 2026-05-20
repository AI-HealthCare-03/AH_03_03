import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { getLatestAnalysisResults, runDummyAnalysis } from "../api/analysis";
import { getAnalysisReadiness } from "../api/health";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type AnalysisResult = Record<string, unknown>;

const analysisTypeLabels: Record<string, string> = {
  DIABETES: "당뇨",
  OBESITY: "비만",
  DYSLIPIDEMIA: "이상지질혈증",
  HYPERTENSION: "고혈압",
};

const riskFallbackScores: Record<string, number> = {
  HIGH: 80,
  MEDIUM: 55,
  LOW: 25,
};

function getRiskLevel(result: AnalysisResult): string {
  return String(result.risk_level ?? "").toUpperCase();
}

function getRiskScore(result: AnalysisResult): number {
  const raw = Number(result.risk_score);
  if (Number.isFinite(raw) && raw > 0) {
    return Math.round(raw <= 1 ? raw * 100 : raw);
  }
  return riskFallbackScores[getRiskLevel(result)] ?? 0;
}

function getOverallLevel(results: AnalysisResult[]): string {
  const levels = results.map(getRiskLevel).filter(Boolean);
  if (levels.includes("HIGH")) {
    return "HIGH";
  }
  if (levels.includes("MEDIUM")) {
    return "MEDIUM";
  }
  if (levels.length > 0 && levels.every((level) => level === "LOW")) {
    return "LOW";
  }
  return "-";
}

function getOverallScore(results: AnalysisResult[]): number {
  if (results.length === 0) {
    return 0;
  }
  const scores = results.map(getRiskScore).filter((score) => score > 0);
  if (scores.length > 0) {
    return Math.max(...scores);
  }
  return riskFallbackScores[getOverallLevel(results)] ?? 0;
}

export default function AnalysisPage() {
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [healthRecordId, setHealthRecordId] = useState<number | null>(null);
  const [error, setError] = useState("");

  const load = async () => {
    setResults(await getLatestAnalysisResults<AnalysisResult[]>());
    const readiness = await getAnalysisReadiness<{ latest_health_record_id: number | null }>();
    setHealthRecordId(readiness.latest_health_record_id);
  };

  useEffect(() => {
    void load().catch(() => setError("분석 결과를 불러오지 못했습니다."));
  }, []);

  const run = async () => {
    if (!healthRecordId) {
      setError("더미 분석을 실행할 건강정보가 없습니다.");
      return;
    }
    await runDummyAnalysis<unknown>(healthRecordId);
    await load();
  };

  const overallLevel = getOverallLevel(results);
  const overallScore = getOverallScore(results);

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>건강 분석 결과</h1>
          <p>당뇨, 고혈압, 비만, 이상지질혈증 위험도를 한 화면에서 확인합니다.</p>
        </div>
        <div className="button-row">
          <button onClick={run}>더미 분석 실행</button>
          <Link className="button secondary" to="/analysis/history">
            전체 결과
          </Link>
        </div>
      </div>
      {error && <ErrorMessage message={error} />}
      <div className="page-grid">
        <Card title="종합 위험도">
          <div className="score-panel">
            <span>종합 위험도 게이지</span>
            <strong>{overallLevel}</strong>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${overallScore}%` }} />
            </div>
            <p>이 결과는 MVP 시연용 더미 분석이며 실제 의료 진단이 아닙니다.</p>
          </div>
        </Card>
        <Card title="AI 건강 코멘트">
          <p>혈압과 혈당을 함께 추적하고, 식후 산책과 저당 식단 챌린지를 병행해보세요.</p>
        </Card>
      </div>
      <div className="metric-grid">
        {results.map((result) => {
          const level = getRiskLevel(result);
          const score = getRiskScore(result);
          return (
            <div className="metric-card card" key={String(result.id)}>
              <span>{analysisTypeLabels[String(result.analysis_type)] ?? String(result.analysis_type)} 위험도</span>
              <strong>{score}/100</strong>
              <span className={`badge risk-${level.toLowerCase()}`}>{level || "-"}</span>
              <p>{String(result.summary ?? "주요 factor는 상세 화면에서 확인할 수 있습니다.")}</p>
            </div>
          );
        })}
        {results.length === 0 && (
          <div className="metric-card card">
            <span>분석 결과 없음</span>
            <strong>-</strong>
            <p>건강정보를 입력한 뒤 더미 분석을 실행하면 seed 데이터와 같은 결과 카드가 표시됩니다.</p>
          </div>
        )}
      </div>
      <Card title="추천 액션">
        <div className="button-row">
          <Link className="button secondary" to="/challenges">
            식후 10분 걷기
          </Link>
          <Link className="button secondary" to="/diets">
            저녁 식단 추천
          </Link>
          <Link className="button secondary" to="/dashboard">
            추적 대시보드 이동
          </Link>
          <Link className="button secondary" to="/diets">
            식단 이미지 분석
          </Link>
        </div>
      </Card>
      <Card title="상세 요인 카드">
        <div className="card-list">
          {results.map((result) => (
            <div className="mini-card" key={String(result.id)}>
              <strong>{analysisTypeLabels[String(result.analysis_type)] ?? String(result.analysis_type)}</strong>
              <span>주요 factor: 혈당, 혈압, BMI, 지질 지표</span>
              <Link to={`/analysis/${String(result.id)}`}>상세보기</Link>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
