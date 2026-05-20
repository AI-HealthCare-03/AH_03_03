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
            <strong>{results.some((item) => item.risk_level === "HIGH") ? "HIGH" : "MEDIUM"}</strong>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: "68%" }} />
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
          const level = String(result.risk_level).toLowerCase();
          const score = Math.round(Number(result.risk_score ?? 0) * 100);
          return (
            <div className="metric-card card" key={String(result.id)}>
              <span>{analysisTypeLabels[String(result.analysis_type)] ?? String(result.analysis_type)} 위험도</span>
              <strong>{score}/100</strong>
              <span className={`badge risk-${level}`}>{String(result.risk_level)}</span>
              <p>{String(result.summary ?? "주요 factor는 상세 화면에서 확인할 수 있습니다.")}</p>
            </div>
          );
        })}
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
