import { useEffect, useState } from "react";

import { getLatestAnalysisResults, runDummyAnalysis } from "../api/analysis";
import { getAnalysisReadiness } from "../api/health";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type AnalysisResult = Record<string, unknown>;

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
    <div className="page-grid">
      <Card title="건강 분석" actions={<button onClick={run}>더미 분석 실행</button>}>
        {error && <ErrorMessage message={error} />}
        <div className="card-list">
          {results.map((result) => (
            <div className="mini-card" key={String(result.id)}>
              <strong>{String(result.analysis_type)}</strong>
              <span>위험도: {String(result.risk_level)}</span>
              <span>점수: {String(result.risk_score)}</span>
              <p>{String(result.summary ?? "주요 위험요인 카드는 더미 factor 저장 후 상세 API에서 확장 예정입니다.")}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
