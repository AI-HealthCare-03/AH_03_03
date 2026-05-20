import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getAnalysisResultDetail, listAnalysisResults } from "../api/analysis";
import Card from "../components/Card";

type AnalysisResult = Record<string, unknown>;
type AnalysisDetail = {
  result?: AnalysisResult;
  factors?: AnalysisResult[];
  snapshot?: AnalysisResult | null;
};

const labels: Record<string, string> = {
  DIABETES: "당뇨",
  OBESITY: "비만",
  DYSLIPIDEMIA: "이상지질혈증",
  HYPERTENSION: "고혈압",
};

export default function AnalysisHistoryPage() {
  const { analysisId } = useParams();
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);

  useEffect(() => {
    const load = async () => {
      if (analysisId) {
        setDetail(await getAnalysisResultDetail<AnalysisDetail>(Number(analysisId)));
        return;
      }
      setResults(await listAnalysisResults<AnalysisResult[]>());
    };
    void load().catch(() => {
      setResults([]);
      setDetail(null);
    });
  }, [analysisId]);

  if (analysisId) {
    const result = detail?.result;
    return (
      <div className="page-grid">
        <Card
          title="위험도 추론 결과 상세"
          actions={
            <Link className="button secondary" to="/analysis/history">
              전체 리스트
            </Link>
          }
        >
          <div className="score-panel">
            <span>{labels[String(result?.analysis_type)] ?? String(result?.analysis_type ?? "분석")}</span>
            <strong>{String(result?.risk_level ?? "-")}</strong>
            <p>{String(result?.summary ?? "")}</p>
          </div>
        </Card>
        <Card title="주요 위험 요인">
          <pre>{JSON.stringify(detail?.factors ?? [], null, 2)}</pre>
        </Card>
        <Card title="분석 스냅샷">
          <pre>{JSON.stringify(detail?.snapshot ?? {}, null, 2)}</pre>
        </Card>
      </div>
    );
  }

  return (
    <Card
      title="위험도 추론 결과 전체 리스트"
      actions={
        <Link className="button" to="/analysis">
          분석 실행
        </Link>
      }
    >
      <div className="filter-tabs">
        {["전체", "당뇨 위험도", "고혈압 위험도", "종합 위험도", "최근 3개월"].map((tab, index) => (
          <span className={index === 0 ? "filter-tab active" : "filter-tab"} key={tab}>
            {tab}
          </span>
        ))}
      </div>
      <div className="table-list">
        {results.map((result) => (
          <Link className="table-row" key={String(result.id)} to={`/analysis/${String(result.id)}`}>
            <span>{String(result.risk_level)}</span>
            <strong>{labels[String(result.analysis_type)] ?? String(result.analysis_type)}</strong>
            <span>{String(result.analyzed_at ?? result.created_at ?? "")}</span>
          </Link>
        ))}
        {results.length === 0 && <p className="placeholder">아직 분석 결과가 없습니다.</p>}
      </div>
    </Card>
  );
}
