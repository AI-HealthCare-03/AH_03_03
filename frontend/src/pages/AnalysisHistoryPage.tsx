import { useMemo, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getAnalysisResultDetail, listAnalysisResults } from "../api/analysis";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

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
  OVERALL: "종합 위험도",
};

const tabs = ["전체", "당뇨", "고혈압", "이상지질혈증", "종합 위험도"];
const tabToType: Record<string, string | null> = {
  전체: null,
  당뇨: "DIABETES",
  고혈압: "HYPERTENSION",
  이상지질혈증: "DYSLIPIDEMIA",
  "종합 위험도": "OVERALL",
};

function modelLabel(result: AnalysisResult | undefined): string | null {
  const modelName = result?.model_name ? String(result.model_name) : "";
  const modelVersion = result?.model_version ? String(result.model_version) : "";
  if (!modelName && !modelVersion) {
    return null;
  }
  if (modelName.toLowerCase() === "catboost") {
    return modelVersion ? `CatBoost · ${modelVersion}` : "CatBoost";
  }
  if (modelName) {
    return modelVersion ? `${modelName} · ${modelVersion}` : modelName;
  }
  return modelVersion;
}

export default function AnalysisHistoryPage() {
  const { analysisId } = useParams();
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [activeTab, setActiveTab] = useState("전체");
  const [error, setError] = useState("");

  const formatDate = (value: unknown) => {
    if (!value) {
      return "-";
    }
    const date = new Date(String(value));
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString("ko-KR");
  };

  const scoreLabel = (value: unknown) => {
    const score = Number(value);
    if (!Number.isFinite(score)) {
      return "-";
    }
    return `${Math.round(score <= 1 ? score * 100 : score)}/100`;
  };

  const getScore = (result: AnalysisResult) => {
    const score = Number(result.risk_score);
    if (!Number.isFinite(score)) {
      return null;
    }
    return score <= 1 ? score * 100 : score;
  };

  const overallRow = useMemo<AnalysisResult | null>(() => {
    if (results.length === 0) {
      return null;
    }
    const scoreValues = results.map(getScore).filter((value): value is number => value !== null);
    const highest = scoreValues.length > 0 ? Math.max(...scoreValues) : null;
    const hasHigh = results.some((result) => String(result.risk_level).toUpperCase() === "HIGH");
    const hasMedium = results.some((result) => String(result.risk_level).toUpperCase() === "MEDIUM");
    return {
      id: "overall",
      analysis_type: "OVERALL",
      risk_level: hasHigh ? "HIGH" : hasMedium ? "MEDIUM" : "LOW",
      risk_score: highest,
      created_at: results[0]?.analyzed_at ?? results[0]?.created_at,
      summary: "질환별 최신 결과를 기준으로 계산한 종합 요약입니다.",
    };
  }, [results]);

  const displayResults = useMemo(() => {
    const merged = overallRow ? [...results, overallRow] : results;
    const selectedType = tabToType[activeTab];
    if (!selectedType) {
      return merged;
    }
    return merged.filter((result) => String(result.analysis_type) === selectedType);
  }, [activeTab, overallRow, results]);

  const mainFactorLabel = (result: AnalysisResult) => {
    const summary = result.summary ?? result.guide_message ?? result.message;
    if (summary) {
      return String(summary);
    }
    if (String(result.analysis_type) === "OVERALL") {
      return "종합 위험도는 질환별 위험도를 기준으로 요약됩니다.";
    }
    return "상세보기에서 주요 요인을 확인할 수 있습니다.";
  };

  const snapshotInput =
    detail?.snapshot?.input_payload && typeof detail.snapshot.input_payload === "object"
      ? (detail.snapshot.input_payload as Record<string, unknown>)
      : {};
  const snapshotOutput =
    detail?.snapshot?.output_payload && typeof detail.snapshot.output_payload === "object"
      ? (detail.snapshot.output_payload as Record<string, unknown>)
      : {};

  useEffect(() => {
    const load = async () => {
      setError("");
      if (analysisId) {
        setDetail(await getAnalysisResultDetail<AnalysisDetail>(Number(analysisId)));
        return;
      }
      setResults(await listAnalysisResults<AnalysisResult[]>());
    };
    void load().catch(() => {
      setResults([]);
      setDetail(null);
      setError("분석 결과를 불러오지 못했습니다.");
    });
  }, [analysisId]);

  if (analysisId) {
    const result = detail?.result;
    return (
      <div className="page-grid">
        {error && <ErrorMessage message={error} />}
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
            <span className="badge badge-reference">{scoreLabel(result?.risk_score)}</span>
            <span className="badge badge-reference">{result?.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
            {modelLabel(result) && <span className="badge badge-reference">{modelLabel(result)}</span>}
            <p>{String(result?.summary ?? "")}</p>
          </div>
        </Card>
        <Card title="주요 위험 요인">
          <div className="card-list">
            {(detail?.factors ?? []).length === 0 && <div className="state-box">표시할 주요 요인이 없습니다.</div>}
            {(detail?.factors ?? []).map((factor) => (
              <div className="mini-card" key={String(factor.id ?? factor.factor_key)}>
                <div className="record-row">
                  <div>
                    <strong>{String(factor.factor_name ?? factor.factor_key ?? "요인")}</strong>
                    <p className="muted">{String(factor.factor_value ?? "값 정보 없음")}</p>
                  </div>
                  <span className="badge badge-reference">{String(factor.direction ?? "NEUTRAL")}</span>
                </div>
                <span className="badge risk-medium">기여도 {String(factor.contribution_score ?? "-")}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card title="분석 입력 요약">
          <div className="record-table">
            {[
              ["키", snapshotInput.height_cm],
              ["몸무게", snapshotInput.weight_kg],
              ["BMI", snapshotInput.bmi],
              ["혈압", `${String(snapshotInput.systolic_bp ?? "-")} / ${String(snapshotInput.diastolic_bp ?? "-")}`],
              ["공복혈당", snapshotInput.fasting_glucose],
              ["위험도", snapshotOutput.risk_level ?? result?.risk_level],
            ].map(([label, value]) => (
              <div className="record-table-row" key={String(label)}>
                <span>{String(label)}</span>
                <strong>{String(value ?? "-")}</strong>
              </div>
            ))}
          </div>
          {!detail?.snapshot && <div className="state-box">분석 입력 요약이 아직 저장되지 않았습니다.</div>}
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
      {error && <ErrorMessage message={error} />}
      <div className="filter-tabs">
        {tabs.map((tab) => (
          <button
            className={activeTab === tab ? "filter-tab active" : "filter-tab"}
            key={tab}
            onClick={() => setActiveTab(tab)}
            type="button"
          >
            {tab}
          </button>
        ))}
      </div>
      <div className="table-list">
        {displayResults.map((result) => {
          const isOverall = String(result.analysis_type) === "OVERALL";
          const content = (
            <>
            <span className={`badge risk-${String(result.risk_level ?? "").toLowerCase()}`}>
              {String(result.risk_level ?? "-")}
            </span>
            <div>
              <strong>{labels[String(result.analysis_type)] ?? String(result.analysis_type)}</strong>
              <p className="muted">{mainFactorLabel(result)}</p>
            </div>
            <span>{scoreLabel(result.risk_score)}</span>
            <span>{formatDate(result.analyzed_at ?? result.created_at)}</span>
            <span className="badge badge-reference">{result.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
            <span className="badge badge-reference">{isOverall ? "요약" : "상세보기"}</span>
            </>
          );
          return isOverall ? (
            <div className="table-row analysis-history-row" key="overall">
              {content}
            </div>
          ) : (
            <Link className="table-row analysis-history-row" key={String(result.id)} to={`/analysis/${String(result.id)}`}>
              {content}
            </Link>
          );
        })}
        {displayResults.length === 0 && (
          <div className="state-box">
            {activeTab === "종합 위험도" ? "종합 위험도는 후속 산출 예정입니다." : "표시할 분석 결과가 없습니다."}
          </div>
        )}
      </div>
    </Card>
  );
}
