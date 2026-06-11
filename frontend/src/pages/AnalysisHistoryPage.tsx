import { useMemo, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getAnalysisResultDetail, listAnalysisResults } from "../api/analysis";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import { getDisplayRiskLabel, getDisplayRiskScoreLabel, getRiskClassName } from "../utils/riskDisplay";

type AnalysisResult = Record<string, unknown>;
type ReferenceSource = {
  id?: string | null;
  title?: string | null;
  source_org?: string | null;
  source_url?: string | null;
  year?: number | null;
  status?: string | null;
};
type AnalysisExplanation = {
  summary?: string;
  caution?: string;
  recommended_action?: string;
  safety_notice?: string;
  source?: string;
  reference_summary?: string | null;
  reference_sources?: ReferenceSource[];
};
type AnalysisDetail = {
  result?: AnalysisResult;
  factors?: AnalysisResult[];
  snapshot?: AnalysisResult | null;
  explanation?: AnalysisExplanation | null;
};

const labels: Record<string, string> = {
  DIABETES: "당뇨",
  OBESITY: "비만",
  DYSLIPIDEMIA: "콜레스테롤·중성지방",
  HYPERTENSION: "고혈압",
};

const tabs = ["전체", "당뇨", "고혈압", "콜레스테롤·중성지방", "비만"];
const tabToType: Record<string, string | null> = {
  전체: null,
  당뇨: "DIABETES",
  고혈압: "HYPERTENSION",
  "콜레스테롤·중성지방": "DYSLIPIDEMIA",
  비만: "OBESITY",
};

const factorValueLabels: Record<string, Record<string, string>> = {
  family_htn: { YES: "있음", NO: "없음", UNKNOWN: "모름" },
  family_dm: { YES: "있음", NO: "없음", UNKNOWN: "모름" },
  family_dyslipidemia: { YES: "있음", NO: "없음", UNKNOWN: "모름" },
  smoking_status: { NON_SMOKER: "비흡연", PAST_SMOKER: "과거 흡연", CURRENT_SMOKER: "현재 흡연" },
  drinking_frequency: {
    RARE: "월 1회 미만",
    MONTHLY_2_4: "월 2-4회",
    WEEKLY_2_3: "주 2-3회",
    WEEKLY_4_PLUS: "주 4회 이상",
  },
  drinking_amount: {
    NONE: "마시지 않음",
    ONE_TO_TWO: "1-2잔",
    THREE_TO_FOUR: "3-4잔",
    FIVE_TO_SIX: "5-6잔",
    SEVEN_PLUS: "7잔 이상",
  },
};

const factorDirectionLabels: Record<string, string> = {
  POSITIVE: "위험 증가",
  NEGATIVE: "위험 감소",
  NEUTRAL: "중립",
};

function displayFactorValue(factor: AnalysisResult): string {
  const key = String(factor.factor_key ?? "");
  const rawValue = factor.factor_value;
  if (rawValue === undefined || rawValue === null || rawValue === "") {
    return "값 정보 없음";
  }
  const value = String(rawValue);
  return factorValueLabels[key]?.[value] ?? value;
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

  const displayResults = useMemo(() => {
    const selectedType = tabToType[activeTab];
    if (!selectedType) {
      return results;
    }
    return results.filter((result) => String(result.analysis_type) === selectedType);
  }, [activeTab, results]);

  const mainFactorLabel = (result: AnalysisResult) => {
    const summary = result.summary ?? result.guide_message ?? result.message;
    if (summary) {
      return String(summary);
    }
    return "상세보기에서 주요 요인을 확인할 수 있습니다.";
  };

  const snapshotInput =
    detail?.snapshot?.input_payload && typeof detail.snapshot.input_payload === "object"
      ? (detail.snapshot.input_payload as Record<string, unknown>)
      : {};
  const referenceSources = detail?.explanation?.reference_sources ?? [];

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
            <strong>{getDisplayRiskLabel(result)}</strong>
            <span className="badge badge-reference">{getDisplayRiskScoreLabel(result)}</span>
            <span className="badge badge-reference">{result?.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
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
                    <p className="muted">{displayFactorValue(factor)}</p>
                  </div>
                  <span className="badge badge-reference">
                    {factorDirectionLabels[String(factor.direction ?? "NEUTRAL")] ?? String(factor.direction ?? "중립")}
                  </span>
                </div>
                <span className="badge risk-medium">기여도 {String(factor.contribution_score ?? "-")}</span>
              </div>
            ))}
          </div>
        </Card>
        {detail?.explanation && (
          <Card title="분석 설명">
            <div className="card-list">
              <div className="mini-card">
                <strong>{detail.explanation.summary ?? "분석 설명"}</strong>
                {detail.explanation.caution && <p className="muted">{detail.explanation.caution}</p>}
                {detail.explanation.recommended_action && <p>{detail.explanation.recommended_action}</p>}
                {detail.explanation.reference_summary && <p className="muted">{detail.explanation.reference_summary}</p>}
                {detail.explanation.safety_notice && <p className="muted">{detail.explanation.safety_notice}</p>}
              </div>
              {referenceSources.length > 0 && (
                <div className="mini-card">
                  <strong>참고 출처</strong>
                  <div className="chip-list">
                    {referenceSources.map((source) => (
                      <span className="badge badge-reference" key={String(source.id ?? source.title)}>
                        {String(source.source_org ?? source.title ?? "참고 출처")}
                        {source.status ? ` · ${String(source.status)}` : ""}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}
        <Card title="분석 입력 요약">
          <div className="record-table">
            {[
              ["키", snapshotInput.height_cm],
              ["몸무게", snapshotInput.weight_kg],
              ["BMI", snapshotInput.bmi],
              ["혈압", `${String(snapshotInput.systolic_bp ?? "-")} / ${String(snapshotInput.diastolic_bp ?? "-")}`],
              ["공복혈당", snapshotInput.fasting_glucose],
              ["관리 필요 단계", getDisplayRiskLabel(result)],
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
          const content = (
            <>
              <span className={`badge ${getRiskClassName(result)}`}>{getDisplayRiskLabel(result)}</span>
              <div>
                <strong>{labels[String(result.analysis_type)] ?? String(result.analysis_type)}</strong>
                <p className="muted">{mainFactorLabel(result)}</p>
              </div>
              <span>{getDisplayRiskScoreLabel(result)}</span>
              <span>{formatDate(result.analyzed_at ?? result.created_at)}</span>
              <span className="badge badge-reference">{result.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
              <span className="badge badge-reference">상세보기</span>
            </>
          );
          return (
            <Link className="table-row analysis-history-row" key={String(result.id)} to={`/analysis/${String(result.id)}`}>
              {content}
            </Link>
          );
        })}
        {displayResults.length === 0 && (
          <div className="state-box">표시할 분석 결과가 없습니다.</div>
        )}
      </div>
    </Card>
  );
}
