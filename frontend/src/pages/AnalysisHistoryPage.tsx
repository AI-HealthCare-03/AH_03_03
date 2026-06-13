import { useMemo, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getAnalysisResultDetail, listAnalysisResults } from "../api/analysis";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import RiskStageBoard, { type DiseaseRiskItem } from "../components/RiskStageBoard";
import {
  getAnalysisSourceBadgeLabel,
  getAnalysisTypeLabel,
  getDisplayRiskLabel,
  getExpectedAnalysisTypesByMode,
  getLatestResultsByAnalysisType,
  getRiskClassName,
  getX2StageSummary,
  isKnownAnalysisType,
  mergeResultsWithExpectedAnalysisTypes,
} from "../utils/riskDisplay";

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

const tabs = ["전체", "당뇨", "고혈압", "콜레스테롤·중성지방", "비만"];
const tabToType: Record<string, string | null> = {
  전체: null,
  당뇨: "DIABETES",
  고혈압: "HYPERTENSION",
  "콜레스테롤·중성지방": "DYSLIPIDEMIA",
  비만: "OBESITY",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function getSnapshotInputFeatures(snapshot: AnalysisResult | null | undefined): Record<string, unknown> {
  if (!isRecord(snapshot?.input_payload)) {
    return {};
  }
  const inputPayload = snapshot.input_payload;
  return isRecord(inputPayload.input_features) ? inputPayload.input_features : inputPayload;
}

function formatSummaryValue(value: unknown, unit = ""): string {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return `${String(value)}${unit}`;
}

function formatBloodPressure(input: Record<string, unknown>): string {
  const systolic = input.systolic_bp;
  const diastolic = input.diastolic_bp;
  if ((systolic === undefined || systolic === null || systolic === "") && (diastolic === undefined || diastolic === null || diastolic === "")) {
    return "-";
  }
  return `${String(systolic ?? "-")} / ${String(diastolic ?? "-")} mmHg`;
}

function getResultTimestamp(result: AnalysisResult | null | undefined): number {
  const candidates = [result?.analyzed_at, result?.created_at, result?.analyzedAt, result?.createdAt];
  for (const candidate of candidates) {
    const parsed = Date.parse(String(candidate ?? ""));
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
}

function getRelatedAnalysisRunResults(target: AnalysisResult | null | undefined, allResults: AnalysisResult[]): AnalysisResult[] {
  if (!target) {
    return [];
  }
  const targetJobId = target.async_job_id;
  const targetMode = String(target.analysis_mode ?? "");
  const targetTimestamp = getResultTimestamp(target);
  const related = [target, ...allResults].filter((result) => {
    if (!isKnownAnalysisType(result.analysis_type)) {
      return false;
    }
    if (targetJobId !== undefined && targetJobId !== null && targetJobId !== "") {
      return String(result.async_job_id ?? "") === String(targetJobId);
    }
    if (!targetTimestamp) {
      return String(result.id ?? "") === String(target.id ?? "");
    }
    return String(result.analysis_mode ?? "") === targetMode && Math.abs(getResultTimestamp(result) - targetTimestamp) <= 120000;
  });
  const deduped = new Map<string, AnalysisResult>();
  related.forEach((result) => {
    deduped.set(String(result.id ?? `${String(result.analysis_type)}-${getResultTimestamp(result)}`), result);
  });
  return Array.from(deduped.values());
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
  const diseaseRiskItems: DiseaseRiskItem[] = getLatestResultsByAnalysisType(
    displayResults.filter((result) => isKnownAnalysisType(result.analysis_type)),
  )
    .map((result) => ({
      analyzed_at: result.analyzed_at,
      created_at: result.created_at,
      diseaseName: getAnalysisTypeLabel(result.analysis_type),
      id: result.id,
      risk_level: result.risk_level,
      service_band: result.service_band,
      service_band_label: result.service_band_label,
    }));

  const snapshotInput = getSnapshotInputFeatures(detail?.snapshot);
  const referenceSources = detail?.explanation?.reference_sources ?? [];

  useEffect(() => {
    const load = async () => {
      setError("");
      if (analysisId) {
        const [nextDetail, nextResults] = await Promise.all([
          getAnalysisResultDetail<AnalysisDetail>(Number(analysisId)),
          listAnalysisResults<AnalysisResult[]>(),
        ]);
        setDetail(nextDetail);
        setResults(nextResults);
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
    const detailRiskClassName = getRiskClassName(result);
    const sourceBadgeLabel = getAnalysisSourceBadgeLabel(result);
    const x2StageSummary = getX2StageSummary(result);
    const relatedResults = getRelatedAnalysisRunResults(result, results);
    const detailSlots = result
      ? mergeResultsWithExpectedAnalysisTypes(relatedResults, result.analysis_mode)
      : [];
    const expectedSlotCount = result ? getExpectedAnalysisTypesByMode(result.analysis_mode).length : 0;
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
          <div className={`score-panel risk-detail-panel ${detailRiskClassName}`}>
            <span>{getAnalysisTypeLabel(result?.analysis_type, "분석")}</span>
            <strong>{getDisplayRiskLabel(result)}</strong>
            <div className="button-row">
              <span className={`badge ${detailRiskClassName}`}>{result?.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
              {sourceBadgeLabel && <span className="badge badge-reference">{sourceBadgeLabel}</span>}
            </div>
            <p>{String(result?.summary ?? "")}</p>
            {x2StageSummary && <p className="muted">{x2StageSummary}</p>}
          </div>
          {detail?.explanation && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ marginBottom: 8, marginTop: 16 }}>분석 설명</h3>
              <div className="mini-card">
                <strong>{detail.explanation.summary ?? "분석 설명"}</strong>
                {detail.explanation.caution && <p className="muted">{detail.explanation.caution}</p>}
                {detail.explanation.recommended_action && <p>{detail.explanation.recommended_action}</p>}
                {detail.explanation.reference_summary && <p className="muted">{detail.explanation.reference_summary}</p>}
                {detail.explanation.safety_notice && <p className="muted">{detail.explanation.safety_notice}</p>}
              </div>
            </div>
          )}
        </Card>
        {detailSlots.length > 0 && (
          <Card title={result?.analysis_mode === "PRECISION" ? "정밀분석 질환별 판정" : "간편분석 질환별 판정"}>
            <div className="card-list">
              {detailSlots.map((slot) => {
                if (slot.isUnavailable || !slot.result) {
                  return (
                    <div className="mini-card" key={slot.analysisType}>
                      <strong>{slot.diseaseName} 관리 단계</strong>
                      <div className="button-row">
                        <span className="badge badge-missing">검진 수치 부족</span>
                        <span className="badge badge-reference">판정 불가</span>
                      </div>
                      <p className="muted">{slot.unavailableReason}</p>
                    </div>
                  );
                }
                const slotRiskClassName = getRiskClassName(slot.result);
                const slotSourceBadgeLabel = getAnalysisSourceBadgeLabel(slot.result);
                const slotX2StageSummary = getX2StageSummary(slot.result);
                return (
                  <Link to={`/analysis/${String(slot.result.id)}`} style={{ textDecoration: "none" }}>
                    <div className="mini-card" key={String(slot.result.id ?? slot.analysisType)}>
                      <strong>{slot.diseaseName} 관리 단계</strong>
                      <div className="button-row">
                        <span className={`badge ${slotRiskClassName}`}>{getDisplayRiskLabel(slot.result)}</span>
                        <span className="badge badge-reference">{slot.result.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
                        {slotSourceBadgeLabel && <span className="badge badge-reference">{slotSourceBadgeLabel}</span>}
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
            {result?.analysis_mode === "PRECISION" && detailSlots.length < expectedSlotCount && (
              <div className="state-box">정밀분석 표시 항목을 불러오는 중입니다.</div>
            )}
          </Card>
        )}
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
      {diseaseRiskItems.length > 0 && <RiskStageBoard items={diseaseRiskItems} />}
      <div className="table-list">
        {displayResults.map((result) => {
          const sourceBadgeLabel = getAnalysisSourceBadgeLabel(result);
          const content = (
            <>
              <span className={`badge ${getRiskClassName(result)}`}>{getDisplayRiskLabel(result)}</span>
              <div>
                <strong>{getAnalysisTypeLabel(result.analysis_type)}</strong>
                <p className="muted">{mainFactorLabel(result)}</p>
              </div>
              <span>{formatDate(result.analyzed_at ?? result.created_at)}</span>
              <span className="badge badge-reference">{result.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
              {sourceBadgeLabel && <span className="badge badge-reference">{sourceBadgeLabel}</span>}
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
