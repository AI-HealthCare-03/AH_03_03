import { useMemo, useEffect, useState } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";

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

type PrecisionMeasurementItem = {
  key: string;
  label: string;
  unit?: string;
};

const analysisTypeOptions: Record<string, string | null> = {
  "전체": null,
  "고혈압": "HYPERTENSION",
  "당뇨": "DIABETES",
  "이상지질혈증": "DYSLIPIDEMIA",
  "비만": "OBESITY",
  "복부비만": "ABDOMINAL_OBESITY",
  "지방간": "FATTY_LIVER",
  "빈혈": "ANEMIA",
  "간기능": "LIVER_FUNCTION",
  "신장기능": "KIDNEY_FUNCTION",
  "만성콩팥병": "CHRONIC_KIDNEY_DISEASE",
};

const precisionMeasurementItems: PrecisionMeasurementItem[] = [
  { key: "ast", label: "AST", unit: "U/L" },
  { key: "alt", label: "ALT", unit: "U/L" },
  { key: "gamma_gtp", label: "감마GTP", unit: "U/L" },
  { key: "creatinine", label: "크레아티닌", unit: "mg/dL" },
  { key: "egfr", label: "eGFR", unit: "mL/min/1.73m²" },
  { key: "hemoglobin", label: "혈색소", unit: "g/dL" },
];

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

function getSnapshotX2InputPayload(snapshot: AnalysisResult | null | undefined): Record<string, unknown> {
  if (!isRecord(snapshot?.input_payload)) {
    return {};
  }
  const inputPayload = snapshot.input_payload;
  return isRecord(inputPayload.x2_input_payload) ? inputPayload.x2_input_payload : {};
}

function getSnapshotX2FieldSources(snapshot: AnalysisResult | null | undefined): Record<string, unknown> {
  if (!isRecord(snapshot?.input_payload)) {
    return {};
  }
  const inputPayload = snapshot.input_payload;
  return isRecord(inputPayload.x2_field_sources) ? inputPayload.x2_field_sources : {};
}

function getSnapshotStringValue(snapshot: AnalysisResult | null | undefined, key: string): string | null {
  if (!isRecord(snapshot?.input_payload)) {
    return null;
  }
  const value = snapshot.input_payload[key];
  return value === undefined || value === null || value === "" ? null : String(value);
}

function formatSummaryValue(value: unknown, unit = ""): string {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return `${String(value)}${unit}`;
}

function formatPrecisionMeasurementValue(value: unknown, unit = ""): string {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return `${String(value)}${unit ? ` ${unit}` : ""}`;
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
  const navigate = useNavigate();
  const { analysisId } = useParams();
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [activeTab, setActiveTab] = useState("전체");
  const [activeMode, setActiveMode] = useState("전체");
  const [error, setError] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const formatDate = (value: unknown) => {
    if (!value) {
      return "-";
    }
    const date = new Date(String(value));
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString("ko-KR");
  };

  const displayResults = useMemo(() => {
    let filtered = results;
    const selectedType = analysisTypeOptions[activeTab];
    if (selectedType) {
      filtered = filtered.filter((result) => String(result.analysis_type) === selectedType);
    }
    if (activeMode === "간편") {
      filtered = filtered.filter((result) => result.analysis_mode === "BASIC");
    } else if (activeMode === "정밀") {
      filtered = filtered.filter((result) => result.analysis_mode === "PRECISION");
    }
    return filtered;
  }, [activeTab, activeMode, results]);

  const totalPages = Math.ceil(displayResults.length / itemsPerPage);
  const pagedResults = displayResults.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  const mainFactorLabel = (result: AnalysisResult) => {
    const summary = result.summary ?? result.guide_message ?? result.message;
    if (summary) {
      return String(summary);
    }
    return "상세보기에서 주요 요인을 확인할 수 있습니다.";
  };
  const diseaseRiskItems: DiseaseRiskItem[] = getLatestResultsByAnalysisType(
    results.filter((result) => isKnownAnalysisType(result.analysis_type)),
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
  const snapshotX2Input = getSnapshotX2InputPayload(detail?.snapshot);
  const snapshotX2Sources = getSnapshotX2FieldSources(detail?.snapshot);
  const referenceSources = detail?.explanation?.reference_sources ?? [];
  const visiblePrecisionMeasurements = precisionMeasurementItems.filter((item) => {
    const value = snapshotX2Input[item.key];
    return value !== undefined && value !== null && value !== "";
  });
  const x2MeasurementSource = getSnapshotStringValue(detail?.snapshot, "x2_measurement_source");
  const selectedExamReportId = getSnapshotStringValue(detail?.snapshot, "selected_exam_report_id");

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
            <Link className="button" to="/analysis/history">
              전체 리스트
            </Link>
          }
        >
          <div className={`score-panel risk-detail-panel ${detailRiskClassName}`}>
            <span>{getAnalysisTypeLabel(result?.analysis_type, "분석")}</span>
            <strong>{getDisplayRiskLabel(result)}</strong>
            <div className="button-row">
              <span className="badge badge-reference">{result?.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
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
          {visiblePrecisionMeasurements.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ marginBottom: 8, marginTop: 16 }}>분석에 사용된 검진 수치</h3>
              <div className="state-box">
                <p style={{ margin: "0 0 8px" }}>
                  정밀분석에 반영된 검진표 기반 수치입니다. 건강정보 수기 입력 항목에 저장되지 않은 값도 검진표 확정 결과에서 보강될 수 있습니다.
                </p>
                <div className="chip-list">
                  {x2MeasurementSource && (
                    <span className="badge badge-reference">
                      출처 {x2MeasurementSource === "exam_measurements" ? "검진표 OCR" : x2MeasurementSource}
                    </span>
                  )}
                  {selectedExamReportId && (
                    <span className="badge badge-reference">검진표 #{selectedExamReportId}</span>
                  )}
                </div>
              </div>
              <div className="readonly-health-grid" style={{ marginTop: 12 }}>
                {visiblePrecisionMeasurements.map((item) => (
                  <div className="readonly-health-item" key={item.key}>
                    <div className="item-header">
                      <span>{item.label}</span>
                      {snapshotX2Sources[item.key] === "exam_measurements" && (
                        <em className="badge badge-reference">OCR</em>
                      )}
                    </div>
                    <div className="item-value-row">
                      <strong>{formatPrecisionMeasurementValue(snapshotX2Input[item.key], item.unit)}</strong>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
        {detailSlots.length > 0 && (
          <Card title={result?.analysis_mode === "PRECISION" ? "정밀분석 질환별 예측 결과" : "간편분석 질환별 예측 결과"}>
            <p className="muted" style={{ marginBottom: 12 }}>질환별 카드를 클릭하시면 해당 질병의 분석 결과를 바로 확인하실 수 있습니다.</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
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
                      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 4 }}>
                        <Link className="btn-primary" style={{ fontSize: "13px", padding: "4px 12px" }} to="/health/profile">
                          추가 정보 입력
                        </Link>
                      </div>
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
      title="위험도 예측 결과 전체 리스트"
      actions={
        <button className="button" onClick={() => navigate(-1)} type="button">
          뒤로가기
        </button>
      }
    >
      {error && <ErrorMessage message={error} />}

      <div style={{ display: "flex", justifyContent: "flex-start", gap: 8, marginBottom: 8 }}>
        <select
          onChange={(e) => { setActiveTab(e.target.value); setCurrentPage(1); }}
          style={{ fontSize: 13, padding: "6px 4px", border: "none", borderBottom: "1.5px solid var(--color-border-secondary)", background: "transparent", color: "var(--color-text-primary)", width: 200, cursor: "pointer", outline: "none"  }}
          value={activeTab}
        >
          {Object.keys(analysisTypeOptions).map((key) => (
            <option key={key} value={key}>{key}</option>
          ))}
        </select>
        <select
          onChange={(e) => { setActiveMode(e.target.value); setCurrentPage(1); }}
          style={{ fontSize: 13, padding: "6px 4px", border: "none", borderBottom: "1.5px solid var(--color-border-secondary)", background: "transparent", color: "var(--color-text-primary)", width: 100, cursor: "pointer", outline: "none"  }}
          value={activeMode}
        >
          <option>전체</option>
          <option>간편</option>
          <option>정밀</option>
        </select>
      </div>
      {diseaseRiskItems.length > 0 && <RiskStageBoard items={diseaseRiskItems} />}

      <div className="table-list" style={{ marginTop: 16 }}>
        {pagedResults.map((result) => {
          const sourceBadgeLabel = getAnalysisSourceBadgeLabel(result);
          const content = (
            <>
              <div>

                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <span className={`badge ${getRiskClassName(result)}`}>{getDisplayRiskLabel(result)}</span>
                  <strong>{getAnalysisTypeLabel(result.analysis_type)}</strong>
                </div>
                <p className="muted">{mainFactorLabel(result)}</p>
              </div>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "space-between", minWidth: 120 }}>
                <span style={{ color: "var(--color-muted)", fontSize: 13 }}>상세보기 →</span>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span className="muted" style={{ fontSize: 13 }}>{formatDate(result.analyzed_at ?? result.created_at)}</span>
                  <span className="badge badge-reference">{result.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
                  {sourceBadgeLabel && <span className="badge badge-reference">{sourceBadgeLabel}</span>}
                </div>
              </div>
            </>
          );
          return (
            <Link className="table-row analysis-history-row" key={String(result.id)} to={`/analysis/${String(result.id)}`}>
              {content}
            </Link>
          );
        })}
        {pagedResults.length === 0 && (
          <div className="state-box">표시할 분석 결과가 없습니다.</div>
        )}
      </div>
      {totalPages > 1 && (
        <div className="button-row" style={{ justifyContent: "center", marginTop: 16 }}>
          <button className="secondary" disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)} type="button">
            이전
          </button>
          <span>{currentPage} / {totalPages}</span>
          <button className="secondary" disabled={currentPage === totalPages} onClick={() => setCurrentPage(p => p + 1)} type="button">
            다음
          </button>
        </div>
      )}
    </Card>
  );
}
