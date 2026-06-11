import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { AnalysisMode, getAnalysisResultDetail, getLatestAnalysisResults, runAnalysisAsync } from "../api/analysis";
import { listChallengeRecommendations, listChallenges } from "../api/challenges";
import { getAnalysisReadiness } from "../api/health";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import RiskStageBoard, { type DiseaseRiskItem } from "../components/RiskStageBoard";
import { useAsyncJobPolling } from "../hooks/useAsyncJobPolling";
import {
  getAnalysisTypeLabel,
  getDisplayRiskLabel,
  getLatestResultsByAnalysisType,
  isKnownAnalysisType,
} from "../utils/riskDisplay";

type AnalysisResult = Record<string, unknown>;
type AnalysisFactor = Record<string, unknown>;
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
type AnalysisDetailPreview = {
  factors?: AnalysisFactor[];
  explanation?: AnalysisExplanation | null;
};
type Readiness = {
  is_ready: boolean;
  basic_ready?: boolean;
  precision_ready?: boolean;
  latest_health_record_id: number | null;
  missing_fields: string[];
  missing_basic_fields?: string[];
  missing_precision_fields?: string[];
  message: string;
};

const asyncJobStatusMessages: Record<string, string> = {
  PENDING: "분석 작업 대기 중입니다.",
  PROCESSING: "분석 중입니다.",
  SUCCESS: "분석이 완료되었습니다.",
  FAILED: "분석에 실패했습니다.",
  CANCELED: "분석 작업이 취소되었습니다.",
};

const missingFieldLabels: Record<string, string> = {
  height_cm: "키",
  weight_kg: "몸무게",
  bmi: "BMI",
  systolic_bp: "수축기 혈압",
  diastolic_bp: "이완기 혈압",
  fasting_glucose: "공복혈당",
  hba1c: "당화혈색소",
  total_cholesterol: "총콜레스테롤",
  triglyceride: "중성지방",
  hdl_cholesterol: "HDL 콜레스테롤",
  ldl_cholesterol: "LDL 콜레스테롤",
  occupation_code: "직업군",
  family_htn: "고혈압 가족력 여부",
  family_dm: "당뇨병 가족력 여부",
  family_dyslipidemia: "콜레스테롤·중성지방 이상 가족력 여부",
  smoking_status: "현재 흡연 여부",
  drinking_frequency: "1년간 음주 빈도",
  drinking_amount: "한 번 음주량",
  walking_days_per_week: "1주일간 걷기 일수",
  strength_days_per_week: "1주일간 근력운동 일수",
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

function displayFactorValue(factor: AnalysisFactor): string {
  const key = String(factor.factor_key ?? "");
  const rawValue = factor.factor_value;
  if (rawValue === undefined || rawValue === null || rawValue === "") {
    return "";
  }
  const value = String(rawValue);
  return factorValueLabels[key]?.[value] ?? value;
}

function buildAnalysisComment(results: AnalysisResult[], explanationsByResultId: Record<string, AnalysisExplanation>): string {
  const firstResult = results[0];
  if (!firstResult) {
    return "건강정보를 입력하고 분석을 실행하면 질환별 위험도와 관리 설명이 표시됩니다.";
  }
  const explanation = explanationsByResultId[String(firstResult.id)];
  if (explanation?.recommended_action) {
    return explanation.recommended_action;
  }
  if (explanation?.summary) {
    return explanation.summary;
  }
  return "분석 결과는 입력된 건강정보 기준의 참고 신호입니다. 상세 화면에서 주요 요인을 확인해보세요.";
}

export default function AnalysisPage() {
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [healthRecordId, setHealthRecordId] = useState<number | null>(null);
  const [basicReady, setBasicReady] = useState(false);
  const [precisionReady, setPrecisionReady] = useState(false);
  const [missingFields, setMissingFields] = useState<string[]>([]);
  const [precisionMissingFields, setPrecisionMissingFields] = useState<string[]>([]);
  const [factorsByResultId, setFactorsByResultId] = useState<Record<string, AnalysisFactor[]>>({});
  const [explanationsByResultId, setExplanationsByResultId] = useState<Record<string, AnalysisExplanation>>({});
  const [recommendedChallenges, setRecommendedChallenges] = useState<AnalysisResult[]>([]);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [runningMode, setRunningMode] = useState<AnalysisMode | null>(null);
  const [analysisJobId, setAnalysisJobId] = useState<number | null>(null);

  const { latestJob, isPolling, pollingError } = useAsyncJobPolling({
    jobId: analysisJobId,
    enabled: analysisJobId !== null && runningMode !== null,
    intervalMs: 1500,
    timeoutMs: 120000,
    onSuccess: async () => {
      await load();
      setNotice("분석이 완료되었습니다.");
      setRunningMode(null);
      setAnalysisJobId(null);
    },
    onFailure: (job) => {
      setError(
        job.status === "CANCELED"
          ? "분석 작업이 취소되었습니다."
          : "분석에 실패했습니다. 잠시 후 다시 시도해주세요.",
      );
      setRunningMode(null);
      setAnalysisJobId(null);
    },
    onTimeout: () => {
      setNotice("분석 시간이 길어지고 있습니다. 잠시 후 결과를 다시 확인해주세요.");
      setRunningMode(null);
      setAnalysisJobId(null);
    },
  });

  const load = async () => {
    const latestResults = await getLatestAnalysisResults<AnalysisResult[]>();
    setResults(latestResults);
    const detailEntries = await Promise.all(
      latestResults.map(async (result) => {
        try {
          const detail = await getAnalysisResultDetail<AnalysisDetailPreview>(Number(result.id));
          return [String(result.id), detail] as const;
        } catch {
          return [String(result.id), { factors: [] as AnalysisFactor[], explanation: null }] as const;
        }
      }),
    );
    setFactorsByResultId(Object.fromEntries(detailEntries.map(([id, detail]) => [id, detail.factors ?? []])));
    setExplanationsByResultId(
      Object.fromEntries(
        detailEntries
          .filter((entry): entry is readonly [string, AnalysisDetailPreview & { explanation: AnalysisExplanation }] =>
            Boolean(entry[1].explanation),
          )
          .map(([id, detail]) => [id, detail.explanation]),
      ),
    );
    const readiness = await getAnalysisReadiness<Readiness>();
    setHealthRecordId(readiness.latest_health_record_id);
    setBasicReady(Boolean(readiness.basic_ready ?? readiness.is_ready));
    setPrecisionReady(Boolean(readiness.precision_ready));
    setMissingFields((readiness.basic_ready ?? readiness.is_ready) ? [] : readiness.missing_basic_fields ?? readiness.missing_fields);
    setPrecisionMissingFields(readiness.missing_precision_fields ?? []);
    const [recommendations, challenges] = await Promise.allSettled([
      listChallengeRecommendations<AnalysisResult[]>({ limit: 3 }),
      listChallenges<AnalysisResult[]>({ limit: 20 }),
    ]);
    if (recommendations.status === "fulfilled" && challenges.status === "fulfilled") {
      const ids = recommendations.value.map((recommendation) => Number(recommendation.challenge_id));
      setRecommendedChallenges(
        ids
          .map((id) => challenges.value.find((challenge) => Number(challenge.id) === id))
          .filter((challenge): challenge is AnalysisResult => Boolean(challenge)),
      );
    }
  };

  useEffect(() => {
    void load().catch(() => setError("분석 결과를 불러오지 못했습니다."));
  }, []);

  const run = async (mode: AnalysisMode) => {
    setError("");
    setNotice("");
    setAnalysisJobId(null);
    setRunningMode(mode);
    try {
      const readiness = await getAnalysisReadiness<Readiness>();
      const currentBasicReady = Boolean(readiness.basic_ready ?? readiness.is_ready);
      const currentPrecisionReady = Boolean(readiness.precision_ready);
      setHealthRecordId(readiness.latest_health_record_id);
      setBasicReady(currentBasicReady);
      setPrecisionReady(currentPrecisionReady);
      setMissingFields(currentBasicReady ? [] : readiness.missing_basic_fields ?? readiness.missing_fields);
      setPrecisionMissingFields(readiness.missing_precision_fields ?? []);
      if (!readiness.latest_health_record_id) {
        setNotice(mode === "PRECISION" ? "건강검진 데이터를 입력해주세요." : "분석을 실행할 건강정보가 없습니다. 먼저 건강정보를 입력해주세요.");
        return;
      }
      if (!currentBasicReady) {
        setNotice("기본 분석에 필요한 정보가 부족해 분석을 실행하지 않았습니다.");
        return;
      }
      if (mode === "PRECISION" && !currentPrecisionReady) {
        setNotice("건강검진 데이터를 입력해주세요.");
        return;
      }
      const job = await runAnalysisAsync(readiness.latest_health_record_id, mode);
      setAnalysisJobId(job.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "분석 실행에 실패했습니다.");
      setRunningMode(null);
    }
  };

  const analysisComment = buildAnalysisComment(results, explanationsByResultId);
  const latestDiseaseResults = getLatestResultsByAnalysisType(results.filter((result) => isKnownAnalysisType(result.analysis_type)));
  const diseaseRiskItems: DiseaseRiskItem[] = latestDiseaseResults
    .map((result) => ({
      analyzed_at: result.analyzed_at,
      created_at: result.created_at,
      diseaseName: getAnalysisTypeLabel(result.analysis_type),
      id: result.id,
      risk_level: result.risk_level,
      service_band: result.service_band,
      service_band_label: result.service_band_label,
    }));
  const jobStatusMessage = latestJob
    ? asyncJobStatusMessages[latestJob.status] ?? "분석 작업 상태를 확인하고 있습니다."
    : analysisJobId
      ? "분석 작업 대기 중입니다."
      : "";
  const showJobStatus = analysisJobId !== null && (isPolling || Boolean(jobStatusMessage));

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>건강 분석 결과</h1>
          <p>당뇨, 고혈압, 비만, 콜레스테롤·중성지방 이상 위험도를 한 화면에서 확인합니다.</p>
        </div>
        <div className="button-row">
          <button disabled={runningMode !== null} onClick={() => void run("BASIC")} type="button">
            {runningMode === "BASIC" ? "간편 분석 중..." : "간편 분석 실행"}
          </button>
          <button disabled={runningMode !== null} onClick={() => void run("PRECISION")} type="button">
            {runningMode === "PRECISION" ? "정밀 분석 중..." : "정밀 분석 실행"}
          </button>
          <Link className="button secondary" to="/analysis/history">
            전체 결과
          </Link>
        </div>
      </div>
      {error && <ErrorMessage message={error} />}
      {pollingError && <ErrorMessage message={pollingError.message} />}
      {notice && <div className="state-box">{notice}</div>}
      {showJobStatus && (
        <div className="state-box">
          {jobStatusMessage}
          {latestJob?.status && latestJob.status !== "SUCCESS" && latestJob.status !== "FAILED" && (
            <span className="muted"> 현재 상태: {latestJob.status}</span>
          )}
        </div>
      )}
      {missingFields.length > 0 && (
        <Card title="분석에 필요한 정보가 부족합니다">
          <div className="readiness-card">
            <p>직업군, 가족력, 신장, 체중, 흡연/음주/운동 정보를 입력하면 기본 위험도 분석을 실행할 수 있습니다.</p>
            <div className="chip-list">
              {missingFields.map((field) => (
                <span className="badge badge-missing" key={field}>
                  {missingFieldLabels[field] ?? field}
                </span>
              ))}
            </div>
            <Link className="button" to="/health/profile">
              필수 건강정보 입력하기
            </Link>
          </div>
        </Card>
      )}
      {precisionMissingFields.length > 0 && (
        <Card title="정밀 분석 선택 입력">
          <div className="readiness-card">
            <p>정밀 분석은 혈압/혈당/지질/허리둘레 등 검진 수치를 함께 반영합니다.</p>
            <p>부족한 검진값을 입력하면 정밀 분석을 실행할 수 있습니다.</p>
            <div className="chip-list">
              {precisionMissingFields.map((field) => (
                <span className="badge badge-reference" key={field}>
                  {missingFieldLabels[field] ?? field}
                </span>
              ))}
            </div>
            <Link className="button secondary" to="/health/profile">
              검진값 입력하기
            </Link>
          </div>
        </Card>
      )}
      <div className="page-grid">
        <Card title="분석 기반 건강 코멘트">
          <p>{analysisComment}</p>
        </Card>
      </div>
      <Card title="질환별 위험도">
        <RiskStageBoard items={diseaseRiskItems} />
      </Card>
      <div className="metric-grid">
        {latestDiseaseResults.map((result) => {
          const explanation = explanationsByResultId[String(result.id)];
          const referenceSources = explanation?.reference_sources ?? [];
          return (
            <div className="metric-card card" key={String(result.id)}>
              <span>{getAnalysisTypeLabel(result.analysis_type)} 관리 필요 단계</span>
              <strong>{getDisplayRiskLabel(result)}</strong>
              <span className="badge badge-reference">{result.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
              <p>{String(result.summary ?? "주요 factor는 상세 화면에서 확인할 수 있습니다.")}</p>
              {explanation?.reference_summary && <p className="muted">{explanation.reference_summary}</p>}
              {referenceSources.length > 0 && (
                <div className="chip-list">
                  {referenceSources.slice(0, 2).map((source) => (
                    <span className="badge badge-reference" key={String(source.id ?? source.title)}>
                      {String(source.source_org ?? source.title ?? "참고 출처")}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
        {latestDiseaseResults.length === 0 && (
          <div className="metric-card card">
            <span>분석 결과 없음</span>
            <strong>-</strong>
            <p>건강정보를 입력한 뒤 간편 분석을 실행하면 위험도 결과 카드가 표시됩니다.</p>
          </div>
        )}
      </div>
      <Card title="추천 액션">
        {recommendedChallenges.length > 0 ? (
          <div className="card-list">
            {recommendedChallenges.map((challenge) => (
              <div className="mini-card" key={String(challenge.id)}>
                <strong>{String(challenge.title ?? "추천 챌린지")}</strong>
                <p className="muted">{String(challenge.description ?? "분석 결과를 바탕으로 추천된 챌린지입니다.")}</p>
                <Link className="button secondary" to={`/challenges/${String(challenge.id)}`}>
                  상세보기
                </Link>
              </div>
            ))}
          </div>
        ) : (
          <div className="button-row">
            <Link className="button secondary" to="/challenges">
              챌린지 보기
            </Link>
            <Link className="button secondary" to="/diets">
              식단 이미지 분석
            </Link>
            <Link className="button secondary" to="/dashboard">
              추적 대시보드 이동
            </Link>
          </div>
        )}
      </Card>
      <Card title="상세 요인 카드">
        <div className="card-list">
          {latestDiseaseResults.map((result) => {
            const factors = factorsByResultId[String(result.id)] ?? [];
            return (
              <div className="mini-card" key={String(result.id)}>
                <strong>{getAnalysisTypeLabel(result.analysis_type)}</strong>
                {factors.length > 0 ? (
                  <div className="chip-list">
                    {factors.slice(0, 4).map((factor) => {
                      const value = displayFactorValue(factor);
                      return (
                        <span className="badge badge-reference" key={String(factor.id ?? factor.factor_key)}>
                          {String(factor.factor_name ?? factor.factor_key)}
                          {value ? `: ${value}` : ""}
                        </span>
                      );
                    })}
                  </div>
                ) : (
                  <span>상세 요인은 분석 입력값과 질환별 기준에 따라 표시됩니다.</span>
                )}
                <Link to={`/analysis/${String(result.id)}`}>상세보기</Link>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
