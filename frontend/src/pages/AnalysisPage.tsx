import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { AnalysisMode, getAnalysisResultDetail, getLatestAnalysisResults, runAnalysis } from "../api/analysis";
import { listChallengeRecommendations, listChallenges } from "../api/challenges";
import { getAnalysisReadiness } from "../api/health";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

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

function modelLabel(result: AnalysisResult): string | null {
  const modelName = result.model_name ? String(result.model_name) : "";
  const modelVersion = result.model_version ? String(result.model_version) : "";
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
  hdl_cholesterol: "HDL",
  ldl_cholesterol: "LDL",
  occupation_code: "직업군",
  family_htn: "고혈압 가족력 여부",
  family_dm: "당뇨병 가족력 여부",
  family_dyslipidemia: "이상지질혈증 가족력 여부",
  smoking_status: "현재 흡연 여부",
  drinking_frequency: "1년간 음주 빈도",
  drinking_amount: "한 번 음주량",
  walking_days_per_week: "1주일간 걷기 일수",
  strength_days_per_week: "1주일간 근력운동 일수",
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
  const [error, setError] = useState("");
  const [runningMode, setRunningMode] = useState<AnalysisMode | null>(null);

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
        setError("분석을 실행할 건강정보가 없습니다. 먼저 건강정보를 입력해주세요.");
        return;
      }
      if (!currentBasicReady) {
        setError("기본 분석에 필요한 정보가 부족해 분석을 실행하지 않았습니다.");
        return;
      }
      if (mode === "PRECISION" && !currentPrecisionReady) {
        setError("정밀 분석에 필요한 검진값이 부족해 분석을 실행하지 않았습니다.");
        return;
      }
      await runAnalysis(readiness.latest_health_record_id, mode);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "분석 실행에 실패했습니다.");
    } finally {
      setRunningMode(null);
    }
  };

  const overallLevel = getOverallLevel(results);
  const overallScore = getOverallScore(results);
  const analysisComment = buildAnalysisComment(results, explanationsByResultId);

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>건강 분석 결과</h1>
          <p>당뇨, 고혈압, 비만, 이상지질혈증 위험도를 한 화면에서 확인합니다.</p>
        </div>
        <div className="button-row">
          <button disabled={!basicReady || runningMode !== null} onClick={() => void run("BASIC")}>
            {runningMode === "BASIC" ? "간편 분석 중..." : "간편 분석 실행"}
          </button>
          <button disabled={!basicReady || !precisionReady || runningMode !== null} onClick={() => void run("PRECISION")}>
            {runningMode === "PRECISION" ? "정밀 분석 중..." : "정밀 분석 실행"}
          </button>
          <Link className="button secondary" to="/analysis/history">
            전체 결과
          </Link>
        </div>
      </div>
      {error && <ErrorMessage message={error} />}
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
        <Card title="종합 위험도">
          <div className="score-panel">
            <span>종합 위험도 게이지</span>
            <strong>{overallLevel}</strong>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${overallScore}%` }} />
            </div>
            <p>이 결과는 저장된 건강정보와 검진값을 바탕으로 계산한 건강관리 참고 신호이며 실제 의료 진단이 아닙니다.</p>
          </div>
        </Card>
        <Card title="분석 기반 건강 코멘트">
          <p>{analysisComment}</p>
        </Card>
      </div>
      <div className="metric-grid">
        {results.map((result) => {
          const level = getRiskLevel(result);
          const score = getRiskScore(result);
          const explanation = explanationsByResultId[String(result.id)];
          const referenceSources = explanation?.reference_sources ?? [];
          return (
            <div className="metric-card card" key={String(result.id)}>
              <span>{analysisTypeLabels[String(result.analysis_type)] ?? String(result.analysis_type)} 위험도</span>
              <strong>{score}/100</strong>
              <span className={`badge risk-${level.toLowerCase()}`}>{level || "-"}</span>
              <span className="badge badge-reference">{result.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
              {modelLabel(result) && <span className="badge badge-reference">{modelLabel(result)}</span>}
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
        {results.length === 0 && (
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
          {results.map((result) => {
            const factors = factorsByResultId[String(result.id)] ?? [];
            return (
              <div className="mini-card" key={String(result.id)}>
                <strong>{analysisTypeLabels[String(result.analysis_type)] ?? String(result.analysis_type)}</strong>
                {factors.length > 0 ? (
                  <div className="chip-list">
                    {factors.slice(0, 4).map((factor) => (
                      <span className="badge badge-reference" key={String(factor.id ?? factor.factor_key)}>
                        {String(factor.factor_name ?? factor.factor_key)}
                        {factor.factor_value ? `: ${String(factor.factor_value)}` : ""}
                      </span>
                    ))}
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
