import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { getAnalysisResultDetail, getLatestAnalysisResults } from "../api/analysis";
import { getChallenge, listChallengeRecommendations } from "../api/challenges";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import RiskStageBoard, { type DiseaseRiskItem } from "../components/RiskStageBoard";
import {
  getAnalysisTypeLabel,
  getDisplayRiskLabel,
  getLatestAnalysisMode,
  getLatestResultsByAnalysisType,
  getRiskColor,
  isKnownAnalysisType,
  mergeResultsWithExpectedAnalysisTypes,
} from "../utils/riskDisplay";

import { categoryIcon } from "./ChallengePage";

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

function getChallengeCardDescription(value: unknown): string {
  const cleaned = String(value ?? "")
    .replace(/\[target_disease=[A-Z_]+\]\s*/g, "")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const fallback = cleaned || "분석 결과를 바탕으로 추천된 챌린지입니다.";
  return fallback.length > 140 ? `${fallback.slice(0, 139).trim()}…` : fallback;
}

const COMMON_ANALYSIS_NOTICE =
  "이 결과는 입력된 건강정보를 바탕으로 생활관리 방향을 안내하는 참고 자료입니다. 정확한 진단과 치료는 의료진 상담을 통해 확인해 주세요.";

function cleanAnalysisText(value: unknown, fallback = "상세보기에서 주요 요인을 확인할 수 있습니다."): string {
  const cleaned = String(value ?? "")
    .replace(/이 결과는\s*(간편|정밀)?\s*분석 참고용 판정이며 의료 진단이 아닙니다\.?/g, "")
    .replace(/이 설명은 진단이 아니며[^.。]*[.。]?/g, "")
    .replace(/정확한 진단과 치료는 의료진 상담이 필요합니다\.?/g, "")
    .replace(/참고 정보:\s*/g, "")
    .replace(/출처:\s*/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned || fallback;
}

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
    return cleanAnalysisText(explanation.recommended_action);
  }
  if (explanation?.summary) {
    return cleanAnalysisText(explanation.summary);
  }
  return "분석 결과는 입력된 건강정보 기준의 참고 신호입니다. 상세 화면에서 주요 요인을 확인해보세요.";
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

function getResultId(result: AnalysisResult | null | undefined): number {
  const parsed = Number(result?.id);
  return Number.isFinite(parsed) ? parsed : 0;
}

function getMostRecentAnalysisRunResults(items: AnalysisResult[]): AnalysisResult[] {
  const latest = items.reduce<AnalysisResult | null>((current, item) => {
    if (!current) {
      return item;
    }
    const itemTimestamp = getResultTimestamp(item);
    const currentTimestamp = getResultTimestamp(current);
    if (itemTimestamp !== currentTimestamp) {
      return itemTimestamp > currentTimestamp ? item : current;
    }
    return getResultId(item) > getResultId(current) ? item : current;
  }, null);
  if (!latest) {
    return [];
  }
  const latestJobId = latest.async_job_id;
  const latestMode = String(latest.analysis_mode ?? "");
  const latestTimestamp = getResultTimestamp(latest);
  return items.filter((item) => {
    if (latestJobId !== undefined && latestJobId !== null && latestJobId !== "") {
      return String(item.async_job_id ?? "") === String(latestJobId);
    }
    if (!latestTimestamp) {
      return String(item.id ?? "") === String(latest.id ?? "");
    }
    return String(item.analysis_mode ?? "") === latestMode && Math.abs(getResultTimestamp(item) - latestTimestamp) <= 120000;
  });
}

export default function AnalysisPage() {
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [factorsByResultId, setFactorsByResultId] = useState<Record<string, AnalysisFactor[]>>({});
  const [explanationsByResultId, setExplanationsByResultId] = useState<Record<string, AnalysisExplanation>>({});
  const [recommendedChallenges, setRecommendedChallenges] = useState<AnalysisResult[]>([]);
  const [recommendedChallengesLoading, setRecommendedChallengesLoading] = useState(false);
  const [error, setError] = useState("");

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
    setRecommendedChallengesLoading(true);
    try {
      const recommendations = await listChallengeRecommendations<AnalysisResult[]>({ limit: 4 });
      const ids = Array.from(
        new Set(
          recommendations
            .map((recommendation) => Number(recommendation.challenge_id))
            .filter((id) => Number.isFinite(id) && id > 0),
        ),
      );
      if (ids.length > 0) {
        const challengeDetails = await Promise.allSettled(ids.map((id) => getChallenge<AnalysisResult>(id)));
        challengeDetails.forEach((result, index) => {
          if (result.status === "rejected") {
            console.warn("추천 챌린지 상세를 불러오지 못했습니다.", { challenge_id: ids[index] });
          }
        });
        setRecommendedChallenges(
          challengeDetails
            .map((result) => (result.status === "fulfilled" ? result.value : null))
            .filter((challenge): challenge is AnalysisResult => Boolean(challenge))
            .slice(0, 4),
        );
      } else {
        setRecommendedChallenges([]);
      }
    } catch {
      console.warn("추천 챌린지 목록을 불러오지 못했습니다.");
      setRecommendedChallenges([]);
    } finally {
      setRecommendedChallengesLoading(false);
    }
  };

  useEffect(() => {
    void load().catch(() => setError("분석 결과를 불러오지 못했습니다."));
  }, []);

  const analysisComment = buildAnalysisComment(results, explanationsByResultId);
  const latestRunResults = getMostRecentAnalysisRunResults(results.filter((result) => isKnownAnalysisType(result.analysis_type)));
  const latestDiseaseResults = getLatestResultsByAnalysisType(latestRunResults);
  const displayMode = getLatestAnalysisMode(latestDiseaseResults);
  const analysisSlots = latestDiseaseResults.length > 0 ? mergeResultsWithExpectedAnalysisTypes(latestDiseaseResults, displayMode) : [];
  const availableDiseaseResults = analysisSlots
    .filter((slot) => !slot.isUnavailable && slot.result)
    .map((slot) => slot.result as AnalysisResult);
  const diseaseRiskItems: DiseaseRiskItem[] = availableDiseaseResults
    .map((result) => ({
      analyzed_at: result.analyzed_at,
      created_at: result.created_at,
      diseaseName: getAnalysisTypeLabel(result.analysis_type),
      id: result.id,
      risk_level: result.risk_level,
      service_band: result.service_band,
      service_band_label: result.service_band_label,
    }));

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>건강 분석 결과</h1>
          <p>나의 만성질환 위험도 예측 결과를 한눈에 확인해 보세요.</p>
        </div>
        <div className="button-row">
          <Link className="button secondary" to="/health/profile">
            정보 수정
          </Link>
          <Link className="button" to="/analysis/history">
            전체 결과
          </Link>
        </div>
      </div>
      {error && <ErrorMessage message={error} />}
      <div className="page-grid">
        <Card title="분석결과 기반 AI 건강 코멘트">
          <p>{analysisComment}</p>
        </Card>
      </div>
      <Card title="질환별 위험도">
        <RiskStageBoard items={diseaseRiskItems} />
      </Card>
      <div className="state-box analysis-common-notice">{COMMON_ANALYSIS_NOTICE}</div>
      <div className="metric-grid">
        {analysisSlots.map((slot) => {
          if (slot.isUnavailable || !slot.result) {
            return (
              <div className="metric-card card" key={slot.analysisType}>
                <span>{slot.diseaseName} 관리 단계</span>
                <strong>검진 수치 부족</strong>
                <div className="button-row">
                  <span className="badge badge-missing">판정 불가</span>
                </div>
                <p>{slot.unavailableReason}</p>
              </div>
            );
          }
          const result = slot.result;
          return (
            <div className="metric-card card" key={String(result.id)}>
              <span>{slot.diseaseName} 위험도 예측 결과</span>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <strong style={{ color: getRiskColor(result) }}>{getDisplayRiskLabel(result)}</strong>
                <div style={{ display: "flex", gap: 6 }}>
                  <span className="badge badge-reference">{result.analysis_mode === "PRECISION" ? "정밀" : "간편"}</span>
                </div>
              </div>
              <p>{cleanAnalysisText(result.summary)}</p>
              <Link className="button secondary" style={{ marginTop: 8, textAlign: "center", display: "block" }} to={`/analysis/${String(result.id)}`}>
                상세보기
              </Link>
              <Link
                className="button secondary"
                style={{ marginTop: 8, textAlign: "center", display: "block" }}
                to={`/chatbot?context_type=ANALYSIS&target_id=${String(result.id)}&initial_question=${encodeURIComponent("이 분석 결과에서 먼저 관리할 점을 알려줘")}`}
              >
                AI에게 질문하기
              </Link>
            </div>
          );
        })}
        {analysisSlots.length === 0 && (
          <div className="metric-card card">
            <span>분석 결과 없음</span>
            <strong>-</strong>
            <p>건강정보를 입력한 뒤 간편 분석을 실행하면 위험도 결과 카드가 표시됩니다.</p>
          </div>
        )}
      </div>
      <Card title="추천 챌린지">
        {recommendedChallengesLoading ? (
          <div className="state-box">추천 챌린지를 불러오는 중입니다...</div>
        ) : recommendedChallenges.length > 0 ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {recommendedChallenges.map((challenge) => (
              <div className="mini-card" key={String(challenge.id)}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="challenge-icon">
                    {categoryIcon[String(challenge.category ?? "COMMON").toUpperCase()] ?? null}
                  </span>
                  <strong>{String(challenge.title ?? "추천 챌린지")}</strong>
                </div>
                <p className="muted">{getChallengeCardDescription(challenge.description)}</p>
                <Link className="button secondary" to={`/challenges/${String(challenge.id)}`}>
                  상세보기
                </Link>
              </div>
            ))}
          </div>
        ) : (
          <div className="state-box">
            <strong>추천 챌린지가 아직 없습니다.</strong>
            <div className="button-row" style={{ marginTop: 12 }}>
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
          </div>
        )}
      </Card>
    </div>
  );
}
