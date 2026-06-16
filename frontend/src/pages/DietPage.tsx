import { ChangeEvent, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import {
  analyzeDiet,
  listDietRecords,
  type DietCandidateFood,
  type DietAnalyzeResponse,
} from "../api/diets";
import { normalizeImageForPreview } from "../api/uploads";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import { useAnalysisFeedbackDialog } from "../hooks/useAnalysisFeedbackDialog";
import { useAsyncJobPolling } from "../hooks/useAsyncJobPolling";
import { isHeicFile } from "../utils/files";
import { formatDateTime, mealTypeLabel } from "../utils/format";

type DietRecord = Record<string, unknown>;

const mealTypeOptions = [
  { value: "BREAKFAST", label: "아침" },
  { value: "LUNCH", label: "점심" },
  { value: "DINNER", label: "저녁" },
  { value: "SNACK", label: "간식" },
  { value: "LATE_NIGHT", label: "야식" },
];

function getDefaultMealType(): string {
  const hour = new Date().getHours();
  if (hour < 10) return "BREAKFAST";
  if (hour < 15) return "LUNCH";
  if (hour < 20) return "DINNER";
  if (hour < 23) return "LATE_NIGHT";
  return "SNACK";
}

const DIET_ANALYSIS_JOB_POLLING_INTERVAL_MS = 10000;

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function normalizeFoodItems(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object");
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of ["foods", "detected_foods", "items"]) {
      if (Array.isArray(record[key])) {
        return normalizeFoodItems(record[key]);
      }
    }
    return Object.entries(value as Record<string, unknown>).map(([name, detail]) => ({
      name,
      value: detail,
    }));
  }
  return [];
}

function foodDisplayName(food: Record<string, unknown>): string {
  return (
    String(food.matched_food_name ?? food.name ?? food.food_name ?? food.original_name ?? food.query_name ?? "").trim() ||
    "음식명 확인 불가"
  );
}

function mfdsCandidateName(food: Record<string, unknown>): string {
  const source = String(food.match_source ?? "").toLowerCase();
  const matched = String(food.matched_food_name ?? "").trim();
  if (!source.startsWith("mfds_") || !matched || matched === foodDisplayName(food)) {
    return "";
  }
  return matched;
}

function matchStatusLabel(food: Record<string, unknown>): string {
  const source = String(food.match_source ?? "").toLowerCase();
  const sourceLabels: Record<string, string> = {
    mfds_matched: "영양성분 후보 확인 필요",
    mfds_multiple_candidates: "후보 여러 개 확인 필요",
    mfds_weak_match: "낮은 신뢰 후보",
    mfds_no_candidates: "영양성분 후보 없음",
    mfds_skipped_generic: "재료/소스성 후보로 영양 검색 제외",
    mfds_skipped_low_confidence: "낮은 신뢰도로 영양 검색 제외",
    mfds_skipped_lookup_limit: "검색 제한으로 제외",
  };
  return sourceLabels[source] ?? (food.needs_user_confirmation === true ? "확인 필요" : "확인 완료");
}

function mfdsNutrition(food: Record<string, unknown>): Record<string, unknown> {
  return asRecord(asRecord(food.match_metadata).nutrition);
}

function hasMfdsNutrition(food: Record<string, unknown>): boolean {
  const nutrition = mfdsNutrition(food);
  return ["calories_kcal", "carbohydrate_g", "protein_g", "fat_g", "sodium_mg"].some(
    (key) => nutrition[key] !== null && nutrition[key] !== undefined && nutrition[key] !== "",
  );
}

function publicNutritionText(value: unknown): string {
  return String(value ?? "")
    .replace(/[Mm][Ff][Dd][Ss]\s*기준\s*영양성분\s*후보/g, "식품영양성분 데이터 기준 후보")
    .replace(/[Mm][Ff][Dd][Ss]\s*기준/g, "식품영양성분 데이터 기준")
    .replace(/[Mm][Ff][Dd][Ss]\s*후보/g, "영양성분 후보")
    .replace(/\b[Mm][Ff][Dd][Ss]\b/g, "식품영양성분 데이터");
}

function nutritionValue(value: unknown, unit: string): string {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? `${numberValue.toLocaleString("ko-KR")} ${unit}` : `${String(value)} ${unit}`;
}

function candidateFoodsFromPayload(payload: Record<string, unknown> | null): {
  autoConfirmed: DietCandidateFood[];
  needsConfirmation: DietCandidateFood[];
  noCandidate: DietCandidateFood[];
  nutritionStatus: string;
} {
  const rawOutput = asRecord(payload?.raw_output);
  const photoResult = asRecord(payload?.photo_result);
  const photoRawOutput = asRecord(photoResult.raw_output);
  const source: Record<string, unknown> = [payload, rawOutput, photoRawOutput].find((item) => hasCandidateFoods(item)) ?? {};
  return {
    autoConfirmed: candidateFoodList(source.auto_confirmed_foods),
    needsConfirmation: candidateFoodList(source.needs_confirmation_foods),
    noCandidate: candidateFoodList(source.no_candidate_foods),
    nutritionStatus: String(source.nutrition_calculation_status ?? asRecord(source.summary).nutrition_calculation_status ?? ""),
  };
}

function hasCandidateFoods(value: unknown): value is Record<string, unknown> {
  const record = asRecord(value);
  return ["auto_confirmed_foods", "needs_confirmation_foods", "no_candidate_foods"].some((key) =>
    Array.isArray(record[key]),
  );
}

function candidateFoodList(value: unknown): DietCandidateFood[] {
  return Array.isArray(value) ? (value.filter((item) => item && typeof item === "object") as DietCandidateFood[]) : [];
}

function candidateFoodName(food: DietCandidateFood): string {
  return (
    String(food.display_name ?? food.vision_food_name ?? food.raw_food_name ?? food.selected_candidate?.food_name ?? "").trim() ||
    "음식명 확인 필요"
  );
}

function dietRecordDisplayTitle(record: DietRecord): string {
  const foodNames = normalizeFoodItems(record.detected_foods)
    .map(foodDisplayName)
    .filter((name) => name && name !== "음식명 확인 불가");
  if (foodNames.length === 1) {
    return foodNames[0];
  }
  if (foodNames.length > 1) {
    return `${foodNames[0]} 외 ${foodNames.length - 1}개`;
  }
  const description = String(record.description ?? "").trim();
  if (description && description !== "사진으로 선택한 식단") {
    return description;
  }
  return "식단 기록";
}

function recordNeedsFoodConfirmation(record: DietRecord): boolean {
  const summary = asRecord(record.nutrition_summary);
  const status = String(record.match_source ?? record.nutrition_calculation_status ?? summary.nutrition_calculation_status ?? "").toLowerCase();
  return (
    record.needs_user_confirmation === true ||
    summary.needs_user_confirmation === true ||
    Number(record.needs_user_confirmation_count ?? summary.needs_user_confirmation_count ?? 0) > 0 ||
    status === "partial" ||
    status === "needs_user_confirmation" ||
    status === "needs_confirmation" ||
    status === "mfds_multiple_candidates" ||
    status.includes("confirmation") ||
    normalizeFoodItems(record.detected_foods).some((food) => {
      const foodStatus = String(food.match_source ?? food.match_status ?? food.nutrition_status ?? "").toLowerCase();
      return food.needs_user_confirmation === true || foodStatus.includes("confirmation") || foodStatus === "mfds_multiple_candidates";
    })
  );
}

export default function DietPage() {
  const [analysisMealType, setAnalysisMealType] = useState(getDefaultMealType());
  const [analysisDescription, setAnalysisDescription] = useState("");
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [records, setRecords] = useState<DietRecord[]>([]);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [selectedImagePreviewUrl, setSelectedImagePreviewUrl] = useState("");
  const [detectedFoodsImagePreviewFailed, setDetectedFoodsImagePreviewFailed] = useState(false);
  const [imagePreviewMessage, setImagePreviewMessage] = useState("");
  const [analysisJobId, setAnalysisJobId] = useState<number | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [canRetryAnalysis, setCanRetryAnalysis] = useState(false);
  const { clearFeedback, feedbackDialog, showFailure, showFeedback, showProcessing, showSuccess } =
    useAnalysisFeedbackDialog();
  const analysisRequestInFlightRef = useRef(false);
  const analysisStartedAtRef = useRef<number | null>(null);
  const handledAnalysisJobIdsRef = useRef<Set<number>>(new Set());

  const load = async () => {
    setError("");
    try {
      setRecords(await listDietRecords<DietRecord[]>());
    } catch (err) {
      setError(err instanceof Error ? err.message : "식단 기록을 불러오지 못했습니다.");
    }
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    return () => {
      if (selectedImagePreviewUrl) {
        URL.revokeObjectURL(selectedImagePreviewUrl);
      }
    };
  }, [selectedImagePreviewUrl]);

  useEffect(() => {
    const checkMobile = () => {
      const coarsePointer = window.matchMedia?.("(pointer: coarse)").matches ?? false;
      const mobileUserAgent = /Android|iPhone|iPad|iPod/i.test(window.navigator.userAgent);
      setIsMobileDevice(coarsePointer || mobileUserAgent);
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  useAsyncJobPolling({
    jobId: analysisJobId,
    enabled: isAnalyzing && analysisJobId !== null,
    intervalMs: DIET_ANALYSIS_JOB_POLLING_INTERVAL_MS,
    timeoutMs: 120000,
    onSuccess: async (job) => {
      if (handledAnalysisJobIdsRef.current.has(job.id)) {
        return;
      }
      handledAnalysisJobIdsRef.current.add(job.id);
      const finishedAt = Date.now();
      if (analysisStartedAtRef.current !== null) {
        console.info("Diet analysis completed", {
          jobId: job.id,
          elapsedMs: finishedAt - analysisStartedAtRef.current,
        });
      }
      analysisRequestInFlightRef.current = false;
      setIsAnalyzing(false);
      setAnalysisJobId(null);
      try {
        const result = dietAnalysisResultFromPayload(job.result_payload);
        if (!result) {
          throw new Error("식단 분석 결과를 불러오지 못했습니다.");
        }
        setAnalysisResult(result as unknown as Record<string, unknown>);
        setCanRetryAnalysis(false);
        showSuccess({ message: "저장된 식단 분석 결과를 확인해 주세요." });
        await load();
      } catch {
        showFailure();
        setCanRetryAnalysis(true);
      }
    },
    onFailure: (job) => {
      if (handledAnalysisJobIdsRef.current.has(job.id)) {
        return;
      }
      handledAnalysisJobIdsRef.current.add(job.id);
      analysisRequestInFlightRef.current = false;
      showFailure({ message: job.status === "CANCELED" ? "분석 작업이 취소되었습니다." : "다시 시도해주세요." });
      setCanRetryAnalysis(true);
      setIsAnalyzing(false);
      setAnalysisJobId(null);
    },
    onTimeout: () => {
      analysisRequestInFlightRef.current = false;
      showFailure();
      setCanRetryAnalysis(true);
      setIsAnalyzing(false);
      setAnalysisJobId(null);
    },
  });

  const handleDietImageChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (selectedImagePreviewUrl) {
      URL.revokeObjectURL(selectedImagePreviewUrl);
    }
    setImagePreviewMessage("");
    setDetectedFoodsImagePreviewFailed(false);
    clearFeedback();
    if (!file) {
      setSelectedImageFile(null);
      setSelectedImagePreviewUrl("");
      return;
    }
    setSelectedImageFile(file);
    if (!isHeicFile(file)) {
      setSelectedImagePreviewUrl(URL.createObjectURL(file));
      return;
    }

    setSelectedImagePreviewUrl("");
    setImagePreviewMessage("HEIC 이미지를 미리보기용 JPG로 변환 중입니다.");
    try {
      const previewBlob = await normalizeImageForPreview(file);
      setSelectedImagePreviewUrl(URL.createObjectURL(previewBlob));
      setImagePreviewMessage("");
    } catch (err) {
      setImagePreviewMessage(
        err instanceof Error
          ? err.message
          : "HEIC 미리보기를 생성하지 못했습니다. 분석은 업로드 후 다시 시도해주세요.",
      );
    }
  };

  const clearSelectedImage = () => {
    if (selectedImagePreviewUrl) {
      URL.revokeObjectURL(selectedImagePreviewUrl);
    }
    setSelectedImageFile(null);
    setSelectedImagePreviewUrl("");
    setDetectedFoodsImagePreviewFailed(false);
    setImagePreviewMessage("");
  };

  const runDietAnalysis = async () => {
    if (isAnalyzing || analysisRequestInFlightRef.current) {
      return;
    }
    setError("");
    setMessage("");
    clearFeedback();
    if (!selectedImageFile) {
      showFeedback("info", {
        title: "이미지를 선택해 주세요",
        message: "식단 분석을 진행하려면 먼저 식단 사진을 업로드해 주세요.",
      });
      return;
    }
    setAnalysisResult(null);
    setCanRetryAnalysis(false);
    analysisRequestInFlightRef.current = true;
    analysisStartedAtRef.current = Date.now();
    setIsAnalyzing(true);
    showProcessing();
    try {
      const payload = buildDietAnalysisFormData(selectedImageFile, analysisDescription, analysisMealType);
      const job = await analyzeDiet(payload);
      handledAnalysisJobIdsRef.current.delete(job.id);
      console.info("Diet analysis job started", { jobId: job.id });
      setMessage("");
      setAnalysisJobId(job.id);
    } catch (err) {
      analysisRequestInFlightRef.current = false;
      setError("분석 요청을 시작하지 못했습니다. 입력 내용을 확인한 뒤 다시 시도해주세요.");
      showFailure();
      setCanRetryAnalysis(true);
      setIsAnalyzing(false);
    }
  };

  const detectedFoods = Array.isArray(analysisResult?.detected_foods)
    ? (analysisResult.detected_foods as Record<string, unknown>[])
    : [];
  const warnings = Array.isArray(analysisResult?.warnings) ? (analysisResult.warnings as string[]) : [];
  const recommendedActions = Array.isArray(analysisResult?.recommended_actions)
    ? (analysisResult.recommended_actions as string[])
    : [];
  const candidateFoods = candidateFoodsFromPayload(analysisResult);
  const hasCandidateSection =
    candidateFoods.autoConfirmed.length > 0 ||
    candidateFoods.needsConfirmation.length > 0 ||
    candidateFoods.noCandidate.length > 0;
  const hasAnyMfdsNutrition = detectedFoods.some(hasMfdsNutrition);

  return (
    <div className="page-stack">
      <header className="dashboard-header">
        <div>
          <h1>식단 분석</h1>
          <p>식단 사진을 등록하고 영양 정보와 개선 포인트를 확인합니다.</p>
        </div>
      </header>
    <div className="page-grid" style={{ gridTemplateColumns: "1fr" }}>
      {error && <ErrorMessage message={error} />}
      {canRetryAnalysis ? (
        <div className="button-row">
          <button disabled={isAnalyzing} onClick={runDietAnalysis} type="button">
            다시 시도
          </button>
        </div>
      ) : null}
      {message && <div className="state-box">{message}</div>}
      {feedbackDialog}
      <div className="state-box" style={{ gridColumn: "1 / -1" }}>
        식단 사진을 업로드하면 음식 후보와 영양 정보를 분석합니다. 직접 입력 기반 영양성분 자동 산정은 추후 제공
        예정입니다.
      </div>
      <Card
        title="식단 이미지 분석"
        className="diet-analysis-card"
        actions={
          <Link className="button secondary" to="/diets/history">
            결과 전체
          </Link>
        }
      >
        <form className="form" onSubmit={(event) => event.preventDefault()} style={{ display: "flex", flexDirection: "column", flex: 1 }}>
          <label>
            식사 구분
            <select value={analysisMealType} onChange={(event) => setAnalysisMealType(event.target.value)}>
              {mealTypeOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            식단 메모
            <textarea value={analysisDescription} onChange={(event) => setAnalysisDescription(event.target.value)} />
          </label>
          <div className="upload-box">
            <strong>음식 사진 선택</strong>
            <span>이미지 파일을 선택하거나, 지원되는 모바일 브라우저에서는 후면 카메라로 바로 촬영할 수 있습니다.</span>
            <div className="upload-action-grid">
              <label className="upload-action-button" htmlFor="diet-file-input">
                이미지 파일 선택
              </label>
              {isMobileDevice ? (
                <label className="upload-action-button" htmlFor="diet-camera-input">
                  카메라로 촬영
                </label>
              ) : (
                <span className="upload-action-button upload-action-button--disabled">
                  <span style={{ fontSize: "14px", fontWeight: 600 }}>카메라 촬영</span>
                  <span style={{ fontSize: "11px", fontWeight: 400, opacity: 0.7 }}>
                    카메라 촬영은 모바일에서 사용할 수 있습니다.
                  </span>
                </span>
              )}
            </div>
            <input
              accept="image/*,.heic,.heif"
              disabled={isAnalyzing}
              id="diet-file-input"
              onChange={handleDietImageChange}
              type="file"
            />
            <input
              accept="image/*,.heic,.heif"
              capture="environment"
              disabled={isAnalyzing}
              id="diet-camera-input"
              onChange={handleDietImageChange}
              type="file"
            />
            {selectedImageFile && (
              <div className="state-box upload-selected-file">
                <strong>선택한 이미지: {selectedImageFile.name}</strong>
                <span className="muted">이미지를 다시 선택하려면 파일 선택 또는 카메라 촬영을 눌러주세요.</span>
                <span className="muted">이미지에서 음식명 후보를 찾고 식단 기준표로 점수화합니다. 결과는 저장 전 확인해주세요.</span>
                <button className="button secondary" disabled={isAnalyzing} onClick={clearSelectedImage} type="button">
                  선택 이미지 삭제
                </button>
              </div>
            )}
            {imagePreviewMessage ? (
              <div className="state-box heic-preview-notice">
                {imagePreviewMessage}
              </div>
            ) : null}
            {selectedImagePreviewUrl ? (
              <img alt="선택한 음식 사진 미리보기" className="upload-preview" src={selectedImagePreviewUrl} />
            ) : null}
          </div>  {/* upload-box 닫힘 */}
          <div style={{ flex: 1 }} />
          <div className="button-row" style={{ justifyContent: "flex-end" }}>
            <button disabled={isAnalyzing} type="button" onClick={runDietAnalysis}>
              {isAnalyzing ? "식단 분석 중..." : "간편 식단 분석"}
            </button>
          </div>
        </form>
      </Card>
      <Card title="간편 분석 결과">
        <div className="state-box">
          참고용 식단 분석입니다. 음식명과 영양 정보는 사진 상태와 매칭 결과에 따라 달라질 수 있으니 저장 전 확인해 주세요.
        </div>
        {!analysisResult && (
          <div className="state-box">식단 메모를 입력하거나 사진 업로드 영역을 확인한 뒤 간편 분석을 실행해보세요.</div>
        )}
        {analysisResult && (
          <div className="card-list">
            <div className="mini-card">
              <p>{String(analysisResult.diet_feedback ?? "분석 결과를 확인해보세요.")}</p>
            </div>
            <div className="mini-card" style={{ padding: "10px 14px" }}>
              {selectedImagePreviewUrl && !detectedFoodsImagePreviewFailed && (
                <img
                  alt="업로드한 식단 사진"
                  className="upload-preview"
                  onError={() => setDetectedFoodsImagePreviewFailed(true)}
                  src={selectedImagePreviewUrl}
                  style={{ maxHeight: 140 }}
                />
              )}
              <span className="muted">감지된 음식</span>
              <strong>
                {detectedFoods.length > 0 ? detectedFoods.map(foodDisplayName).join(", ") : "음식명 확인 불가"}
              </strong>
            </div>
            <div className="chip-list">
              {detectedFoods.length === 0 && <span className="badge badge-reference">음식명 확인 불가</span>}
              {detectedFoods.map((food, index) => (
                <span className="badge badge-reference" key={`${foodDisplayName(food)}-${index}`}>
                  {foodDisplayName(food)} {food.confidence ? `${Math.round(Number(food.confidence) * 100)}%` : ""}
                </span>
              ))}
            </div>
            {!hasAnyMfdsNutrition && (
              <div className="state-box">
                영양성분 후보를 찾지 못했습니다. 음식 후보를 확인하면 더 정확한 분석이 가능합니다.
              </div>
            )}
            {detectedFoods.some((food) => mfdsCandidateName(food) || hasMfdsNutrition(food)) && (
              <div className="card-list">
                <div className="state-box">
                  식품영양성분 데이터 기준 후보입니다. 기준량은 100g 또는 1회 제공량일 수 있으며, 실제 음식 후보
                  확정 전까지 총 영양성분은 확정되지 않습니다.
                </div>
                {detectedFoods
                  .filter((food) => mfdsCandidateName(food) || hasMfdsNutrition(food))
                  .map((food, index) => {
                    const candidateName = mfdsCandidateName(food);
                    const mfds = mfdsNutrition(food);
                    return (
                      <div className="mini-card" key={`${foodDisplayName(food)}-mfds-${index}`}>
                        <strong>{foodDisplayName(food)}</strong>
                        {candidateName && <span className="muted">영양성분 후보: {candidateName}</span>}
                        <span className="badge badge-reference">상태: {matchStatusLabel(food)}</span>
                        {hasMfdsNutrition(food) ? (
                          <>
                            <span className="muted">식품영양성분 데이터 기준</span>
                            <div className="nutrition-grid">
                              <div>
                                <span>기준량</span>
                                <strong>{publicNutritionText(mfds.basis_label ?? "기준량 확인 필요")}</strong>
                              </div>
                              <div>
                                <span>열량</span>
                                <strong>{nutritionValue(mfds.calories_kcal, "kcal")}</strong>
                              </div>
                              <div>
                                <span>탄수화물</span>
                                <strong>{nutritionValue(mfds.carbohydrate_g, "g")}</strong>
                              </div>
                              <div>
                                <span>단백질</span>
                                <strong>{nutritionValue(mfds.protein_g, "g")}</strong>
                              </div>
                              <div>
                                <span>지방</span>
                                <strong>{nutritionValue(mfds.fat_g, "g")}</strong>
                              </div>
                              <div>
                                <span>나트륨</span>
                                <strong>{nutritionValue(mfds.sodium_mg, "mg")}</strong>
                              </div>
                            </div>
                            {(mfds.serving_reference || mfds.food_weight) && (
                              <span className="muted">
                                {[mfds.serving_reference ? `제공량: ${String(mfds.serving_reference)}` : "", mfds.food_weight ? `식품중량: ${String(mfds.food_weight)}` : ""]
                                  .filter(Boolean)
                                  .join(" · ")}
                              </span>
                            )}
                          </>
                        ) : (
                          <span className="muted">표시할 영양성분 후보가 없습니다.</span>
                        )}
                      </div>
                    );
                  })}
              </div>
            )}
            <div className="mini-card">
              <strong>식단 관리 포인트</strong>
              <span className="muted">
                현재 식단에서 확인할 성분과 다음 행동을 중심으로 보여드립니다.
              </span>
              {recommendedActions.length > 0 ? (
                <div className="chip-list">
                  {recommendedActions.map((action) => (
                    <span className="badge risk-low" key={`management-${action}`}>
                      {publicNutritionText(action)}
                    </span>
                  ))}
                </div>
              ) : (
                <span>음식 후보와 영양성분을 확인한 뒤 국물, 소스, 튀김류, 단 음료처럼 조절할 부분을 살펴보세요.</span>
              )}
            </div>
            {hasCandidateSection && (
              <div className="card-list">
                <div className="mini-card">
                  <strong>음식 후보 확인</strong>
                  <span className="muted">
                    후보 확인형 분석 응답이 포함된 경우에만 표시됩니다. 확정 전 음식은 영양 합산에서 제외될 수 있습니다.
                  </span>
                  {candidateFoods.nutritionStatus && (
                    <span className="badge badge-reference">
                      영양 계산 상태: {candidateFoods.nutritionStatus === "partial" ? "일부 확인 필요" : "확인 완료"}
                    </span>
                  )}
                </div>
                <div className="nutrition-grid">
                  <div>
                    <span>자동 확정</span>
                    <strong>{candidateFoods.autoConfirmed.length}개</strong>
                  </div>
                  <div>
                    <span>확인 필요</span>
                    <strong>{candidateFoods.needsConfirmation.length}개</strong>
                  </div>
                  <div>
                    <span>후보 없음</span>
                    <strong>{candidateFoods.noCandidate.length}개</strong>
                  </div>
                </div>
                {candidateFoods.needsConfirmation.length > 0 && (
                  <div className="chip-list">
                    {candidateFoods.needsConfirmation.slice(0, 8).map((food, index) => (
                      <span className="badge badge-reference" key={`${candidateFoodName(food)}-${index}`}>
                        {candidateFoodName(food)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
            {warnings.length > 0 && (
              <div className="warning-card card-list">
                {warnings.map((warning) => (
                  <span key={warning}>{warning}</span>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>
      <Card title="최근 식단">
        <div className="card-list">
          {records.length === 0 && <div className="state-box">최근 식단 기록이 없습니다.</div>}
          {records.slice(0, 5).map((record) => {
            const isManual = String(record.analysis_method ?? "").toUpperCase() === "MANUAL";
            const needsConfirmation = recordNeedsFoodConfirmation(record);
            return (
              <Link className="mini-card" key={String(record.id)} to={`/diets/${String(record.id)}`}>
                <div className="diet-record-meta">
                  <span className="muted">{formatDateTime(record.meal_time ?? record.created_at)}</span>
                  <span className="badge badge-reference">{mealTypeLabel(record.meal_type)}</span>
                  {isManual && <span className="badge badge-reference">직접 기록</span>}
                  {needsConfirmation && <span className="badge badge-reference">음식 후보 확인 필요</span>}
                </div>
                 <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px" }}>
                    <strong>{dietRecordDisplayTitle(record)}</strong>
                    <span className="muted" style={{ fontSize: "12px" }}>상세보기 →</span>
                  </div>
                {needsConfirmation && <span className="muted">영양성분은 후보 기준입니다.</span>}
              </Link>
            );
          })}
        </div>
      </Card>
    </div>
    </div>
  );
}

function buildDietAnalysisFormData(file: File, description: string, mealType: string): FormData {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("description", description || "사진으로 선택한 식단");
  formData.append("meal_time", new Date().toISOString());
  formData.append("meal_type", mealType);
  formData.append("image_path", file.name);
  return formData;
}

function dietAnalysisResultFromPayload(payload: Record<string, unknown> | null | undefined): DietAnalyzeResponse | null {
  if (!payload || !payload.diet_record || !payload.photo_result) {
    return null;
  }
  return payload as unknown as DietAnalyzeResponse;
}
