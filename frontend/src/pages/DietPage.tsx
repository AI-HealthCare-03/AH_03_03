import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  analyzeDiet,
  createDietRecord,
  listDietRecords,
  type DietFoodItem,
  type DietAnalyzeResponse,
  type DietNutritionSummary,
  type DietRecordPayload,
} from "../api/diets";
import { normalizeImageForPreview } from "../api/uploads";
import Card from "../components/Card";
import ConfirmDialog from "../components/ConfirmDialog";
import ErrorMessage from "../components/ErrorMessage";
import { useAsyncJobPolling } from "../hooks/useAsyncJobPolling";
import { isHeicFile } from "../utils/files";
import { formatDateTime, mealTypeLabel, scoreBadgeClass } from "../utils/format";

type DietRecord = Record<string, unknown>;

type ManualFoodDraft = {
  name: string;
  quantity: string;
  memo: string;
};

type NutritionDraft = {
  calories: string;
  carbohydrate_g: string;
  protein_g: string;
  fat_g: string;
  sodium_mg: string;
};
type FeedbackDialog = {
  message: string;
  title: string;
  tone?: "default" | "danger";
};

const diseaseScoreLabels: Record<string, string> = {
  DM: "당뇨",
  HTN: "고혈압",
  DL: "콜레스테롤·중성지방",
  OBE: "비만",
  ANEM: "빈혈",
};

const mealTypeOptions = [
  { value: "BREAKFAST", label: "아침" },
  { value: "LUNCH", label: "점심" },
  { value: "DINNER", label: "저녁" },
  { value: "SNACK", label: "간식" },
  { value: "LATE_NIGHT", label: "야식" },
];

function currentLocalDateTime(): string {
  const now = new Date();
  const offsetMs = now.getTimezoneOffset() * 60 * 1000;
  return new Date(now.getTime() - offsetMs).toISOString().slice(0, 16);
}

const emptyFoodDraft: ManualFoodDraft = { name: "", quantity: "", memo: "" };
const emptyNutritionDraft: NutritionDraft = {
  calories: "",
  carbohydrate_g: "",
  protein_g: "",
  fat_g: "",
  sodium_mg: "",
};

function parseOptionalNumber(value: string): number | null {
  if (value.trim() === "") return null;
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

function buildNutritionSummary(draft: NutritionDraft): DietNutritionSummary | null {
  const nutrition: DietNutritionSummary = {
    calories: parseOptionalNumber(draft.calories),
    carbohydrate_g: parseOptionalNumber(draft.carbohydrate_g),
    protein_g: parseOptionalNumber(draft.protein_g),
    fat_g: parseOptionalNumber(draft.fat_g),
    sodium_mg: parseOptionalNumber(draft.sodium_mg),
  };
  return Object.values(nutrition).some((value) => value !== null) ? nutrition : null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function diseaseScoreEntries(value: unknown): Array<[string, unknown]> {
  const scores = asRecord(value);
  return Object.entries(diseaseScoreLabels)
    .map(([code, label]) => [label, scores[code]] as [string, unknown])
    .filter(([, score]) => score !== null && score !== undefined && score !== "");
}

function foodDisplayName(food: Record<string, unknown>): string {
  return String(food.food_name ?? food.name ?? food.matched_food_name ?? "").trim() || "음식명 확인 불가";
}

function scoringSourceLabel(value: unknown): string {
  const source = String(value ?? "").toLowerCase();
  if (!source) {
    return "";
  }
  if (source.includes("vision") || source.includes("gpt")) {
    return "이미지 인식 + 식단 기준표";
  }
  return "식단 기준표";
}

export default function DietPage() {
  const [analysisDescription, setAnalysisDescription] = useState("");
  const [manualMealType, setManualMealType] = useState("LUNCH");
  const [manualMealTime, setManualMealTime] = useState(currentLocalDateTime());
  const [manualDescription, setManualDescription] = useState("");
  const [manualMemo, setManualMemo] = useState("");
  const [manualFoods, setManualFoods] = useState<ManualFoodDraft[]>([{ ...emptyFoodDraft }]);
  const [manualNutrition, setManualNutrition] = useState<NutritionDraft>({ ...emptyNutritionDraft });
  const [records, setRecords] = useState<DietRecord[]>([]);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [selectedImagePreviewUrl, setSelectedImagePreviewUrl] = useState("");
  const [imagePreviewMessage, setImagePreviewMessage] = useState("");
  const [analysisJobId, setAnalysisJobId] = useState<number | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [feedbackDialog, setFeedbackDialog] = useState<FeedbackDialog | null>(null);
  const [canRetryAnalysis, setCanRetryAnalysis] = useState(false);

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

  useAsyncJobPolling({
    jobId: analysisJobId,
    enabled: isAnalyzing && analysisJobId !== null,
    intervalMs: 1500,
    timeoutMs: 120000,
    onSuccess: async (job) => {
      try {
        const result = dietAnalysisResultFromPayload(job.result_payload);
        if (!result) {
          throw new Error("식단 분석 결과를 불러오지 못했습니다.");
        }
        setAnalysisResult(result as unknown as Record<string, unknown>);
        setCanRetryAnalysis(false);
        setFeedbackDialog({
          title: "식단 분석이 완료되었습니다.",
          message: "저장된 식단 분석 결과를 확인해 주세요.",
        });
        await load();
      } catch {
        setFeedbackDialog({
          title: "식단 분석에 실패했습니다.",
          message: "잠시 후 다시 시도해 주세요.",
          tone: "danger",
        });
        setCanRetryAnalysis(true);
      } finally {
        setIsAnalyzing(false);
        setAnalysisJobId(null);
      }
    },
    onFailure: (job) => {
      setFeedbackDialog({
        title: "식단 분석에 실패했습니다.",
        message: job.status === "CANCELED" ? "식단 분석 작업이 취소되었습니다." : "잠시 후 다시 시도해 주세요.",
        tone: "danger",
      });
      setCanRetryAnalysis(true);
      setIsAnalyzing(false);
      setAnalysisJobId(null);
    },
    onTimeout: () => {
      setFeedbackDialog({
        title: "식단 분석에 실패했습니다.",
        message: "잠시 후 다시 시도해 주세요.",
        tone: "danger",
      });
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
    setImagePreviewMessage("");
  };

  const submitManualDiet = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    const foodItems: DietFoodItem[] = manualFoods
      .map((food) => ({
        name: food.name.trim(),
        quantity: food.quantity.trim() || null,
        memo: food.memo.trim() || null,
      }))
      .filter((food) => food.name.length > 0);
    if (foodItems.length === 0 && !manualDescription.trim()) {
      setError("음식명 또는 식단 설명을 입력해주세요.");
      return;
    }
    const nutritionSummary = buildNutritionSummary(manualNutrition);
    const description =
      manualDescription.trim() || foodItems.map((food) => food.name).join(", ") || "직접 입력한 식단";
    const payload: DietRecordPayload = {
      meal_type: manualMealType,
      meal_time: manualMealTime ? new Date(manualMealTime).toISOString() : new Date().toISOString(),
      description,
      detected_foods: foodItems.length > 0 ? foodItems : null,
      nutrition_summary: nutritionSummary,
      analysis_method: "MANUAL",
      is_user_corrected: true,
      memo: manualMemo.trim() || null,
    };
    try {
      await createDietRecord(payload);
      setManualMealType("LUNCH");
      setManualMealTime(currentLocalDateTime());
      setManualDescription("");
      setManualMemo("");
      setManualFoods([{ ...emptyFoodDraft }]);
      setManualNutrition({ ...emptyNutritionDraft });
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "식단 기록 저장에 실패했습니다.");
    }
  };

  const runDietAnalysis = async () => {
    setError("");
    setMessage("");
    setAnalysisResult(null);
    setCanRetryAnalysis(false);
    setIsAnalyzing(true);
    try {
      const payload = selectedImageFile
        ? buildDietAnalysisFormData(selectedImageFile, analysisDescription)
        : {
            description: analysisDescription || "기록된 식단",
            meal_time: new Date().toISOString(),
            image_path: null,
          };
      const job = await analyzeDiet(payload);
      setMessage("");
      setAnalysisJobId(job.id);
    } catch (err) {
      setError("분석 요청을 시작하지 못했습니다. 입력 내용을 확인한 뒤 다시 시도해주세요.");
      setCanRetryAnalysis(true);
      setIsAnalyzing(false);
    }
  };

  const detectedFoods = Array.isArray(analysisResult?.detected_foods)
    ? (analysisResult.detected_foods as Record<string, unknown>[])
    : [];
  const nutrition =
    analysisResult?.nutrition_summary && typeof analysisResult.nutrition_summary === "object"
      ? (analysisResult.nutrition_summary as Record<string, unknown>)
      : {};
  const diseaseScores = diseaseScoreEntries(analysisResult?.disease_scores ?? nutrition.disease_scores);
  const foodScoreDetails = Array.isArray(analysisResult?.food_score_details)
    ? (analysisResult.food_score_details as Record<string, unknown>[])
    : [];
  const warnings = Array.isArray(analysisResult?.warnings) ? (analysisResult.warnings as string[]) : [];
  const recommendedActions = Array.isArray(analysisResult?.recommended_actions)
    ? (analysisResult.recommended_actions as string[])
    : [];
  const scoringSource = scoringSourceLabel(analysisResult?.scoring_source);

  return (
    <div className="page-grid">
      {error && <ErrorMessage message={error} />}
      {canRetryAnalysis ? (
        <div className="button-row">
          <button disabled={isAnalyzing} onClick={runDietAnalysis} type="button">
            다시 시도
          </button>
        </div>
      ) : null}
      {message && <div className="state-box">{message}</div>}
      {feedbackDialog && (
        <ConfirmDialog
          confirmLabel="확인"
          message={feedbackDialog.message}
          onConfirm={() => setFeedbackDialog(null)}
          showCancel={false}
          title={feedbackDialog.title}
          tone={feedbackDialog.tone}
        />
      )}
      <Card
        title="식단 이미지 분석"
        actions={
          <Link className="button secondary" to="/diets/history">
            결과 전체
          </Link>
        }
      >
        <form className="form" onSubmit={(event) => event.preventDefault()}>
          <label>
            식단 메모
            <textarea value={analysisDescription} onChange={(event) => setAnalysisDescription(event.target.value)} />
          </label>
          <div className="upload-box">
            <strong>음식 사진 선택</strong>
            <span>이미지 파일을 선택하거나, 지원되는 모바일 브라우저에서는 후면 카메라로 바로 촬영할 수 있습니다.</span>
            <div className="button-row">
              <label className="button secondary" htmlFor="diet-file-input">
                이미지 파일 선택
              </label>
              <label className="button secondary" htmlFor="diet-camera-input">
                카메라로 촬영
              </label>
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
          </div>
          <div className="button-row">
            <button disabled={isAnalyzing} type="button" onClick={runDietAnalysis}>
              {isAnalyzing ? "식단 분석 중..." : "간편 식단 분석"}
            </button>
          </div>
        </form>
      </Card>
      <Card title="식단 직접 입력">
        <form className="form" onSubmit={submitManualDiet}>
          <div className="state-box">사진 없이 식단을 직접 기록할 수 있습니다. 영양정보를 알고 있다면 선택적으로 입력해주세요.</div>
          <div className="form-grid">
            <label>
              식사 구분
              <select value={manualMealType} onChange={(event) => setManualMealType(event.target.value)}>
                {mealTypeOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              식사 시각
              <input type="datetime-local" value={manualMealTime} onChange={(event) => setManualMealTime(event.target.value)} />
            </label>
          </div>
          <label>
            식단 설명
            <input
              placeholder="예: 현미밥, 닭가슴살 샐러드"
              value={manualDescription}
              onChange={(event) => setManualDescription(event.target.value)}
            />
          </label>
          <div className="manual-food-list">
            <div className="diet-record-header">
              <strong>음식 목록</strong>
              <button
                className="secondary compact-button"
                onClick={() => setManualFoods((prev) => [...prev, { ...emptyFoodDraft }])}
                type="button"
              >
                음식 추가
              </button>
            </div>
            {manualFoods.map((food, index) => (
              <div className="manual-food-row" key={`manual-food-${index}`}>
                <label>
                  음식명
                  <input
                    placeholder="예: 현미밥"
                    value={food.name}
                    onChange={(event) =>
                      setManualFoods((prev) => prev.map((item, itemIndex) => (itemIndex === index ? { ...item, name: event.target.value } : item)))
                    }
                  />
                </label>
                <label>
                  수량
                  <input
                    placeholder="예: 1공기"
                    value={food.quantity}
                    onChange={(event) =>
                      setManualFoods((prev) =>
                        prev.map((item, itemIndex) => (itemIndex === index ? { ...item, quantity: event.target.value } : item)),
                      )
                    }
                  />
                </label>
                <label>
                  메모
                  <input
                    placeholder="선택"
                    value={food.memo}
                    onChange={(event) =>
                      setManualFoods((prev) => prev.map((item, itemIndex) => (itemIndex === index ? { ...item, memo: event.target.value } : item)))
                    }
                  />
                </label>
                {manualFoods.length > 1 && (
                  <button
                    className="btn-danger-outline compact-button"
                    onClick={() => setManualFoods((prev) => prev.filter((_, itemIndex) => itemIndex !== index))}
                    type="button"
                  >
                    삭제
                  </button>
                )}
              </div>
            ))}
          </div>
          <div className="nutrition-input-grid">
            <label>
              칼로리 kcal
              <input
                min="0"
                type="number"
                value={manualNutrition.calories}
                onChange={(event) => setManualNutrition((prev) => ({ ...prev, calories: event.target.value }))}
              />
            </label>
            <label>
              탄수화물 g
              <input
                min="0"
                type="number"
                value={manualNutrition.carbohydrate_g}
                onChange={(event) => setManualNutrition((prev) => ({ ...prev, carbohydrate_g: event.target.value }))}
              />
            </label>
            <label>
              단백질 g
              <input
                min="0"
                type="number"
                value={manualNutrition.protein_g}
                onChange={(event) => setManualNutrition((prev) => ({ ...prev, protein_g: event.target.value }))}
              />
            </label>
            <label>
              지방 g
              <input
                min="0"
                type="number"
                value={manualNutrition.fat_g}
                onChange={(event) => setManualNutrition((prev) => ({ ...prev, fat_g: event.target.value }))}
              />
            </label>
            <label>
              나트륨 mg
              <input
                min="0"
                type="number"
                value={manualNutrition.sodium_mg}
                onChange={(event) => setManualNutrition((prev) => ({ ...prev, sodium_mg: event.target.value }))}
              />
            </label>
          </div>
          <label>
            메모
            <textarea value={manualMemo} onChange={(event) => setManualMemo(event.target.value)} />
          </label>
          <div className="button-row">
            <button type="submit">직접 기록 저장</button>
          </div>
        </form>
      </Card>
      <Card title="간편 분석 결과">
        {!analysisResult && (
          <div className="state-box">식단 메모를 입력하거나 사진 업로드 영역을 확인한 뒤 간편 분석을 실행해보세요.</div>
        )}
        {analysisResult && (
          <div className="card-list">
            <div className="score-panel">
              <span>식단 점수</span>
              <strong>{String(analysisResult.diet_score ?? "-")}</strong>
              <p>{String(analysisResult.diet_feedback ?? "분석 결과를 확인해보세요.")}</p>
            </div>
            <div className="mini-card">
              <span className="muted">분석 음식</span>
              <strong>
                {detectedFoods.length > 0 ? detectedFoods.map(foodDisplayName).join(", ") : "음식명 확인 불가"}
              </strong>
            </div>
            <div className="nutrition-grid">
              <div>
                <span>칼로리</span>
                <strong>{String(nutrition.calories ?? "-")} kcal</strong>
              </div>
              <div>
                <span>탄수화물</span>
                <strong>{String(nutrition.carbohydrate_g ?? "-")} g</strong>
              </div>
              <div>
                <span>단백질</span>
                <strong>{String(nutrition.protein_g ?? "-")} g</strong>
              </div>
              <div>
                <span>지방</span>
                <strong>{String(nutrition.fat_g ?? "-")} g</strong>
              </div>
              <div>
                <span>나트륨</span>
                <strong>{String(nutrition.sodium_mg ?? "-")} mg</strong>
              </div>
            </div>
            <div className="chip-list">
              {detectedFoods.length === 0 && <span className="badge badge-reference">음식명 확인 불가</span>}
              {detectedFoods.map((food, index) => (
                <span className="badge badge-reference" key={`${foodDisplayName(food)}-${index}`}>
                  {foodDisplayName(food)} {food.confidence ? `${Math.round(Number(food.confidence) * 100)}%` : ""}
                </span>
              ))}
            </div>
            {diseaseScores.length > 0 && (
              <div className="nutrition-grid">
                {diseaseScores.map(([label, score]) => (
                  <div key={label}>
                    <span>{label} 식단 점수</span>
                    <strong>{Math.round(Number(score))}점</strong>
                  </div>
                ))}
              </div>
            )}
            {foodScoreDetails.length > 0 && (
              <div className="card-list">
                {foodScoreDetails.slice(0, 3).map((detail, index) => {
                  const matched = detail.matched_food_name ? String(detail.matched_food_name) : "매칭 정보 없음";
                  const detailScores = diseaseScoreEntries(detail.scores);
                  return (
                    <div className="mini-card" key={`${foodDisplayName(detail)}-${index}`}>
                      <strong>{foodDisplayName(detail)}</strong>
                      <span className="muted">점수 기준: {matched}</span>
                      {detailScores.length > 0 ? (
                        <div className="chip-list">
                          {detailScores.map(([label, score]) => (
                            <span className="badge badge-reference" key={`${foodDisplayName(detail)}-${label}`}>
                              {label}: {Math.round(Number(score))}점
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="muted">음식별 점수 확인 불가</span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
            {scoringSource && <span className="badge badge-reference">점수 기준: {scoringSource}</span>}
            {warnings.length > 0 && (
              <div className="warning-card card-list">
                {warnings.map((warning) => (
                  <span key={warning}>{warning}</span>
                ))}
              </div>
            )}
            <div className="chip-list">
              {recommendedActions.map((action) => (
                <span className="badge risk-low" key={action}>
                  {action}
                </span>
              ))}
            </div>
          </div>
        )}
      </Card>
      <Card title="최근 식단">
        <div className="card-list">
          {records.length === 0 && <div className="state-box">최근 식단 기록이 없습니다.</div>}
          {records.slice(0, 5).map((record) => {
            const scoreRaw = record.diet_score != null ? Number(record.diet_score) : null;
            const isManual = String(record.analysis_method ?? "").toUpperCase() === "MANUAL";
            return (
              <Link className="mini-card" key={String(record.id)} to={`/diets/${String(record.id)}`}>
                <div className="diet-record-meta">
                  <span className="muted">{formatDateTime(record.meal_time ?? record.created_at)}</span>
                  <span className="badge badge-reference">{mealTypeLabel(record.meal_type)}</span>
                  {isManual && <span className="badge badge-reference">직접 기록</span>}
                </div>
                <strong>{String(record.description ?? "식단 기록")}</strong>
                {scoreRaw !== null ? (
                  <span className={`badge ${scoreBadgeClass(scoreRaw)}`}>{scoreRaw}점</span>
                ) : isManual ? (
                  <span className="badge badge-reference">점수 미산정</span>
                ) : null}
              </Link>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

function buildDietAnalysisFormData(file: File, description: string): FormData {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("description", description || "사진으로 선택한 식단");
  formData.append("meal_time", new Date().toISOString());
  formData.append("image_path", file.name);
  return formData;
}

function dietAnalysisResultFromPayload(payload: Record<string, unknown> | null | undefined): DietAnalyzeResponse | null {
  if (!payload || !payload.diet_record || !payload.photo_result) {
    return null;
  }
  return payload as unknown as DietAnalyzeResponse;
}
