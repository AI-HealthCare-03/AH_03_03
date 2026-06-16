import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import {
  getDietHealthRecommendations,
  getDietRecord,
  getDietRecordImageByUrl,
  listDietPhotoResults,
  type DietCandidateFood,
  type DietHealthRecommendation,
} from "../api/diets";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import { formatDateTime, mealTypeLabel } from "../utils/format";

type Item = Record<string, unknown>;

const DEFAULT_DIET_RECOMMENDATION_NOTICE =
  "이 내용은 진단이나 처방이 아닌 생활관리 참고 정보입니다. 실제 섭취량이 확정되지 않아 영양 판단은 참고용입니다.";

function analysisMethodLabel(value: unknown): string {
  const method = String(value ?? "").toUpperCase();
  if (method === "MANUAL") {
    return "직접 입력";
  }
  if (method === "RECOMMENDATION") {
    return "추천 식단";
  }
  if (method) {
    return "식단 분석";
  }
  return "수동 기록";
}

function isManualRecord(record: Item | null): boolean {
  return String(record?.analysis_method ?? "").toUpperCase() === "MANUAL";
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
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

function candidateFoodsFromPayload(...payloads: Array<Record<string, unknown> | null | undefined>): {
  autoConfirmed: DietCandidateFood[];
  needsConfirmation: DietCandidateFood[];
  noCandidate: DietCandidateFood[];
  nutritionStatus: string;
} {
  const source: Record<string, unknown> = payloads.find((item) => hasCandidateFoods(item)) ?? {};
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

function publicImageUrlFromPayloads(...payloads: Array<Record<string, unknown> | null | undefined>): string {
  // `image_path` is currently an original filename, not a browser-accessible URL.
  const urlKeys = ["image_url", "file_url", "stored_file_url", "original_image_url"];
  for (const payload of payloads) {
    const record = asRecord(payload);
    for (const key of urlKeys) {
      const value = String(record[key] ?? "").trim();
      if (isBrowserAccessibleImageUrl(value)) {
        return value;
      }
    }
  }
  return "";
}

function isBrowserAccessibleImageUrl(value: string): boolean {
  return (
    value.startsWith("https://") ||
    value.startsWith("http://") ||
    value.startsWith("/") ||
    value.startsWith("blob:") ||
    value.startsWith("data:image/")
  );
}

function shouldLoadImageWithApiClient(value: string): boolean {
  return value.startsWith("/api/v1/") || value.startsWith("/diets/");
}

function imageUrlFromPayload(payload: Record<string, unknown> | null | undefined): string {
  return publicImageUrlFromPayloads(payload);
}

function hasDietRecommendationContent(recommendation: DietHealthRecommendation | null): boolean {
  if (!recommendation) {
    return false;
  }
  return [
    recommendation.nutrition_findings,
    recommendation.disease_context,
    recommendation.recommended_foods,
    recommendation.caution_foods,
    recommendation.recommended_challenges,
  ].some((items) => Array.isArray(items) && items.length > 0);
}

function hasConfirmationFlag(...payloads: Array<Record<string, unknown> | null | undefined>): boolean {
  return payloads.some((payload) => {
    const record = asRecord(payload);
    const summary = asRecord(record.summary);
    const status = String(record.nutrition_calculation_status ?? summary.nutrition_calculation_status ?? "").toLowerCase();
    return (
      record.needs_user_confirmation === true ||
      summary.needs_user_confirmation === true ||
      Number(record.needs_user_confirmation_count ?? summary.needs_user_confirmation_count ?? 0) > 0 ||
      status === "partial" ||
      status === "needs_user_confirmation" ||
      status === "needs_confirmation"
    );
  });
}

function truncateText(value: unknown, maxLength = 150): string {
  const cleaned = publicNutritionText(value)
    .replace(/https?:\/\/\S+/g, "참고 링크")
    .replace(/\s+/g, " ")
    .trim();
  if (cleaned.length <= maxLength) {
    return cleaned;
  }
  return `${cleaned.slice(0, maxLength).trim()}...`;
}

function isCautionFinding(type: string): boolean {
  return ["excess_candidate", "habit_candidate", "medical_caution"].includes(type);
}

export default function DietResultPage() {
  const { dietRecordId } = useParams();
  const navigate = useNavigate();
  const [record, setRecord] = useState<Item | null>(null);
  const [photoResults, setPhotoResults] = useState<Item[]>([]);
  const [error, setError] = useState("");
  const [recommendation, setRecommendation] = useState<DietHealthRecommendation | null>(null);
  const [recommendationError, setRecommendationError] = useState("");
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [detailImageFailed, setDetailImageFailed] = useState(false);
  const [detailImageObjectUrl, setDetailImageObjectUrl] = useState("");

  const normalizeFoods = (value: unknown): Item[] => {
    if (Array.isArray(value)) {
      return value.filter((item): item is Item => Boolean(item) && typeof item === "object");
    }
    if (value && typeof value === "object") {
      const record = value as Record<string, unknown>;
      for (const key of ["foods", "detected_foods", "items"]) {
        if (Array.isArray(record[key])) {
          return normalizeFoods(record[key]);
        }
      }
      return Object.entries(value as Record<string, unknown>).map(([name, detail]) => ({
        name,
        value: detail,
      }));
    }
    return [];
  };

  const nutrition =
    record?.nutrition_summary && typeof record.nutrition_summary === "object"
      ? (record.nutrition_summary as Record<string, unknown>)
      : {};
  const hasNutrition = ["calories", "carbohydrate_g", "protein_g", "fat_g", "sodium_mg"].some(
    (key) => nutrition[key] !== null && nutrition[key] !== undefined && nutrition[key] !== "",
  );
  const detectedFoods = normalizeFoods(record?.detected_foods ?? photoResults[0]?.detected_foods);
  const detectedFoodNames = detectedFoods.map(foodDisplayName).filter((name) => name && name !== "음식명 확인 불가");
  const descriptionText = String(record?.description ?? "").trim();
  const dietRecordTitle =
    detectedFoodNames.length === 1
      ? detectedFoodNames[0]
      : detectedFoodNames.length > 1
        ? `${detectedFoodNames[0]} 외 ${detectedFoodNames.length - 1}개`
        : descriptionText && descriptionText !== "사진으로 선택한 식단"
          ? descriptionText
          : "식단 기록";
  const photoConfidence =
    photoResults[0]?.confidence_payload && typeof photoResults[0].confidence_payload === "object"
      ? (photoResults[0].confidence_payload as Record<string, unknown>)
      : {};
  const rawOutput = asRecord(photoResults[0]?.raw_output);
  const candidateFoods = candidateFoodsFromPayload(rawOutput, record);
  const hasCandidateSection =
    candidateFoods.autoConfirmed.length > 0 ||
    candidateFoods.needsConfirmation.length > 0 ||
    candidateFoods.noCandidate.length > 0;
  const needsFoodConfirmation =
    candidateFoods.needsConfirmation.length > 0 ||
    candidateFoods.noCandidate.length > 0 ||
    detectedFoods.some((food) => {
      const status = String(food.match_status ?? food.nutrition_status ?? "").toLowerCase();
      return food.needs_user_confirmation === true || status.includes("confirmation");
    }) ||
    hasConfirmationFlag(rawOutput, record, photoConfidence, nutrition);
  const hasAnyMfdsNutrition = detectedFoods.some(hasMfdsNutrition);
  const nutritionFindings = recommendation?.nutrition_findings ?? [];
  const cautionFindings = nutritionFindings.filter((finding) => isCautionFinding(finding.type));
  const supportFindings = nutritionFindings.filter((finding) => !isCautionFinding(finding.type));
  const recommendedFoods = recommendation?.recommended_foods ?? [];
  const cautionFoods = recommendation?.caution_foods ?? [];
  const hasManagementPoints =
    nutritionFindings.length > 0 ||
    Boolean(recommendation?.disease_context.length) ||
    recommendedFoods.length > 0 ||
    cautionFoods.length > 0 ||
    Boolean(recommendation?.rag_comment?.summary);
  const detailImageUrl =
    imageUrlFromPayload(record) ||
    imageUrlFromPayload(photoResults[0]) ||
    imageUrlFromPayload(rawOutput) ||
    imageUrlFromPayload(photoConfidence);
  const displayImageUrl = detailImageObjectUrl || (shouldLoadImageWithApiClient(detailImageUrl) ? "" : detailImageUrl);

  useEffect(() => {
    const load = async () => {
      if (!dietRecordId) {
        return;
      }
      setError("");
      try {
        const nextRecord = await getDietRecord<Item>(Number(dietRecordId));
        setRecord(nextRecord);
        if (String(nextRecord.analysis_method ?? "").toUpperCase() === "MANUAL") {
          setPhotoResults([]);
        } else {
          setPhotoResults(await listDietPhotoResults<Item[]>(Number(dietRecordId)));
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "식단 분석 결과를 불러오지 못했습니다.");
      }
    };
    void load();
  }, [dietRecordId]);

  useEffect(() => {
    const loadRecommendations = async () => {
      const numericDietRecordId = Number(dietRecordId);
      if (!Number.isFinite(numericDietRecordId) || numericDietRecordId <= 0) {
        setRecommendation(null);
        return;
      }
      setRecommendationLoading(true);
      setRecommendationError("");
      try {
        setRecommendation(await getDietHealthRecommendations(numericDietRecordId));
      } catch (err) {
        setRecommendation(null);
        setRecommendationError(
          err instanceof Error ? err.message : "식단 건강관리 추천을 불러오지 못했습니다.",
        );
      } finally {
        setRecommendationLoading(false);
      }
    };

    void loadRecommendations();
  }, [dietRecordId]);

  useEffect(() => {
    setDetailImageFailed(false);
  }, [detailImageUrl]);

  useEffect(() => {
    let objectUrl = "";
    let isMounted = true;
    setDetailImageObjectUrl("");

    if (!detailImageUrl) {
      return undefined;
    }

    if (!shouldLoadImageWithApiClient(detailImageUrl)) {
      return undefined;
    }

    const loadImage = async () => {
      try {
        const blob = await getDietRecordImageByUrl(detailImageUrl);
        if (!isMounted) {
          return;
        }
        objectUrl = URL.createObjectURL(blob);
        setDetailImageObjectUrl(objectUrl);
      } catch {
        if (isMounted) {
          setDetailImageFailed(true);
        }
      }
    };

    void loadImage();
    return () => {
      isMounted = false;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [detailImageUrl, dietRecordId]);

  const isManual = isManualRecord(record);

  return (
    <div className="page-grid" style={{ gridTemplateColumns: "1fr" }}>
      {error && <ErrorMessage message={error} />}
      <div style={{ marginBottom: "8px", gridColumn: "1 / -1" }}>
        <button className="button secondary" onClick={() => navigate(-1)} type="button">
          ← 이전으로
        </button>
      </div>
      <Card
        title={isManual ? "기존 직접 입력 기록" : "식단 분석 결과"}
        actions={
          <Link className="button secondary" to="/diets/history">
            전체 기록
          </Link>
        }
      >
        <div className="card-list">
          {!isManual && (
            <div className="state-box">
              참고용 분석 결과입니다. 음식명과 영양 정보는 사진 상태와 매칭 결과에 따라 달라질 수 있습니다.
            </div>
          )}
          <div className="diet-record-meta">
            <span className="muted">{formatDateTime(record?.meal_time ?? record?.created_at)}</span>
            <span className="badge badge-reference">{mealTypeLabel(record?.meal_type)}</span>
          </div>
          <strong>{dietRecordTitle}</strong>
          {!isManual && (
            <div className="mini-card">
              <span className="muted">분석 음식</span>
              <strong>
                {detectedFoods.length > 0 ? detectedFoods.map(foodDisplayName).join(", ") : "음식명 확인 불가"}
              </strong>
            </div>
          )}
        </div>
        {!isManual && (
          <div className="card-list">
            {displayImageUrl && !detailImageFailed && (
              <div className="mini-card">
                <span className="muted">분석한 식단 사진</span>
                <img
                  alt="분석한 식단 사진"
                  className="upload-preview"
                  onError={() => setDetailImageFailed(true)}
                  src={displayImageUrl}
                />
              </div>
            )}
            <div className="chip-list">
              {detectedFoods.length === 0 && <div className="state-box">음식명 확인 불가</div>}
              {detectedFoods.map((food, index) => (
                <span className="badge badge-reference" key={`${foodDisplayName(food)}-${index}`}>
                  {foodDisplayName(food)} {food.confidence ? `${Math.round(Number(food.confidence) * 100)}%` : ""}
                </span>
              ))}
            </div>
            {detectedFoods.some((food) => mfdsCandidateName(food) || hasMfdsNutrition(food)) && (
              <>
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
              </>
            )}
          </div>
        )}
      </Card>
      {!isManual && hasCandidateSection && (
        <Card title="음식 후보 확인">
          <div className="card-list">
            <div className="mini-card">
              <strong>후보 확인형 분석 응답</strong>
              <span className="muted">
                자동 확정되지 않은 음식은 사용자 확인 전까지 영양 합산에서 제외될 수 있습니다.
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
        </Card>
      )}
      {isManual ? (
        <Card title="입력 영양정보">
          {!hasNutrition ? (
            <div className="state-box">
              사진 분석 또는 음식 정보가 충분한 기록에서 식단 평가를 확인할 수 있습니다.
            </div>
          ) : (
            <div className="card-list">
              {[
                ["탄수화물", nutrition.carbohydrate_g, "g"],
                ["단백질", nutrition.protein_g, "g"],
                ["지방", nutrition.fat_g, "g"],
                ["나트륨", nutrition.sodium_mg, "mg"],
              ].map(([label, value, unit]) => (
                <div className="bar-row" key={String(label)}>
                  <span>{String(label)}</span>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${Math.min(Number(value ?? 0), 100)}%` }} />
                  </div>
                  <strong>
                    {String(value ?? "-")} {String(unit)}
                  </strong>
                </div>
              ))}
              <div className="nutrition-grid" style={{ marginTop: 12 }}>
                <div>
                  <span>칼로리</span>
                  <strong>{String(nutrition.calories ?? "-")} kcal</strong>
                </div>
              </div>
            </div>
          )}
        </Card>
      ) : !hasAnyMfdsNutrition ? (
        <Card title="영양성분 후보">
          <div className="state-box">
            영양성분 후보를 찾지 못했습니다. 음식 후보를 확인하면 더 정확한 분석이 가능합니다.
          </div>
        </Card>
      ) : null}
      {!isManual && (
        <Card title="내 상태에 맞춘 식단 관리 포인트">
          <div className="card-list">
            <div className="state-box">
              점수보다 현재 식단에서 조절하거나 보완할 성분을 중심으로 확인해 주세요. 이 내용은 진단이나 처방이
              아닌 생활관리 참고 정보입니다.
            </div>
            {recommendationLoading ? (
              <div className="state-box">식단 관리 포인트를 불러오는 중입니다.</div>
            ) : recommendationError ? (
              <div className="state-box">식단 평가를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.</div>
            ) : needsFoodConfirmation ? (
              <div className="state-box">
                음식 후보 확인이 필요합니다. 영양성분은 후보 기준이며, 음식명을 확인하면 더 정확한 식단 조언을 볼 수
                있습니다.
              </div>
            ) : null}
            {!recommendationLoading && !recommendationError && !hasManagementPoints && (
              <div className="state-box">
                건강정보를 입력하거나 식단 사진의 음식 후보가 확인되면 내 상태에 맞춘 식단 조언을 볼 수 있습니다.
              </div>
            )}
            {recommendation?.rag_comment?.summary && (
              <div className="mini-card">
                <strong>요약</strong>
                <span>{publicNutritionText(recommendation.rag_comment.summary)}</span>
              </div>
            )}
            {recommendation && recommendation.disease_context.length > 0 && (
              <div className="mini-card">
                <strong>연결된 건강상태 참고</strong>
                <div className="card-list">
                  {recommendation.disease_context.map((context) => (
                    <div className="mini-card" key={`disease-evaluation-${context.disease_code}`}>
                      <span className="badge badge-reference">{context.label}</span>
                      <span>{publicNutritionText(context.message)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {cautionFindings.length > 0 && (
              <div className="mini-card">
                <strong>주의하면 좋은 점</strong>
                <div className="card-list">
                  {cautionFindings.map((finding) => (
                    <div className="mini-card" key={`caution-${finding.issue_key}-${finding.label}`}>
                      <span className="badge badge-reference">{finding.label}</span>
                      <span>{publicNutritionText(finding.message)}</span>
                      {finding.basis && <span className="muted">{publicNutritionText(finding.basis)}</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {supportFindings.length > 0 && (
              <div className="mini-card">
                <strong>보완하면 좋은 점</strong>
                <div className="card-list">
                  {supportFindings.map((finding) => (
                    <div className="mini-card" key={`support-${finding.issue_key}-${finding.label}`}>
                      <span className="badge risk-low">{finding.label}</span>
                      <span>{publicNutritionText(finding.message)}</span>
                      {finding.basis && <span className="muted">{publicNutritionText(finding.basis)}</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {(recommendedFoods.length > 0 || cautionFoods.length > 0) && (
              <div className="mini-card">
                <strong>다음 식사에서 참고할 선택</strong>
                {recommendedFoods.length > 0 && (
                  <>
                    <span className="muted">보완하면 좋은 선택</span>
                    <div className="chip-list">
                      {recommendedFoods.map((food) => (
                        <span className="badge risk-low" key={`management-recommended-${food}`}>
                          {publicNutritionText(food)}
                        </span>
                      ))}
                    </div>
                  </>
                )}
                {cautionFoods.length > 0 && (
                  <>
                    <span className="muted">양을 조절해 볼 선택</span>
                    <div className="chip-list">
                      {cautionFoods.map((food) => (
                        <span className="badge badge-reference" key={`management-caution-${food}`}>
                          {publicNutritionText(food)}
                        </span>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </Card>
      )}
      <Card title="건강관리 추천">
        <div className="card-list">
          <div className="state-box">
            {publicNutritionText(recommendation?.safety_notice || DEFAULT_DIET_RECOMMENDATION_NOTICE)}
          </div>
          {recommendation?.rag_comment && (
            <div className="mini-card">
              <strong>참고 문서 기반 코멘트</strong>
              {recommendation.rag_comment.rewrite_used && <span className="badge badge-reference">문장 다듬기 적용</span>}
              <span>{publicNutritionText(recommendation.rag_comment.summary)}</span>
              {recommendation.rag_comment.disease_comments.length > 0 && (
                <div className="card-list">
                  {recommendation.rag_comment.disease_comments.map((comment) => (
                    <div className="mini-card" key={comment.disease_code}>
                      <span className="badge badge-reference">{comment.label}</span>
                      <span>{publicNutritionText(comment.comment)}</span>
                      <span className="muted">{publicNutritionText(comment.basis)}</span>
                    </div>
                  ))}
                </div>
              )}
              {recommendation.rag_comment.evidence_sources.length > 0 && (
                <div className="chip-list">
                  {recommendation.rag_comment.evidence_sources.map((source) => (
                    <span className="badge badge-reference" key={`${source.disease_code}-${source.title}`}>
                      참고 문서 기반: {source.title}
                    </span>
                  ))}
                </div>
              )}
              <span className="muted">{publicNutritionText(recommendation.rag_comment.safety_notice)}</span>
            </div>
          )}
          {recommendationLoading && <div className="state-box">식단 기반 건강관리 추천을 불러오는 중입니다.</div>}
          {recommendationError && (
            <div className="state-box">
              추천 정보를 불러오지 못했습니다. 기존 식단 분석 결과는 계속 확인할 수 있습니다.
            </div>
          )}
          {!recommendationLoading && !recommendationError && !hasDietRecommendationContent(recommendation) && (
            <div className="state-box">
              표시할 건강관리 추천이 아직 없습니다. 식단 사진의 음식명과 영양성분 후보가 확인되면 참고용 추천이 표시됩니다.
            </div>
          )}
          {recommendation && recommendation.nutrition_findings.length > 0 && (
            <div className="mini-card">
              <strong>영양 참고 포인트</strong>
              <div className="card-list">
                {recommendation.nutrition_findings.map((finding) => (
                  <div className="mini-card" key={`${finding.issue_key}-${finding.label}`}>
                    <span className="badge badge-reference">{finding.label}</span>
                    <span>{publicNutritionText(finding.message)}</span>
                    {finding.basis && <span className="muted">{publicNutritionText(finding.basis)}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}
          {recommendation && recommendation.disease_context.length > 0 && (
            <div className="mini-card">
              <strong>건강상태 참고</strong>
              <div className="card-list">
                {recommendation.disease_context.map((context) => (
                  <div className="mini-card" key={context.disease_code}>
                    <span className="badge badge-reference">{context.label}</span>
                    <span>{publicNutritionText(context.message)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {recommendation &&
            (recommendation.recommended_foods.length > 0 || recommendation.caution_foods.length > 0) && (
              <div className="mini-card">
                <strong>음식 선택 참고</strong>
                {recommendation.recommended_foods.length > 0 && (
                  <>
                    <span className="muted">보완하면 좋은 음식</span>
                    <div className="chip-list">
                      {recommendation.recommended_foods.map((food) => (
                        <span className="badge risk-low" key={`recommended-${food}`}>
                          {publicNutritionText(food)}
                        </span>
                      ))}
                    </div>
                  </>
                )}
                {recommendation.caution_foods.length > 0 && (
                  <>
                    <span className="muted">주의해서 살펴볼 음식</span>
                    <div className="chip-list">
                      {recommendation.caution_foods.map((food) => (
                        <span className="badge badge-reference" key={`caution-${food}`}>
                          {publicNutritionText(food)}
                        </span>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          {recommendation && recommendation.recommended_challenges.length > 0 && (
            <div className="mini-card">
              <strong>추천 챌린지</strong>
              <div className="card-list">
                {recommendation.recommended_challenges.map((challenge) => (
                  <div className="mini-card" key={challenge.challenge_id}>
                    <strong>{challenge.title}</strong>
                    <span className="muted">{truncateText(challenge.reason)}</span>
                    {challenge.challenge_id ? (
                      <Link className="button secondary compact-button" to={`/challenges/${challenge.challenge_id}`}>
                        챌린지 보기
                      </Link>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Card>
      <Card title="추천 액션">
        <div className="button-row">
          {dietRecordId && (
            <Link
              className="button secondary"
              to={`/chatbot?context_type=DIET&target_id=${dietRecordId}&initial_question=${encodeURIComponent("이 식단에서 조심할 점을 알려줘")}`}
            >
              이 식단에 대해 질문하기
            </Link>
          )}
          <button onClick={() => navigate("/diets/history")}>기록 완료</button>
          <Link className="button secondary" to="/dashboard">
            추적 대시보드 이동
          </Link>
          <Link className="button secondary" to="/challenges">
            추천 챌린지
          </Link>
        </div>
      </Card>
      <Card title="분석 요약">
        <div className="card-list">
          <div className="mini-card">
            <span className="muted">분석 일시</span>
            <strong>{formatDateTime(record?.meal_time ?? record?.created_at)}</strong>
          </div>
          <div className="mini-card">
            <span className="muted">식사 구분</span>
            <strong>{mealTypeLabel(record?.meal_type)}</strong>
          </div>
          <div className="mini-card">
            <span className="muted">기록 방식</span>
            <strong>{analysisMethodLabel(record?.analysis_method)}</strong>
          </div>
          <div className="mini-card">
            <span className="muted">식단 메모</span>
            <strong>{String(record?.description ?? record?.memo ?? "기록된 메모가 없습니다.")}</strong>
          </div>
          <div className="state-box">
            {isManual
              ? "기존 직접 입력 기록입니다. 새 식단 분석은 사진 업로드 기반으로 진행됩니다."
              : "자동 분석 결과는 참고용이며, 실제 진단이나 처방을 대신하지 않습니다."}
          </div>
        </div>
      </Card>
    </div>
  );
}
