import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { getDietRecord, getDietRecordImage, listDietPhotoResults, type DietCandidateFood } from "../api/diets";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import { formatDateTime, mealTypeLabel, scoreBadgeClass } from "../utils/format";

type Item = Record<string, unknown>;

const diseaseScoreLabels: Record<string, string> = {
  DM: "당뇨",
  HTN: "고혈압",
  DL: "콜레스테롤·중성지방",
  OBE: "비만",
  ANEM: "빈혈",
};

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

export default function DietResultPage() {
  const { dietRecordId } = useParams();
  const navigate = useNavigate();
  const [record, setRecord] = useState<Item | null>(null);
  const [photoResults, setPhotoResults] = useState<Item[]>([]);
  const [error, setError] = useState("");
  const [detailImageFailed, setDetailImageFailed] = useState(false);
  const [detailImageObjectUrl, setDetailImageObjectUrl] = useState("");

  const normalizeFoods = (value: unknown): Item[] => {
    if (Array.isArray(value)) {
      return value.filter((item): item is Item => Boolean(item) && typeof item === "object");
    }
    if (value && typeof value === "object") {
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
  const photoConfidence =
    photoResults[0]?.confidence_payload && typeof photoResults[0].confidence_payload === "object"
      ? (photoResults[0].confidence_payload as Record<string, unknown>)
      : {};
  const rawOutput = asRecord(photoResults[0]?.raw_output);
  const diseaseScores = diseaseScoreEntries(nutrition.disease_scores ?? rawOutput.disease_scores);
  const foodScoreDetails = Array.isArray(rawOutput.food_score_details)
    ? (rawOutput.food_score_details as Item[])
    : [];
  const scoringSource = scoringSourceLabel(nutrition.scoring_source ?? rawOutput.scoring_source);
  const candidateFoods = candidateFoodsFromPayload(rawOutput, record);
  const hasCandidateSection =
    candidateFoods.autoConfirmed.length > 0 ||
    candidateFoods.needsConfirmation.length > 0 ||
    candidateFoods.noCandidate.length > 0;
  const detailImageUrl = publicImageUrlFromPayloads(record, photoResults[0], rawOutput, photoConfidence);
  const displayImageUrl = detailImageObjectUrl || detailImageUrl;

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
    setDetailImageFailed(false);
  }, [detailImageUrl]);

  useEffect(() => {
    let objectUrl = "";
    let isMounted = true;
    setDetailImageObjectUrl("");

    if (!dietRecordId || !detailImageUrl || !detailImageUrl.startsWith("/api/v1/")) {
      return undefined;
    }

    const loadImage = async () => {
      try {
        const blob = await getDietRecordImage(Number(dietRecordId));
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
    <div className="page-grid">
      {error && <ErrorMessage message={error} />}
      <div style={{ marginBottom: "8px", gridColumn: "1 / -1" }}>
        <button className="button secondary" onClick={() => navigate(-1)} type="button">
          ← 이전으로
        </button>
      </div>
      <Card
        title={isManual ? "식단 직접 기록" : "식단 분석 결과"}
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
          {record?.description != null && <strong>{String(record.description)}</strong>}
          {!isManual && (
            <div className="mini-card">
              <span className="muted">분석 음식</span>
              <strong>
                {detectedFoods.length > 0 ? detectedFoods.map(foodDisplayName).join(", ") : "음식명 확인 불가"}
              </strong>
            </div>
          )}
          <div className="score-panel">
            <span>식단 점수</span>
            {record?.diet_score != null ? (
              <strong className={scoreBadgeClass(Number(record.diet_score))}>{String(record.diet_score)}점</strong>
            ) : isManual ? (
              <strong>점수 미산정</strong>
            ) : (
              <strong>-</strong>
            )}
            <p>
              {isManual
                ? "직접 입력 기록입니다."
                : String(record?.diet_feedback ?? "식단 분석 결과를 확인해보세요.")}
            </p>
          </div>
        </div>
      </Card>
      {isManual ? (
        <Card title="입력한 음식 목록">
          <div className="card-list">
            {detectedFoods.length === 0 && <div className="state-box">입력된 음식 목록이 없습니다.</div>}
            {detectedFoods.map((food, index) => (
              <div className="mini-card" key={`${String(food.name ?? "food")}-${index}`}>
                <strong>{String(food.name ?? "음식")}</strong>
                <span className="muted">
                  {[food.quantity ? `수량: ${String(food.quantity)}` : "", food.memo ? `메모: ${String(food.memo)}` : ""]
                    .filter(Boolean)
                    .join(" · ") || "추가 정보 없음"}
                </span>
              </div>
            ))}
          </div>
        </Card>
      ) : (
        <Card title="감지된 음식">
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
          </div>
        </Card>
      )}
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
                <span>직접 입력 필요</span>
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
      <Card title="영양 구성">
        {!hasNutrition ? (
          <div className="state-box">입력된 영양정보가 없습니다.</div>
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
              {!isManual && (
                <div>
                  <span>인식 신뢰도</span>
                  <strong>
                    {photoConfidence.average_confidence
                      ? `${Math.round(Number(photoConfidence.average_confidence) * 100)}%`
                      : "-"}
                  </strong>
                </div>
              )}
            </div>
          </div>
        )}
      </Card>
      {!isManual && (
        <Card title="질병군별 식단 점수">
          <div className="card-list">
            {diseaseScores.length === 0 && <div className="state-box">표시할 식단 점수가 없습니다.</div>}
            {diseaseScores.length > 0 && (
              <div className="nutrition-grid">
                {diseaseScores.map(([label, score]) => (
                  <div key={label}>
                    <span>{label}</span>
                    <strong>{Math.round(Number(score))}점</strong>
                  </div>
                ))}
              </div>
            )}
            {foodScoreDetails.slice(0, 4).map((detail, index) => {
              const detailScores = diseaseScoreEntries(detail.scores);
              return (
                <div className="mini-card" key={`${foodDisplayName(detail)}-${index}`}>
                  <strong>{foodDisplayName(detail)}</strong>
                  <span className="muted">
                    점수 기준: {String(detail.matched_food_name ?? "매칭 정보 없음")}
                  </span>
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
            {scoringSource && <span className="badge badge-reference">점수 기준: {scoringSource}</span>}
          </div>
        </Card>
      )}
      <Card title="추천 액션">
        <div className="button-row">
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
              ? "직접 입력 기록은 사용자가 입력한 내용을 기준으로 저장됩니다."
              : "자동 분석 결과는 참고용이며, 실제 진단이나 처방을 대신하지 않습니다."}
          </div>
        </div>
      </Card>
    </div>
  );
}
