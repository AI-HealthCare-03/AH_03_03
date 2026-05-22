import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { getDietRecord, listDietPhotoResults } from "../api/diets";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import { formatDateTime, mealTypeLabel, scoreBadgeClass } from "../utils/format";

type Item = Record<string, unknown>;

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

export default function DietResultPage() {
  const { dietRecordId } = useParams();
  const navigate = useNavigate();
  const [record, setRecord] = useState<Item | null>(null);
  const [photoResults, setPhotoResults] = useState<Item[]>([]);
  const [error, setError] = useState("");

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
  const detectedFoods = normalizeFoods(record?.detected_foods ?? photoResults[0]?.detected_foods);
  const photoConfidence =
    photoResults[0]?.confidence_payload && typeof photoResults[0].confidence_payload === "object"
      ? (photoResults[0].confidence_payload as Record<string, unknown>)
      : {};

  useEffect(() => {
    const load = async () => {
      if (!dietRecordId) {
        return;
      }
      setError("");
      try {
        setRecord(await getDietRecord<Item>(Number(dietRecordId)));
        setPhotoResults(await listDietPhotoResults<Item[]>(Number(dietRecordId)));
      } catch (err) {
        setError(err instanceof Error ? err.message : "식단 분석 결과를 불러오지 못했습니다.");
      }
    };
    void load();
  }, [dietRecordId]);

  return (
    <div className="page-grid">
      {error && <ErrorMessage message={error} />}
      <Card
        title="식단 분석 결과"
        actions={
          <Link className="button secondary" to="/diets/history">
            전체 기록
          </Link>
        }
      >
        <div className="card-list">
          <div className="diet-record-meta">
            <span className="muted">{formatDateTime(record?.meal_time ?? record?.created_at)}</span>
            <span className="badge badge-reference">{mealTypeLabel(record?.meal_type)}</span>
          </div>
          {record?.description != null && <strong>{String(record.description)}</strong>}
          <div className="score-panel">
            <span>식단 점수</span>
            {record?.diet_score != null ? (
              <strong className={scoreBadgeClass(Number(record.diet_score))}>{String(record.diet_score)}점</strong>
            ) : (
              <strong>-</strong>
            )}
            <p>{String(record?.diet_feedback ?? "식단 분석 또는 수동 기록 결과가 여기에 표시됩니다.")}</p>
          </div>
        </div>
      </Card>
      <Card title="탐지 음식">
        <div className="chip-list">
          {detectedFoods.length === 0 && <div className="state-box">탐지된 음식 정보가 없습니다.</div>}
          {detectedFoods.map((food, index) => (
            <span className="badge badge-reference" key={`${String(food.name ?? "food")}-${index}`}>
              {String(food.name ?? "음식")} {food.confidence ? `${Math.round(Number(food.confidence) * 100)}%` : ""}
            </span>
          ))}
        </div>
      </Card>
      <Card title="영양 구성">
        {[
          ["탄수화물", Number(nutrition.carbohydrate_g ?? 0), "g"],
          ["단백질", Number(nutrition.protein_g ?? 0), "g"],
          ["지방", Number(nutrition.fat_g ?? 0), "g"],
          ["나트륨", Number(nutrition.sodium_mg ?? 0), "mg"],
        ].map(([label, value, unit]) => (
          <div className="bar-row" key={label}>
            <span>{label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${Math.min(Number(value), 100)}%` }} />
            </div>
            <strong>
              {Number(value) || "-"} {unit}
            </strong>
          </div>
        ))}
        <div className="nutrition-grid" style={{ marginTop: 12 }}>
          <div>
            <span>칼로리</span>
            <strong>{String(nutrition.calories ?? "-")} kcal</strong>
          </div>
          <div>
            <span>인식 신뢰도</span>
            <strong>
              {photoConfidence.average_confidence
                ? `${Math.round(Number(photoConfidence.average_confidence) * 100)}%`
                : "-"}
            </strong>
          </div>
        </div>
      </Card>
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
          <div className="state-box">자동 분석 결과는 참고용이며, 실제 진단이나 처방을 대신하지 않습니다.</div>
        </div>
      </Card>
    </div>
  );
}
