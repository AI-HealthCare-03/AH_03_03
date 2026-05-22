import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { createDietRecord, listDietRecords, runDummyDietAnalysis } from "../api/diets";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import { formatDateTime, mealTypeLabel, scoreBadgeClass } from "../utils/format";

type DietRecord = Record<string, unknown>;

export default function DietPage() {
  const [description, setDescription] = useState("");
  const [records, setRecords] = useState<DietRecord[]>([]);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

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

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    try {
      await createDietRecord({ description, meal_time: new Date().toISOString(), analysis_method: "MANUAL" });
      setDescription("");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "식단 기록 저장에 실패했습니다.");
    }
  };

  const dummyAnalyze = async () => {
    setError("");
    try {
      const result = await runDummyDietAnalysis<Record<string, unknown>>({
        description: description || "시연용 식단 이미지",
        meal_time: new Date().toISOString(),
      });
      setAnalysisResult(result);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "식단 분석에 실패했습니다.");
    }
  };

  const detectedFoods = Array.isArray(analysisResult?.detected_foods)
    ? (analysisResult.detected_foods as Record<string, unknown>[])
    : [];
  const nutrition =
    analysisResult?.nutrition_summary && typeof analysisResult.nutrition_summary === "object"
      ? (analysisResult.nutrition_summary as Record<string, unknown>)
      : {};
  const warnings = Array.isArray(analysisResult?.warnings) ? (analysisResult.warnings as string[]) : [];
  const recommendedActions = Array.isArray(analysisResult?.recommended_actions)
    ? (analysisResult.recommended_actions as string[])
    : [];

  return (
    <div className="page-grid">
      {error && <ErrorMessage message={error} />}
      <Card
        title="식단 이미지 분석"
        actions={
          <Link className="button secondary" to="/diets/history">
            결과 전체
          </Link>
        }
      >
        <form className="form" onSubmit={submit}>
          <label>
            식단 메모
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
          </label>
          <div className="upload-box">
            <strong>이미지 업로드 영역</strong>
            <span>현재 개발 환경에서는 예시 결과로 식단 분석 흐름을 확인할 수 있습니다.</span>
          </div>
          <div className="button-row">
            <button type="submit">기록 저장</button>
            <button type="button" onClick={dummyAnalyze}>
              간편 식단 분석
            </button>
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
              {detectedFoods.map((food) => (
                <span className="badge badge-reference" key={String(food.name)}>
                  {String(food.name ?? "음식")} {food.confidence ? `${Math.round(Number(food.confidence) * 100)}%` : ""}
                </span>
              ))}
            </div>
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
            return (
              <Link className="mini-card" key={String(record.id)} to={`/diets/${String(record.id)}`}>
                <div className="diet-record-meta">
                  <span className="muted">{formatDateTime(record.meal_time ?? record.created_at)}</span>
                  <span className="badge badge-reference">{mealTypeLabel(record.meal_type)}</span>
                </div>
                <strong>{String(record.description ?? "식단 기록")}</strong>
                {scoreRaw !== null && (
                  <span className={`badge ${scoreBadgeClass(scoreRaw)}`}>{scoreRaw}점</span>
                )}
              </Link>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
