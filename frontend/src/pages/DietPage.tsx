import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { createDietRecord, listDietRecords, runDummyDietAnalysis } from "../api/diets";
import Card from "../components/Card";

type DietRecord = Record<string, unknown>;

export default function DietPage() {
  const [description, setDescription] = useState("");
  const [records, setRecords] = useState<DietRecord[]>([]);
  const [dummyResult, setDummyResult] = useState<Record<string, unknown> | null>(null);

  const load = async () => setRecords(await listDietRecords<DietRecord[]>());

  useEffect(() => {
    void load().catch(() => undefined);
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await createDietRecord({ description, meal_time: new Date().toISOString(), analysis_method: "MANUAL" });
    setDescription("");
    await load();
  };

  const dummyAnalyze = async () => {
    const result = await runDummyDietAnalysis<Record<string, unknown>>({
      description: description || "더미 식단 이미지",
      meal_time: new Date().toISOString(),
    });
    setDummyResult(result);
    await load();
  };

  return (
    <div className="page-grid">
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
            <span>실제 이미지 업로드/CV 분석은 후속 구현 예정이며, 현재는 더미 분석으로 시연합니다.</span>
          </div>
          <div className="button-row">
            <button type="submit">기록 저장</button>
            <button type="button" onClick={dummyAnalyze}>
              더미 분석
            </button>
          </div>
        </form>
      </Card>
      <Card title="더미 분석 결과">
        <pre>{JSON.stringify(dummyResult, null, 2)}</pre>
      </Card>
      <Card title="최근 식단">
        <div className="card-list">
          {records.slice(0, 5).map((record) => (
            <Link className="mini-card" key={String(record.id)} to={`/diets/${String(record.id)}`}>
              <strong>{String(record.meal_type ?? "식단 기록")}</strong>
              <span>점수: {String(record.diet_score ?? "-")}</span>
              <p>{String(record.description ?? "")}</p>
            </Link>
          ))}
        </div>
      </Card>
    </div>
  );
}
