import { FormEvent, useEffect, useState } from "react";

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
      <Card title="식단 기록">
        <form className="form" onSubmit={submit}>
          <label>
            식단 메모
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
          </label>
          <div className="placeholder">이미지 업로드/CV 분석은 후속 구현 예정</div>
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
        <pre>{JSON.stringify(records.slice(0, 5), null, 2)}</pre>
      </Card>
    </div>
  );
}
