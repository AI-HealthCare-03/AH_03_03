import { FormEvent, useEffect, useState } from "react";

import { createHealthRecord, getAnalysisReadiness, listHealthRecords } from "../api/health";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type HealthRecord = Record<string, unknown>;
type Readiness = {
  is_ready: boolean;
  latest_health_record_id: number | null;
  missing_fields: string[];
  message: string;
};

const initialForm = {
  height_cm: "",
  weight_kg: "",
  bmi: "",
  systolic_bp: "",
  diastolic_bp: "",
  fasting_glucose: "",
  hba1c: "",
  total_cholesterol: "",
  ldl_cholesterol: "",
  hdl_cholesterol: "",
  triglyceride: "",
  exercise_days_per_week: "",
  sleep_hours: "",
};

export default function HealthRecordPage() {
  const [form, setForm] = useState(initialForm);
  const [records, setRecords] = useState<HealthRecord[]>([]);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [error, setError] = useState("");

  const load = async () => {
    setRecords(await listHealthRecords<HealthRecord[]>());
    setReadiness(await getAnalysisReadiness<Readiness>());
  };

  useEffect(() => {
    void load().catch(() => setError("건강정보를 불러오지 못했습니다."));
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    const payload = Object.fromEntries(
      Object.entries(form).map(([key, value]) => [key, value === "" ? null : Number(value)]),
    );
    await createHealthRecord({
      ...payload,
      has_diabetes: false,
      has_obesity: false,
      has_dyslipidemia: false,
      has_hypertension: false,
      is_smoker: false,
      drinks_alcohol: false,
      measured_at: new Date().toISOString(),
    });
    await load();
  };

  return (
    <div className="page-grid">
      <Card title="건강정보 입력">
        {error && <ErrorMessage message={error} />}
        <form className="form two-col" onSubmit={submit}>
          {Object.keys(initialForm).map((key) => (
            <label key={key}>
              {key}
              <input
                value={form[key as keyof typeof initialForm]}
                onChange={(event) => setForm((prev) => ({ ...prev, [key]: event.target.value }))}
                type="number"
                step="0.01"
              />
            </label>
          ))}
          <button type="submit">저장</button>
        </form>
      </Card>
      <Card title="분석 준비 상태">
        <p className={readiness?.is_ready ? "success-text" : "warning-text"}>{readiness?.message}</p>
        <p>누락 항목: {readiness?.missing_fields.join(", ") || "없음"}</p>
      </Card>
      <Card title="최근 건강정보">
        <pre>{JSON.stringify(records.slice(0, 3), null, 2)}</pre>
      </Card>
    </div>
  );
}
