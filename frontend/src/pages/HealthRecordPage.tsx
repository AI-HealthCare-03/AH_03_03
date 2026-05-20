import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

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
  const navigate = useNavigate();
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
    const latestReadiness = await getAnalysisReadiness<Readiness>();
    if (!latestReadiness.is_ready) {
      setReadiness(latestReadiness);
      setError("분석에 필요한 필수 건강정보가 부족합니다. 필수 건강정보 관리 화면에서 보완해주세요.");
      return;
    }
    navigate("/analysis");
  };

  return (
    <div className="dashboard-grid">
      <Card title="입력 단계">
        <div className="card-list">
          {["기본 정보", "건강/생활 정보", "혈액/검진 정보", "복약 후"].map((step, index) => (
            <span className={index === 0 ? "filter-tab active" : "filter-tab"} key={step}>
              {step}
            </span>
          ))}
        </div>
      </Card>
      <Card title="건강정보 입력">
        {error && <ErrorMessage message={error} />}
        <div className="state-box">
          정확한 분석을 위해 키, 몸무게, 혈압, 혈당, 지질 수치를 함께 입력해주세요.
          <div className="button-row" style={{ marginTop: 12 }}>
            <Link className="button secondary" to="/health/profile">
              필수 건강정보 관리로 이동
            </Link>
          </div>
        </div>
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
          <div className="button-row">
            <button className="secondary" type="button">이전</button>
            <Link className="button secondary" to="/health/profile">추가 정보 입력하기</Link>
            <button type="submit">분석 실행</button>
          </div>
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
