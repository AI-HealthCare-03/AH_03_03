import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import {
  createHealthRecord,
  getAnalysisReadiness,
  getLatestHealthRecord,
  updateHealthRecord,
  type HealthRecordPayload,
} from "../api/health";
import { useAuth } from "../auth/AuthContext";
import ConfirmDialog from "../components/ConfirmDialog";
import HealthProfileForm, { type HealthProfileFormState } from "../components/HealthProfileForm";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type HealthRecord = Record<string, unknown>;
type Readiness = {
  is_ready: boolean;
  latest_health_record_id: number | null;
  missing_fields: string[];
  message: string;
};

type DialogState =
  | { type: "cancel"; title: string; message: string }
  | { type: "save"; title: string; message: string }
  | { type: "reset-first"; title: string; message: string }
  | { type: "reset-second"; title: string; message: string }
  | null;

const emptyForm: HealthProfileFormState = {
  gender: "MALE",
  birth_date: "",
  height_cm: "",
  weight_kg: "",
  waist_cm: "",
  systolic_bp: "",
  diastolic_bp: "",
  fasting_glucose: "",
  postprandial_glucose: "",
  hba1c: "",
  total_cholesterol: "",
  triglyceride: "",
  hdl_cholesterol: "",
  ldl_cholesterol: "",
  smoking_status: "never",
  drinking_frequency: "rare",
  exercise_frequency: "medium",
  sleep_hours: "",
  education_level: "",
  income_level: "",
};

const requiredFields: Array<{ key: keyof HealthProfileFormState; label: string }> = [
  { key: "gender", label: "성별" },
  { key: "birth_date", label: "생년월일" },
  { key: "height_cm", label: "키" },
  { key: "weight_kg", label: "몸무게" },
  { key: "waist_cm", label: "허리둘레" },
  { key: "systolic_bp", label: "수축기 혈압" },
  { key: "diastolic_bp", label: "이완기 혈압" },
  { key: "fasting_glucose", label: "공복혈당" },
  { key: "hba1c", label: "당화혈색소" },
  { key: "total_cholesterol", label: "총콜레스테롤" },
  { key: "triglyceride", label: "중성지방" },
  { key: "hdl_cholesterol", label: "HDL" },
  { key: "ldl_cholesterol", label: "LDL" },
  { key: "smoking_status", label: "흡연 여부" },
  { key: "drinking_frequency", label: "음주 빈도" },
  { key: "exercise_frequency", label: "운동 빈도" },
  { key: "sleep_hours", label: "수면 시간" },
];

const backendMissingLabelMap: Record<string, string> = {
  height_cm: "키",
  weight_kg: "몸무게",
  bmi: "BMI",
  systolic_bp: "수축기 혈압",
  diastolic_bp: "이완기 혈압",
  fasting_glucose: "공복혈당",
  hba1c: "당화혈색소",
  total_cholesterol: "총콜레스테롤",
  triglyceride: "중성지방",
  hdl_cholesterol: "HDL",
  ldl_cholesterol: "LDL",
};

function toStringValue(value: unknown): string {
  return value === undefined || value === null ? "" : String(value);
}

function formFromRecord(record: HealthRecord | null, userGender?: string | null, userBirth?: string | null): HealthProfileFormState {
  return {
    ...emptyForm,
    gender: userGender === "FEMALE" ? "FEMALE" : "MALE",
    birth_date: userBirth ?? "",
    height_cm: toStringValue(record?.height_cm),
    weight_kg: toStringValue(record?.weight_kg),
    waist_cm: toStringValue(record?.waist_cm),
    systolic_bp: toStringValue(record?.systolic_bp),
    diastolic_bp: toStringValue(record?.diastolic_bp),
    fasting_glucose: toStringValue(record?.fasting_glucose),
    hba1c: toStringValue(record?.hba1c),
    total_cholesterol: toStringValue(record?.total_cholesterol),
    triglyceride: toStringValue(record?.triglyceride),
    hdl_cholesterol: toStringValue(record?.hdl_cholesterol),
    ldl_cholesterol: toStringValue(record?.ldl_cholesterol),
    smoking_status: record?.is_smoker ? "current" : "never",
    drinking_frequency: record?.drinks_alcohol ? "weekly" : "rare",
    exercise_frequency:
      Number(record?.exercise_days_per_week ?? 3) >= 5
        ? "high"
        : Number(record?.exercise_days_per_week ?? 3) >= 3
          ? "medium"
          : "low",
    sleep_hours: toStringValue(record?.sleep_hours),
  };
}

function parseNumber(value: string): number | undefined {
  if (value.trim() === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function buildHealthPayload(form: HealthProfileFormState, bmi: number | null): HealthRecordPayload {
  const payload: HealthRecordPayload = {
    measured_at: new Date().toISOString(),
    is_smoker: form.smoking_status === "current",
    drinks_alcohol: form.drinking_frequency !== "rare",
    exercise_days_per_week: form.exercise_frequency === "high" ? 5 : form.exercise_frequency === "medium" ? 3 : 1,
  };
  const numericFields: Array<keyof HealthRecordPayload & keyof HealthProfileFormState> = [
    "height_cm",
    "weight_kg",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "triglyceride",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "sleep_hours",
  ];
  numericFields.forEach((field) => {
    const value = parseNumber(form[field]);
    if (value !== undefined) {
      payload[field] = value;
    }
  });
  if (bmi !== null) {
    payload.bmi = Number(bmi.toFixed(2));
  }
  return payload;
}

function validateForm(form: HealthProfileFormState): string | null {
  const ranges: Array<[keyof HealthProfileFormState, string, number, number]> = [
    ["height_cm", "키", 50, 250],
    ["weight_kg", "몸무게", 20, 300],
    ["waist_cm", "허리둘레", 30, 200],
    ["systolic_bp", "수축기 혈압", 60, 250],
    ["diastolic_bp", "이완기 혈압", 30, 160],
    ["fasting_glucose", "공복혈당", 40, 500],
    ["total_cholesterol", "총콜레스테롤", 50, 500],
    ["triglyceride", "중성지방", 20, 1000],
    ["hdl_cholesterol", "HDL", 10, 150],
    ["ldl_cholesterol", "LDL", 10, 400],
  ];
  for (const [key, label, min, max] of ranges) {
    const value = parseNumber(form[key]);
    if (value !== undefined && (value < min || value > max)) {
      return `${label} 값은 ${min}~${max} 범위로 입력해주세요.`;
    }
  }
  return null;
}

export default function HealthProfilePage() {
  const { backendUser } = useAuth();
  const navigate = useNavigate();
  const [latestRecord, setLatestRecord] = useState<HealthRecord | null>(null);
  const [form, setForm] = useState<HealthProfileFormState>(emptyForm);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [editing, setEditing] = useState(false);
  const [dialog, setDialog] = useState<DialogState>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const bmi = useMemo(() => {
    const height = parseNumber(form.height_cm);
    const weight = parseNumber(form.weight_kg);
    return height && weight ? weight / (height / 100) ** 2 : null;
  }, [form.height_cm, form.weight_kg]);
  const completedRequiredCount = requiredFields.filter(({ key }) => form[key] !== "").length;
  const missingRequiredLabels = requiredFields.filter(({ key }) => form[key] === "").map(({ label }) => label);
  const backendMissingLabels = (readiness?.missing_fields ?? []).map((field) => backendMissingLabelMap[field] ?? field);

  const load = async () => {
    const [record, readinessResult] = await Promise.all([
      getLatestHealthRecord<HealthRecord | null>(),
      getAnalysisReadiness<Readiness>(),
    ]);
    setLatestRecord(record);
    setForm(formFromRecord(record, backendUser?.gender, backendUser?.birthday));
    setReadiness(readinessResult);
  };

  useEffect(() => {
    void load().catch(() => setError("건강정보를 불러오지 못했습니다."));
  }, []);

  const save = async () => {
    const validationError = validateForm(form);
    if (validationError) {
      setError(validationError);
      return;
    }
    const payload = buildHealthPayload(form, bmi);
    if (latestRecord?.id) {
      await updateHealthRecord(Number(latestRecord.id), payload);
    } else {
      await createHealthRecord(payload);
    }
    await load();
    setEditing(false);
    setNotice("건강정보가 저장되었습니다. 이후 분석에 반영됩니다.");
  };

  const analyze = async () => {
    const latestReadiness = await getAnalysisReadiness<Readiness>();
    setReadiness(latestReadiness);
    if (!latestReadiness.is_ready) {
      setError("분석에 필요한 필수 항목이 부족합니다. 부족한 항목을 입력한 뒤 다시 시도해주세요.");
      setEditing(true);
      return;
    }
    navigate("/analysis");
  };

  const handleConfirm = async () => {
    if (dialog?.type === "cancel") {
      setForm(formFromRecord(latestRecord, backendUser?.gender, backendUser?.birthday));
      setEditing(false);
    }
    if (dialog?.type === "save") {
      await save().catch((err) => setError(err instanceof Error ? err.message : "건강정보 저장에 실패했습니다."));
    }
    if (dialog?.type === "reset-first") {
      setDialog({
        type: "reset-second",
        title: "정말 초기화할까요?",
        message: "초기화된 정보는 복구할 수 없습니다. 정말 초기화하시겠습니까?",
      });
      return;
    }
    if (dialog?.type === "reset-second") {
      setForm(emptyForm);
      setEditing(true);
      setNotice("화면의 입력값을 초기화했습니다. 저장 전까지 서버 데이터는 유지됩니다.");
    }
    setDialog(null);
  };

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>필수 건강정보 관리</h1>
          <p>가입 시 입력한 정보와 분석에 필요한 건강정보를 한 화면에서 관리합니다.</p>
        </div>
        <div className="button-row">
          {!editing && (
            <button className="btn-primary" onClick={() => setEditing(true)} type="button">
              수정하기
            </button>
          )}
          <button className="btn-secondary" onClick={analyze} type="button">
            분석하기
          </button>
        </div>
      </div>
      {error && <ErrorMessage message={error} />}
      {notice && <div className="state-box">{notice}</div>}
      <div className="page-grid">
        <Card title="분석 준비도">
          <div className="readiness-card">
            <strong>
              필수 입력 {completedRequiredCount} / {requiredFields.length}
            </strong>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${Math.round((completedRequiredCount / requiredFields.length) * 100)}%` }}
              />
            </div>
            <p className={readiness?.is_ready ? "success-text" : "warning-text"}>
              {readiness?.is_ready ? "분석 준비 완료" : readiness?.message ?? "필수 항목을 확인해주세요."}
            </p>
            <div className="chip-list">
              {(backendMissingLabels.length > 0 ? backendMissingLabels : missingRequiredLabels).map((label) => (
                <span className="badge badge-missing" key={label}>
                  {label}
                </span>
              ))}
              {backendMissingLabels.length === 0 && missingRequiredLabels.length === 0 && (
                <span className="badge badge-saved">부족한 항목 없음</span>
              )}
            </div>
          </div>
        </Card>
        <Card title="저장 상태">
          <div className="profile-summary-grid">
            {[
              ["키", form.height_cm],
              ["몸무게", form.weight_kg],
              ["BMI", bmi ? bmi.toFixed(1) : ""],
              ["혈압", form.systolic_bp && form.diastolic_bp ? `${form.systolic_bp}/${form.diastolic_bp}` : ""],
              ["공복혈당", form.fasting_glucose],
              ["HDL", form.hdl_cholesterol],
              ["LDL", form.ldl_cholesterol],
            ].map(([label, value]) => (
              <div className="profile-summary-item" key={label}>
                <span>{label}</span>
                <strong>{value || "-"}</strong>
                <em className={value ? "badge badge-saved" : "badge badge-missing"}>{value ? "저장됨" : "미입력"}</em>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <Card title={editing ? "건강정보 수정" : "저장된 건강정보"}>
        {editing ? (
          <>
            <HealthProfileForm
              bmi={bmi ? bmi.toFixed(1) : ""}
              form={form}
              onChange={(key, value) => setForm((prev) => ({ ...prev, [key]: value }))}
            />
            <div className="button-row sticky-actions">
              <button
                className="btn-primary"
                onClick={() =>
                  setDialog({
                    type: "save",
                    title: "건강정보 저장",
                    message: "입력한 건강정보를 저장하면 이후 분석에 반영됩니다. 저장하시겠습니까?",
                  })
                }
                type="button"
              >
                저장
              </button>
              <button
                className="btn-secondary"
                onClick={() =>
                  setDialog({
                    type: "cancel",
                    title: "수정 취소",
                    message: "작성 중인 변경사항이 사라집니다. 취소하시겠습니까?",
                  })
                }
                type="button"
              >
                취소
              </button>
              <button
                className="btn-danger"
                onClick={() =>
                  setDialog({
                    type: "reset-first",
                    title: "건강정보 초기화",
                    message: "입력한 건강정보가 초기화됩니다. 계속하시겠습니까?",
                  })
                }
                type="button"
              >
                초기화
              </button>
            </div>
          </>
        ) : (
          <div className="state-box">
            저장된 값을 확인 중입니다. 변경하려면 수정하기를 눌러주세요.{" "}
            <Link to="/health">건강 분석 입력 화면으로 이동</Link>
          </div>
        )}
      </Card>
      {dialog && (
        <ConfirmDialog
          message={dialog.message}
          onCancel={() => setDialog(null)}
          onConfirm={() => void handleConfirm()}
          title={dialog.title}
          tone={dialog.type.includes("reset") ? "danger" : "default"}
        />
      )}
    </div>
  );
}
