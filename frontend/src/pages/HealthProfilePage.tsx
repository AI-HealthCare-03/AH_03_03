import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AnalysisMode, runAnalysisAsync } from "../api/analysis";
import { useAsyncJobPolling } from "../hooks/useAsyncJobPolling";

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
  basic_ready?: boolean;
  precision_ready?: boolean;
  latest_health_record_id: number | null;
  missing_fields: string[];
  missing_basic_fields?: string[];
  missing_precision_fields?: string[];
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
  occupation: "",
  family_htn: "UNKNOWN",
  family_dm: "UNKNOWN",
  family_dyslipidemia: "UNKNOWN",
  height_cm: "",
  weight_kg: "",
  smoking_status: "NON_SMOKER",
  drinking_frequency: "RARE",
  drinking_amount: "",
  walking_days: "",
  strength_days: "",
  systolic_bp: "",
  diastolic_bp: "",
  fasting_glucose: "",
  hba1c: "",
  total_cholesterol: "",
  triglyceride: "",
  hdl_cholesterol: "",
  ldl_cholesterol: "",
  waist_cm: "",
  education_level: "",
  income_level: "",
};

const x1RequiredFields: Array<{ key: keyof HealthProfileFormState | "age" | "bmi"; label: string }> = [
  { key: "gender", label: "성별" },
  { key: "age", label: "나이" },
  { key: "occupation", label: "직업군" },
  { key: "family_htn", label: "고혈압 가족력 여부" },
  { key: "family_dm", label: "당뇨병 가족력 여부" },
  { key: "family_dyslipidemia", label: "콜레스테롤·중성지방 이상 가족력 여부" },
  { key: "height_cm", label: "신장" },
  { key: "weight_kg", label: "체중" },
  { key: "bmi", label: "BMI 자동 계산 가능 여부" },
  { key: "smoking_status", label: "현재 흡연 여부" },
  { key: "drinking_frequency", label: "1년간 음주 빈도" },
  { key: "drinking_amount", label: "한 번 음주량" },
  { key: "walking_days", label: "1주일간 걷기 일수" },
  { key: "strength_days", label: "1주일간 근력운동 일수" },
];

const x2Fields: Array<{ key: keyof HealthProfileFormState; label: string }> = [
  { key: "systolic_bp", label: "수축기 혈압" },
  { key: "diastolic_bp", label: "이완기 혈압" },
  { key: "fasting_glucose", label: "공복혈당" },
  { key: "total_cholesterol", label: "총콜레스테롤" },
  { key: "triglyceride", label: "중성지방" },
  { key: "hdl_cholesterol", label: "HDL 콜레스테롤" },
  { key: "ldl_cholesterol", label: "LDL 콜레스테롤" },
  { key: "waist_cm", label: "허리둘레" },
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
  hdl_cholesterol: "HDL 콜레스테롤",
  ldl_cholesterol: "LDL 콜레스테롤",
  occupation_code: "직업군",
  family_htn: "고혈압 가족력 여부",
  family_dm: "당뇨병 가족력 여부",
  family_dyslipidemia: "콜레스테롤·중성지방 이상 가족력 여부",
  smoking_status: "현재 흡연 여부",
  drinking_frequency: "1년간 음주 빈도",
  drinking_amount: "한 번 음주량",
  walking_days_per_week: "1주일간 걷기 일수",
  strength_days_per_week: "1주일간 근력운동 일수",
};

const readOnlySections: Array<{
  title: string;
  items: Array<{
    key: keyof HealthProfileFormState | "bmi";
    label: string;
    unit?: string;
    optional?: boolean;
    referenceOnly?: boolean;
  }>;
}> = [
  {
    title: "기본정보",
    items: [
      { key: "gender", label: "성별" },
      { key: "birth_date", label: "생년월일" },
      { key: "occupation", label: "직업군" },
    ],
  },
  {
    title: "가족력",
    items: [
      { key: "family_htn", label: "고혈압 가족력 여부" },
      { key: "family_dm", label: "당뇨병 가족력 여부" },
      { key: "family_dyslipidemia", label: "콜레스테롤·중성지방 이상 가족력 여부" },
    ],
  },
  {
    title: "신체계측",
    items: [
      { key: "height_cm", label: "신장", unit: "cm" },
      { key: "weight_kg", label: "체중", unit: "kg" },
      { key: "bmi", label: "BMI" },
    ],
  },
  {
    title: "생활습관",
    items: [
      { key: "smoking_status", label: "현재 흡연 여부" },
      { key: "drinking_frequency", label: "1년간 음주 빈도" },
      { key: "drinking_amount", label: "한 번 음주량" },
      { key: "walking_days", label: "1주일간 걷기 일수" },
      { key: "strength_days", label: "1주일간 근력운동 일수" },
    ],
  },
  {
    title: "정밀 검진값",
    items: [
      { key: "systolic_bp", label: "수축기 혈압", unit: "mmHg" },
      { key: "diastolic_bp", label: "이완기 혈압", unit: "mmHg" },
      { key: "fasting_glucose", label: "공복혈당", unit: "mg/dL" },
      { key: "hba1c", label: "당화혈색소", unit: "%", optional: true },
      { key: "total_cholesterol", label: "총콜레스테롤", unit: "mg/dL" },
      { key: "triglyceride", label: "중성지방", unit: "mg/dL" },
      { key: "hdl_cholesterol", label: "HDL 콜레스테롤", unit: "mg/dL" },
      { key: "ldl_cholesterol", label: "LDL 콜레스테롤", unit: "mg/dL" },
      { key: "waist_cm", label: "허리둘레", unit: "cm" },
    ],
  },
];

const displayValueMap: Partial<Record<keyof HealthProfileFormState, Record<string, string>>> = {
  gender: { MALE: "남성", FEMALE: "여성" },
  occupation: { OFFICE: "사무직", SERVICE: "서비스직", MANUAL: "생산/현장직", STUDENT: "학생", OTHER: "무직/기타" },
  family_htn: { YES: "있음", NO: "없음", UNKNOWN: "모름" },
  family_dm: { YES: "있음", NO: "없음", UNKNOWN: "모름" },
  family_dyslipidemia: { YES: "있음", NO: "없음", UNKNOWN: "모름" },
  smoking_status: { NON_SMOKER: "비흡연", PAST_SMOKER: "과거 흡연", CURRENT_SMOKER: "현재 흡연" },
  drinking_frequency: {
    RARE: "월 1회 미만",
    MONTHLY_2_4: "월 2-4회",
    WEEKLY_2_3: "주 2-3회",
    WEEKLY_4_PLUS: "주 4회 이상",
  },
  drinking_amount: {
    NONE: "마시지 않음",
    ONE_TO_TWO: "1-2잔",
    THREE_TO_FOUR: "3-4잔",
    FIVE_TO_SIX: "5-6잔",
    THREE_TO_SIX: "3-6잔",
    SEVEN_PLUS: "7잔 이상",
  },
};

function toStringValue(value: unknown): string {
  return value === undefined || value === null ? "" : String(value);
}

function normalizeCode<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return allowed.includes(String(value) as T) ? (String(value) as T) : fallback;
}

function getReadOnlyValue(form: HealthProfileFormState, key: keyof HealthProfileFormState | "bmi", bmi: number | null): string {
  if (key === "bmi") {
    return bmi ? bmi.toFixed(1) : "";
  }
  const raw = form[key];
  return displayValueMap[key]?.[raw] ?? raw;
}

function formFromRecord(record: HealthRecord | null, userGender?: string | null, userBirth?: string | null): HealthProfileFormState {
  const walkingDays = toStringValue(record?.walking_days_per_week);
  const strengthDays = toStringValue(record?.strength_days_per_week);
  return {
    ...emptyForm,
    gender: userGender === "FEMALE" ? "FEMALE" : "MALE",
    birth_date: userBirth ?? "",
    occupation: toStringValue(record?.occupation_code),
    family_htn: record?.family_htn === "YES" || record?.family_htn === "NO" ? record.family_htn : "UNKNOWN",
    family_dm: record?.family_dm === "YES" || record?.family_dm === "NO" ? record.family_dm : "UNKNOWN",
    family_dyslipidemia:
      record?.family_dyslipidemia === "YES" || record?.family_dyslipidemia === "NO"
        ? record.family_dyslipidemia
        : "UNKNOWN",
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
    smoking_status: normalizeCode(
      record?.smoking_status,
      ["NON_SMOKER", "PAST_SMOKER", "CURRENT_SMOKER"],
      "NON_SMOKER",
    ),
    drinking_frequency: normalizeCode(
      record?.drinking_frequency,
      ["RARE", "MONTHLY_2_4", "WEEKLY_2_3", "WEEKLY_4_PLUS"],
      "RARE",
    ),
    drinking_amount: toStringValue(record?.drinking_amount),
    walking_days: walkingDays,
    strength_days: strengthDays,
  };
}

function hasMeaningfulHealthData(form: HealthProfileFormState): boolean {
  const values = [
    form.occupation,
    form.height_cm,
    form.weight_kg,
    form.drinking_amount,
    form.walking_days,
    form.strength_days,
    form.systolic_bp,
    form.diastolic_bp,
    form.fasting_glucose,
    form.hba1c,
    form.total_cholesterol,
    form.triglyceride,
    form.hdl_cholesterol,
    form.ldl_cholesterol,
    form.waist_cm,
  ];
  return values.some((value) => value !== "") || [form.family_htn, form.family_dm, form.family_dyslipidemia].some((value) => value !== "UNKNOWN");
}

function parseNumber(value: string): number | undefined {
  if (value.trim() === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function buildHealthPayload(form: HealthProfileFormState, bmi: number | null): HealthRecordPayload {
  const walkingDays = parseNumber(form.walking_days);
  const strengthDays = parseNumber(form.strength_days);
  const payload: HealthRecordPayload = {
    measured_at: new Date().toISOString(),
    occupation_code: form.occupation,
    family_htn: form.family_htn,
    family_dm: form.family_dm,
    family_dyslipidemia: form.family_dyslipidemia,
    smoking_status: form.smoking_status,
    drinking_frequency: form.drinking_frequency,
    drinking_amount: form.drinking_amount,
    ...(walkingDays !== undefined ? { walking_days_per_week: walkingDays } : {}),
    ...(strengthDays !== undefined ? { strength_days_per_week: strengthDays } : {}),
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
  ];
  numericFields.forEach((field) => {
    const value = parseNumber(form[field]);
    if (value !== undefined) {
      (payload as Record<string, unknown>)[field] = value;
    }
  });
  if (bmi !== null) {
    payload.bmi = Number(bmi.toFixed(2));
  }
  return payload;
}

function validateForm(form: HealthProfileFormState): string | null {
  const ranges: Array<[keyof HealthProfileFormState, string, number, number]> = [
    ["height_cm", "신장", 50, 250],
    ["weight_kg", "체중", 20, 300],
    ["waist_cm", "허리둘레", 30, 200],
    ["systolic_bp", "수축기 혈압", 60, 250],
    ["diastolic_bp", "이완기 혈압", 30, 160],
    ["fasting_glucose", "공복혈당", 40, 500],
    ["total_cholesterol", "총콜레스테롤", 50, 500],
    ["triglyceride", "중성지방", 20, 1000],
    ["hdl_cholesterol", "HDL 콜레스테롤", 10, 150],
    ["ldl_cholesterol", "LDL 콜레스테롤", 10, 400],
    ["walking_days", "1주일간 걷기 일수", 0, 7],
    ["strength_days", "1주일간 근력운동 일수", 0, 7],
  ];
  for (const [key, label, min, max] of ranges) {
    const value = parseNumber(form[key]);
    if (value !== undefined && (value < min || value > max)) {
      return `${label} 값은 ${min}~${max} 범위로 입력해주세요.`;
    }
  }
  return null;
}

function getAge(birthDate: string): number | null {
  if (!birthDate) {
    return null;
  }
  const birth = new Date(birthDate);
  if (Number.isNaN(birth.getTime())) {
    return null;
  }
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age -= 1;
  }
  return age >= 0 ? age : null;
}

function isRequiredFilled(form: HealthProfileFormState, key: keyof HealthProfileFormState | "age" | "bmi", bmi: number | null): boolean {
  if (key === "age") {
    return getAge(form.birth_date) !== null;
  }
  if (key === "bmi") {
    return bmi !== null;
  }
  return form[key] !== "";
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
  const [runningMode, setRunningMode] = useState<AnalysisMode | null>(null);
  const [analysisJobId, setAnalysisJobId] = useState<number | null>(null);

  const bmi = useMemo(() => {
    const height = parseNumber(form.height_cm);
    const weight = parseNumber(form.weight_kg);
    return height && weight ? weight / (height / 100) ** 2 : null;
  }, [form.height_cm, form.weight_kg]);

  useAsyncJobPolling({
    jobId: analysisJobId,
    enabled: analysisJobId !== null && runningMode !== null,
    intervalMs: 1500,
    timeoutMs: 120000,
    onSuccess: async () => {
      setRunningMode(null);
      setAnalysisJobId(null);
      navigate("/analysis");
    },
    onFailure: () => {
      setError("분석에 실패했습니다. 다시 시도해주세요.");
      setRunningMode(null);
      setAnalysisJobId(null);
    },
    onTimeout: () => {
      setError("분석 시간이 초과됐습니다. 다시 시도해주세요.");
      setRunningMode(null);
      setAnalysisJobId(null);
    },
  });

  const completedRequiredCount = x1RequiredFields.filter(({ key }) => isRequiredFilled(form, key, bmi)).length;
  const missingRequiredLabels = x1RequiredFields
    .filter(({ key }) => !isRequiredFilled(form, key, bmi))
    .map(({ label }) => label);
  const completedX2Count = x2Fields.filter(({ key }) => form[key] !== "").length;
  const backendMissingLabels = (readiness?.missing_basic_fields ?? readiness?.missing_fields ?? []).map(
    (field) => backendMissingLabelMap[field] ?? field,
  );
  const precisionMissingLabels = (readiness?.missing_precision_fields ?? []).map(
    (field) => backendMissingLabelMap[field] ?? field,
  ).filter((label) => label !== "당화혈색소");

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
  }, [backendUser?.id]);

  const save = async () => {
    const validationError = validateForm(form);
    if (validationError) {
      setError(validationError);
      return;
    }
    if (!hasMeaningfulHealthData(form)) {
      setError("저장할 건강정보를 먼저 입력해주세요.");
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
    setError("");
    try {
      const latestReadiness = await getAnalysisReadiness<Readiness>();
      setReadiness(latestReadiness);
      if (!latestReadiness.latest_health_record_id) {
        setError("저장된 건강정보가 없습니다.");
        return;
      }
      if (!latestReadiness.is_ready) {
        setError("기본 분석에 필요한 정보가 부족합니다.");
        setEditing(true);
        return;
      }
      setRunningMode("PRECISION");
      const job = await runAnalysisAsync(latestReadiness.latest_health_record_id, "PRECISION");
      setAnalysisJobId(job.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "분석 실행에 실패했습니다.");
      setRunningMode(null);
    }
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
          <p>기본 건강 정보와 정밀 건강 정보를 한 화면에서 관리합니다.</p>
        </div>
        <div className="button-row">
          {!editing && (
            <button className="btn-primary" onClick={() => setEditing(true)} type="button">
              수정하기
            </button>
          )}
          <button className="btn-secondary" disabled={runningMode !== null} onClick={analyze} type="button">
            {runningMode === "PRECISION" ? "분석 중..." : "분석하기"}
          </button>
        </div>
      </div>
      {error && <ErrorMessage message={error} />}
      {notice && <div className="state-box">{notice}</div>}
      <div className="page-grid">
        <Card title="분석 준비도">
          <div className="readiness-card">
            <strong>
              기본 건강 정보 입력 {completedRequiredCount} / {x1RequiredFields.length}
            </strong>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${Math.round((completedRequiredCount / x1RequiredFields.length) * 100)}%` }}
              />
            </div>
            <p className={missingRequiredLabels.length === 0 ? "success-text" : "warning-text"}>
              {missingRequiredLabels.length === 0 ? "기본 분석 준비 완료" : "기본 분석에 필요한 항목을 더 입력해 주세요."}
            </p>
            <div className="chip-list">
              {missingRequiredLabels.map((label) => (
                <span className="badge badge-missing" key={label}>
                  {label}
                </span>
              ))}
              {missingRequiredLabels.length === 0 && (
                <span className="badge badge-saved">부족한 항목 없음</span>
              )}
            </div>
            <div className="state-box">
              정밀 건강 정보 입력 {completedX2Count} / {x2Fields.length}
              {backendMissingLabels.length > 0 && (
                <p>기본 분석 부족 항목: {backendMissingLabels.join(", ")}</p>
              )}
              {precisionMissingLabels.length > 0 && (
                <p>검진/혈액검사 수치는 선택 입력입니다. 추가 입력하면 정밀 분석 정확도가 높아집니다: {precisionMissingLabels.join(", ")}</p>
              )}
              <p>당화혈색소는 선택값입니다. 입력하면 정밀 분석에 함께 반영됩니다.</p>
            </div>
          </div>
        </Card>
        <Card title="저장 상태">
          <div className="profile-summary-grid">
            {[
              ["키", form.height_cm],
              ["몸무게", form.weight_kg],
              ["BMI", bmi ? bmi.toFixed(1) : ""],
              ["직업군", displayValueMap.occupation?.[form.occupation] ?? ""],
              [
                "가족력",
                `${displayValueMap.family_htn?.[form.family_htn]}/${displayValueMap.family_dm?.[form.family_dm]}/${displayValueMap.family_dyslipidemia?.[form.family_dyslipidemia]}`,
              ],
              ["혈압", form.systolic_bp && form.diastolic_bp ? `${form.systolic_bp}/${form.diastolic_bp}` : ""],
              ["공복혈당", form.fasting_glucose],
              ["HDL", form.hdl_cholesterol],
              ["LDL", form.ldl_cholesterol],
            ].map(([label, value]) => (
              <div className="profile-summary-item" key={label}>
                <span>{label}</span>
                <div className="value-row">
                  <strong>{value || "-"}</strong>
                  <em className={value ? "badge badge-saved" : "badge badge-missing"}>
                    {value ? "저장됨" : "미입력"}
                  </em>
                </div>
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
          <div className="page-stack">
            <div className="state-box">
              저장된 값을 읽기 전용으로 표시합니다. 변경하려면 수정하기를 눌러주세요.{" "}
              <Link to="/health">건강 분석 입력 화면으로 이동</Link>
            </div>
            {readOnlySections.map((section) => (
              <section className="profile-section" key={section.title}>
                <div className="section-heading">
                  <h3>{section.title}</h3>
                  {section.title === "정밀 검진값" && <p>당화혈색소는 선택값이며, 입력되어 있으면 분석에 함께 반영됩니다.</p>}
                </div>
                <div className="readonly-health-grid">
                  {section.items.map((item) => {
                    const value = getReadOnlyValue(form, item.key, bmi);
                    return (
                      <div className="readonly-health-item" key={`${section.title}-${item.key}`}>
                        <div className="item-header">
                          <span>{item.label}</span>
                          {item.optional ? (
                            <em className="badge badge-reference">선택</em>
                          ) : item.referenceOnly ? (
                            <em className="badge badge-reference">참고</em>
                          ) : (
                            <em className="badge badge-required">필수</em>
                          )}
                        </div>
                        <div className="item-value-row">
                          <strong>
                            {value || "-"}
                            {value && item.unit ? ` ${item.unit}` : ""}
                          </strong>
                          <em className={value ? "badge badge-saved" : "badge badge-missing"}>
                            {value ? "저장됨" : "미입력"}
                          </em>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
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
