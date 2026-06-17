import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { AnalysisMode, runAnalysisAsync } from "../api/analysis";
import { useAsyncJobPolling } from "../hooks/useAsyncJobPolling";

import {
  createHealthRecord,
  deleteHealthRecord,
  getAnalysisReadiness,
  getLatestHealthRecord,
  listHealthRecords,
  type HealthRecordPayload,
} from "../api/health";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ConfirmDialog from "../components/ConfirmDialog";
import ErrorMessage from "../components/ErrorMessage";
import HealthProfileForm, {
  healthProfileSectionTitles,
  type HealthProfileFormState,
} from "../components/HealthProfileForm";
import { useAnalysisFeedbackDialog } from "../hooks/useAnalysisFeedbackDialog";

type HealthRecord = Record<string, unknown>;
type HealthRecordSource = NonNullable<HealthRecordPayload["source"]>;
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

const initialForm: HealthProfileFormState = {
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

const steps = [
  "간편 분석 정보 입력",
  "정밀 분석 정보 입력",
];

const stepToSection: Record<number, string[]> = {
  0: [healthProfileSectionTitles[0], healthProfileSectionTitles[1], healthProfileSectionTitles[2]],
  1: [healthProfileSectionTitles[3]],
};

const healthFieldLabels: Record<string, string> = {
  height_cm: "키",
  weight_kg: "몸무게",
  bmi: "BMI",
  occupation_code: "직업군",
  family_htn: "고혈압 가족력",
  family_dm: "당뇨병 가족력",
  family_dyslipidemia: "이상지질혈증 이상 가족력",
  smoking_status: "현재 흡연 여부",
  drinking_frequency: "음주 빈도",
  drinking_amount: "한 번 음주량",
  walking_days_per_week: "걷기 일수",
  strength_days_per_week: "근력운동 일수",
  systolic_bp: "수축기 혈압",
  diastolic_bp: "이완기 혈압",
  fasting_glucose: "공복혈당",
  hba1c: "당화혈색소",
  total_cholesterol: "총콜레스테롤",
  triglyceride: "중성지방",
  hdl_cholesterol: "HDL 콜레스테롤",
  ldl_cholesterol: "LDL 콜레스테롤",
  waist_cm: "허리둘레",
};

const healthValueLabels: Record<string, Record<string, string>> = {
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
    SEVEN_PLUS: "7잔 이상",
  },
};

function toStringValue(value: unknown): string {
  return value === undefined || value === null ? "" : String(value);
}

function normalizeCode<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return allowed.includes(String(value) as T) ? (String(value) as T) : fallback;
}

function parseNumber(value: string): number | undefined {
  if (value.trim() === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function getUserBirth(user: { birthday?: string | null; birth_date?: string | null } | null): string {
  return user?.birthday ?? user?.birth_date ?? "";
}

function formFromRecord(
  record: HealthRecord | null,
  user: { gender?: string | null; birthday?: string | null; birth_date?: string | null } | null,
): HealthProfileFormState {
  return {
    ...initialForm,
    gender: user?.gender === "FEMALE" ? "FEMALE" : "MALE",
    birth_date: getUserBirth(user),
    occupation: toStringValue(record?.occupation_code),
    family_htn: normalizeCode(record?.family_htn, ["YES", "NO", "UNKNOWN"], "UNKNOWN"),
    family_dm: normalizeCode(record?.family_dm, ["YES", "NO", "UNKNOWN"], "UNKNOWN"),
    family_dyslipidemia: normalizeCode(record?.family_dyslipidemia, ["YES", "NO", "UNKNOWN"], "UNKNOWN"),
    height_cm: toStringValue(record?.height_cm),
    weight_kg: toStringValue(record?.weight_kg),
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
    walking_days: toStringValue(record?.walking_days_per_week),
    strength_days: toStringValue(record?.strength_days_per_week),
    systolic_bp: toStringValue(record?.systolic_bp),
    diastolic_bp: toStringValue(record?.diastolic_bp),
    fasting_glucose: toStringValue(record?.fasting_glucose),
    hba1c: toStringValue(record?.hba1c),
    total_cholesterol: toStringValue(record?.total_cholesterol),
    triglyceride: toStringValue(record?.triglyceride),
    hdl_cholesterol: toStringValue(record?.hdl_cholesterol),
    ldl_cholesterol: toStringValue(record?.ldl_cholesterol),
    waist_cm: toStringValue(record?.waist_cm),
  };
}

function buildHealthPayload(
  form: HealthProfileFormState,
  bmi: number | null,
  source: HealthRecordSource,
): HealthRecordPayload {
  const walkingDays = parseNumber(form.walking_days);
  const strengthDays = parseNumber(form.strength_days);
  const payload: HealthRecordPayload = {
    measured_at: new Date().toISOString(),
    source,
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

function validateHealthInput(form: HealthProfileFormState): string[] {
  const ranges: Array<[keyof HealthProfileFormState, string, number, number]> = [
    ["height_cm", "키", 50, 250],
    ["weight_kg", "몸무게", 20, 300],
    ["waist_cm", "허리둘레", 30, 200],
    ["systolic_bp", "수축기 혈압", 60, 250],
    ["diastolic_bp", "이완기 혈압", 30, 160],
    ["fasting_glucose", "공복혈당", 40, 500],
    ["hba1c", "당화혈색소", 3, 20],
    ["total_cholesterol", "총콜레스테롤", 50, 500],
    ["triglyceride", "중성지방", 20, 1000],
    ["hdl_cholesterol", "HDL 콜레스테롤", 10, 150],
    ["ldl_cholesterol", "LDL 콜레스테롤", 10, 400],
    ["walking_days", "걷기 일수", 0, 7],
    ["strength_days", "근력운동 일수", 0, 7],
  ];
  return ranges
    .map(([key, label, min, max]) => {
      const raw = String(form[key] ?? "").trim();
      if (!raw) {
        return "";
      }
      const value = Number(raw);
      if (!Number.isFinite(value) || value < min || value > max) {
        return `${label} (${min}~${max})`;
      }
      return "";
    })
    .filter(Boolean);
}

function formatDate(value: unknown): string {
  if (!value) {
    return "-";
  }
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString("ko-KR");
}

function formatDateTime(value: unknown): string {
  if (!value) {
    return "-";
  }
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("ko-KR");
}

function getValue(record: HealthRecord, key: string, unit = ""): string {
  const value = record[key];
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return `${String(value)}${unit}`;
}

function getDisplayValue(record: HealthRecord, key: string, unit = ""): string {
  const value = record[key];
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  const stringValue = String(value);
  return `${healthValueLabels[key]?.[stringValue] ?? stringValue}${unit}`;
}

function getSourceLabel(value: unknown): string {
  switch (String(value ?? "MANUAL").toUpperCase()) {
    case "OCR":
      return "검진표 OCR";
    case "PROFILE":
      return "한눈에 보기 수정";
    case "ANALYSIS_PREP":
      return "분석 전 저장";
    case "MANUAL":
    default:
      return "수기 입력";
  }
}

function getRecordSummary(record: HealthRecord): string {
  const items: string[] = [];
  const systolic = record.systolic_bp;
  const diastolic = record.diastolic_bp;
  if (systolic !== undefined && systolic !== null && diastolic !== undefined && diastolic !== null) {
    items.push(`혈압 ${systolic}/${diastolic} mmHg`);
  }
  if (record.weight_kg !== undefined && record.weight_kg !== null) {
    items.push(`체중 ${record.weight_kg}kg`);
  }
  if (record.fasting_glucose !== undefined && record.fasting_glucose !== null) {
    items.push(`공복혈당 ${record.fasting_glucose}mg/dL`);
  }
  if (record.ldl_cholesterol !== undefined && record.ldl_cholesterol !== null) {
    items.push(`LDL ${record.ldl_cholesterol}mg/dL`);
  }
  return items.length > 0 ? items.join(" · ") : "주요 수치 미입력";
}

const recordDetailItems: Array<{ key: string; label: string; unit?: string; display?: boolean }> = [
  { key: "height_cm", label: "키", unit: "cm" },
  { key: "weight_kg", label: "몸무게", unit: "kg" },
  { key: "bmi", label: "BMI" },
  { key: "waist_cm", label: "허리둘레", unit: "cm" },
  { key: "systolic_bp", label: "수축기 혈압", unit: "mmHg" },
  { key: "diastolic_bp", label: "이완기 혈압", unit: "mmHg" },
  { key: "fasting_glucose", label: "공복혈당", unit: "mg/dL" },
  { key: "hba1c", label: "당화혈색소", unit: "%" },
  { key: "total_cholesterol", label: "총콜레스테롤", unit: "mg/dL" },
  { key: "ldl_cholesterol", label: "LDL 콜레스테롤", unit: "mg/dL" },
  { key: "hdl_cholesterol", label: "HDL 콜레스테롤", unit: "mg/dL" },
  { key: "triglyceride", label: "중성지방", unit: "mg/dL" },
  { key: "occupation_code", label: "직업군" },
  { key: "family_htn", label: "고혈압 가족력", display: true },
  { key: "family_dm", label: "당뇨병 가족력", display: true },
  { key: "family_dyslipidemia", label: "이상지질혈증 가족력", display: true },
  { key: "smoking_status", label: "흡연", display: true },
  { key: "drinking_frequency", label: "음주 빈도", display: true },
  { key: "drinking_amount", label: "음주량", display: true },
  { key: "walking_days_per_week", label: "걷기 일수", unit: "일/주" },
  { key: "strength_days_per_week", label: "근력운동 일수", unit: "일/주" },
];

export default function HealthRecordPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { backendUser } = useAuth();
  const [form, setForm] = useState(initialForm);
  const [latestRecord, setLatestRecord] = useState<HealthRecord | null>(null);
  const [records, setRecords] = useState<HealthRecord[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<HealthRecord | null>(null);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [inputErrors, setInputErrors] = useState<string[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState<number | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [runningMode, setRunningMode] = useState<AnalysisMode | null>(null);
  const [analysisJobId, setAnalysisJobId] = useState<number | null>(null);
  const {
    clearFeedback,
    feedbackDialog: analysisFeedbackDialog,
    showFailure,
    showProcessing,
    showSuccess,
  } = useAnalysisFeedbackDialog();

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
      showSuccess({ message: "결과 화면에서 건강 관리 단계를 확인해 주세요." });
      window.setTimeout(() => navigate("/analysis"), 900);
    },
    onFailure: () => {
      setError("분석에 실패했습니다. 다시 시도해주세요.");
      showFailure();
      setRunningMode(null);
      setAnalysisJobId(null);
    },
    onTimeout: () => {
      setError("분석 시간이 초과됐습니다. 다시 시도해주세요.");
      showFailure();
      setRunningMode(null);
      setAnalysisJobId(null);
    },
  });

  const load = async () => {
    const [latest, list, readinessResult] = await Promise.all([
      getLatestHealthRecord<HealthRecord | null>(),
      listHealthRecords<HealthRecord[]>(),
      getAnalysisReadiness<Readiness>(),
    ]);
    setLatestRecord(latest);
    setRecords(list);
    setSelectedRecord((current) => {
      if (!current) {
        return null;
      }
      return list.find((record) => Number(record.id) === Number(current.id)) ?? null;
    });
    setReadiness(readinessResult);
    setForm(formFromRecord(latest, backendUser));
  };

  useEffect(() => {
    void load().catch(() => setError("건강정보를 불러오지 못했습니다."));
  }, [backendUser?.id]);

  useEffect(() => {
    const step = searchParams.get("step");
    if (step === "precision") {
      setActiveStep(1);
    }
    if (step === "basic") {
      setActiveStep(0);
    }
  }, [searchParams]);

  const save = async (
    options: { showNotice?: boolean; source?: HealthRecordSource } = {},
  ): Promise<HealthRecord | null> => {
    const showNoticeMessage = options.showNotice ?? true;
    setError("");
    setNotice("");
    setInputErrors([]);
    if (!hasMeaningfulHealthData(form)) {
      setInputErrors(["저장할 건강정보"]);
      setError("저장할 건강정보를 먼저 입력해주세요.");
      return null;
    }
    const validationErrors = validateHealthInput(form);
    if (validationErrors.length > 0) {
      setInputErrors(validationErrors);
      setError("입력값을 확인해 주세요.");
      return null;
    }
    const payload = buildHealthPayload(form, bmi, options.source ?? "MANUAL");
    try {
      setIsSaving(true);
      const saved = await createHealthRecord<HealthRecord>(payload);
      await load();
      if (showNoticeMessage) {
        setNotice("건강정보가 저장되었습니다.");
      }
      return saved;
    } catch (err) {
      setError(err instanceof Error ? err.message : "건강정보 저장에 실패했습니다.");
      throw err;
    } finally {
      setIsSaving(false);
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await save().catch((err) => setError(err instanceof Error ? err.message : "건강정보 저장에 실패했습니다."));
  };

  const runAnalysis = async (mode: AnalysisMode) => {
    setError("");
    setNotice("");
    clearFeedback();
    try {
      const saved = await save({ showNotice: false, source: "ANALYSIS_PREP" });
      if (!saved) {
        return;
      }
      const latestReadiness = await getAnalysisReadiness<Readiness>();
      setReadiness(latestReadiness);
      if (!latestReadiness.latest_health_record_id) {
        setInputErrors(["저장할 건강정보"]);
        setError("저장된 건강정보가 없습니다.");
        return;
      }
      if (!latestReadiness.is_ready) {
        const missingLabels = (latestReadiness.missing_basic_fields ?? latestReadiness.missing_fields ?? []).map(
          (field) => healthFieldLabels[field] ?? field,
        );
        setInputErrors(missingLabels);
        setError(`기본 분석에 필요한 정보가 부족합니다. 누락 항목: ${missingLabels.join(", ")}`);
        return;
      }
      setRunningMode(mode);
      showProcessing();
      const job = await runAnalysisAsync(latestReadiness.latest_health_record_id, mode);
      setAnalysisJobId(job.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "분석 실행에 실패했습니다.");
      showFailure();
      setRunningMode(null);
    }
  };

  const removeRecord = async () => {
    if (!deleteTargetId || isDeleting) {
      return;
    }
    setError("");
    setNotice("");
    try {
      setIsDeleting(true);
      await deleteHealthRecord(deleteTargetId);
      setDeleteTargetId(null);
      await load();
      setNotice("건강기록이 삭제되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "건강기록 삭제에 실패했습니다.");
    } finally {
      setIsDeleting(false);
    }
  };

  const missingBasicFields = readiness?.missing_basic_fields ?? readiness?.missing_fields ?? [];
  const missingPrecisionFields = (readiness?.missing_precision_fields ?? []).filter(
    (field) => field !== "hba1c" && field !== "당화혈색소",
  );

  return (
    <div className="page-stack">
      <header className="dashboard-header">
        <div>
          <h1>건강 분석</h1>
          <p>건강정보를 입력하고 만성질환 위험도를 분석합니다.</p>
        </div>
      </header>

      {/* 입력 단계 탭 */}
      <div className="health-tab-group" style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        <Link
          className="filter-tab"
          style={{ fontSize: "15px", padding: "8px 18px", textAlign: "center", alignSelf: "flex-start" }}
          to="/health/profile"
        >
          한눈에 보기
        </Link>

        <div className="health-tab-sub-group" style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          {steps.map((step, index) => (
            <button
              className={index === activeStep ? "filter-tab active" : "filter-tab"}
              key={step}
              onClick={() => setActiveStep(index)}
              style={{ fontSize: "15px", padding: "8px 18px" }}
              type="button"
            >
              {step}
            </button>
          ))}
        </div>
      </div>

    <div className="dashboard-grid" style={{ gridTemplateColumns: "1fr" }}>
      <Card
        title="건강정보 입력"
        actions={
          activeStep === 0 ? (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "13px" }}>
              <span className={readiness?.is_ready ? "success-text" : "warning-text"} style={{ fontWeight: 700 }}>
                {readiness?.is_ready ? "기본 분석 준비 완료" : "정보 부족"}
              </span>
              <span style={{ color: "var(--color-border)" }}>|</span>
              <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                {missingBasicFields.length === 0
                  ? <span className="badge badge-saved">누락 항목 없음</span>
                  : missingBasicFields.map((field) => (
                      <span className="badge badge-missing" key={field}>
                        {healthFieldLabels[field] ?? field}
                      </span>
                    ))
                }
              </div>
            </div>
          ) : undefined
        }
      >
        {error && <ErrorMessage message={error} />}
        {inputErrors.length > 0 && (
          <div className="state-box">
            <strong>확인이 필요한 항목</strong>
            <div className="chip-list" style={{ marginTop: 8 }}>
              {inputErrors.map((field) => (
                <span className="badge badge-missing" key={field}>
                  {field}
                </span>
              ))}
            </div>
          </div>
        )}
        {notice && <div className="state-box">{notice}</div>}
        {activeStep < 4 && (
          <div className="state-box ocr-hint" style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 12, padding: "8px 12px" }}>
            <div style={{ lineHeight: "1.4" }}>
              <p style={{ margin: 0 }}>직업군, 가족력, 신장, 체중, 흡연/음주/운동 정보를 입력하면 간편 분석을 진행할 수 있습니다.</p>
              <p style={{ margin: "2px 0 0" }}>정밀 분석 정보를 추가로 입력하시면 예측 정확도가 높아집니다.</p>
              {activeStep === 1 && (
                <p style={{ margin: "2px 0 0" }}>
                  AST, ALT, 감마GTP, 크레아티닌, eGFR, 혈색소 같은 확장 정밀검사 수치는 검진표 OCR 결과와 분석 상세에서 확인할 수 있습니다.
                </p>
              )}
              <p style={{ margin: "2px 0 0" }}>
                검진표 값은 기존 건강정보를 자동으로 덮어쓰지 않으며, 비어 있는 항목만 보충될 수 있습니다.
              </p>
            </div>
            {activeStep === 1 && (
              <Link className="button secondary" style={{ whiteSpace: "nowrap" }} to="/ocr/exam">
                검진표로 입력
              </Link>
            )}
          </div>
        )}
        <form className="form" onSubmit={submit}>
          <HealthProfileForm
            bmi={bmi ? bmi.toFixed(1) : ""}
            form={form}
            onChange={(key, value) => setForm((prev) => ({ ...prev, [key]: value }))}
            visibleSections={stepToSection[activeStep]}
          />
          <div className="button-row" style={{ justifyContent: "flex-end" }}>
            <button
              className="button secondary"
              disabled={isSaving || runningMode !== null}
              onClick={() => void save().catch(() => undefined)}
              style={{ padding: "10px 14px", fontSize: "14px" }}
              type="button"
            >
              {isSaving ? "저장 중..." : "저장"}
            </button>
            {activeStep === 0 && (
              <button
                disabled={isSaving || runningMode !== null}
                onClick={() => void runAnalysis("BASIC")}
                type="button"
              >
                {runningMode === "BASIC" ? "분석 중..." : "간편 분석 실행"}
              </button>
            )}
            {activeStep === 1 && (
              <button
                disabled={isSaving || runningMode !== null}
                onClick={() => void runAnalysis("PRECISION")}
                type="button"
              >
                {runningMode === "PRECISION" ? "분석 중..." : "정밀 분석 실행"}
              </button>
            )}
          </div>
        </form>
      </Card>
      <Card title="건강정보 기록">
        {records.length === 0 ? (
          <div className="state-box">아직 저장된 건강정보 기록이 없습니다.</div>
        ) : (
          <div className="page-stack">
            <div className="state-box">
              저장하거나 검진표 OCR 결과를 반영할 때마다 새 기록으로 남습니다. 분석에는 가장 최근에 저장된 기록이 기본으로 사용됩니다.
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>저장일</th>
                    <th>기록일</th>
                    <th>입력 방식</th>
                    <th>주요 수치</th>
                    <th>상세</th>
                  </tr>
                </thead>
                <tbody>
                  {records.map((record) => (
                    <tr key={String(record.id)}>
                      <td>{formatDateTime(record.created_at)}</td>
                      <td>{formatDate(record.measured_at)}</td>
                      <td>
                        <span className="badge badge-saved">{getSourceLabel(record.source)}</span>
                      </td>
                      <td>{getRecordSummary(record)}</td>
                      <td>
                        <button
                          className="btn-secondary"
                          onClick={() =>
                            setSelectedRecord((current) =>
                              Number(current?.id) === Number(record.id) ? null : record,
                            )
                          }
                          style={{ fontSize: "13px", padding: "4px 12px" }}
                          type="button"
                        >
                          {Number(selectedRecord?.id) === Number(record.id) ? "닫기" : "상세보기"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {selectedRecord && (
              <section className="profile-section">
                <div className="section-heading">
                  <h3>기록 상세</h3>
                  <p>
                    {formatDateTime(selectedRecord.created_at)} · {getSourceLabel(selectedRecord.source)}
                  </p>
                </div>
                <div className="readonly-health-grid">
                  {recordDetailItems.map((item) => (
                    <div className="readonly-health-item" key={item.key}>
                      <div className="item-header">
                        <span>{item.label}</span>
                      </div>
                      <div className="item-value-row">
                        <strong>
                          {item.display
                            ? getDisplayValue(selectedRecord, item.key, item.unit ? ` ${item.unit}` : "")
                            : getValue(selectedRecord, item.key, item.unit ? ` ${item.unit}` : "")}
                        </strong>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}
      </Card>
      {deleteTargetId && (
        <ConfirmDialog
          cancelLabel="취소"
          confirmLabel="삭제하기"
          message="잘못 입력한 건강기록을 삭제하시겠습니까? 삭제 후에는 최근 건강정보 목록에서 제거됩니다."
          onCancel={() => setDeleteTargetId(null)}
          onConfirm={() => void removeRecord()}
          title="건강기록 삭제 확인"
          tone="danger"
        />
      )}
      {analysisFeedbackDialog}
    </div>
    </div>
  );
}
