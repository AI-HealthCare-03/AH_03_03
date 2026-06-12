import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import {
  createHealthRecord,
  deleteHealthRecord,
  getAnalysisReadiness,
  getLatestHealthRecord,
  listHealthRecords,
  updateHealthRecord,
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
  "기본 정보",
  "가족력/생활정보",
  "신체계측",
  "혈액/검진 정보",
];

const stepToSection: Record<number, string[]> = {
  0: [healthProfileSectionTitles[0]],
  1: [healthProfileSectionTitles[1]],
  2: [healthProfileSectionTitles[2]],
  3: [healthProfileSectionTitles[3]],
};

const healthFieldLabels: Record<string, string> = {
  height_cm: "키",
  weight_kg: "몸무게",
  bmi: "BMI",
  occupation_code: "직업군",
  family_htn: "고혈압 가족력",
  family_dm: "당뇨병 가족력",
  family_dyslipidemia: "콜레스테롤·중성지방 이상 가족력",
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

function formatDate(value: unknown): string {
  if (!value) {
    return "-";
  }
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString("ko-KR");
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

export default function HealthRecordPage() {
  const navigate = useNavigate();
  const { backendUser } = useAuth();
  const [form, setForm] = useState(initialForm);
  const [latestRecord, setLatestRecord] = useState<HealthRecord | null>(null);
  const [records, setRecords] = useState<HealthRecord[]>([]);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [deleteTargetId, setDeleteTargetId] = useState<number | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const bmi = useMemo(() => {
    const height = parseNumber(form.height_cm);
    const weight = parseNumber(form.weight_kg);
    return height && weight ? weight / (height / 100) ** 2 : null;
  }, [form.height_cm, form.weight_kg]);

  const load = async () => {
    const [latest, list, readinessResult] = await Promise.all([
      getLatestHealthRecord<HealthRecord | null>(),
      listHealthRecords<HealthRecord[]>(),
      getAnalysisReadiness<Readiness>(),
    ]);
    setLatestRecord(latest);
    setRecords(list);
    setReadiness(readinessResult);
    setForm(formFromRecord(latest, backendUser));
  };

  useEffect(() => {
    void load().catch(() => setError("건강정보를 불러오지 못했습니다."));
  }, [backendUser?.id]);

  const save = async () => {
    setError("");
    setNotice("");
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
    setNotice("건강정보가 저장되었습니다.");
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await save().catch((err) => setError(err instanceof Error ? err.message : "건강정보 저장에 실패했습니다."));
  };

  const runAnalysis = async () => {
    setError("");
    const latestReadiness = await getAnalysisReadiness<Readiness>();
    setReadiness(latestReadiness);
    if (!latestReadiness.latest_health_record_id) {
      setError("저장된 건강정보가 없습니다. 기본 정보를 저장한 뒤 분석을 실행해주세요.");
      return;
    }
    if (!latestReadiness.is_ready) {
      setError("기본 분석에 필요한 정보가 부족합니다.");
      return;
    }
    navigate("/analysis");
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
    <div className="dashboard-grid">
      <Card className="sticky-sidebar" title="입력 단계">
        <div className="card-list">
          {steps.map((step, index) => (
            <button
              className={index === activeStep ? "filter-tab active" : "filter-tab"}
              key={step}
              onClick={() => setActiveStep(index)}
              type="button"
            >
              {step}
            </button>
          ))}
        </div>

        {/* 분석 준비 상태 인라인으로 이동 */}
        <div className="analysis-readiness-panel" style={{ marginTop: 58 }}>
          <div className={`readiness-status ${readiness?.is_ready ? "success-text" : "warning-text"}`}>
            <strong>{readiness?.is_ready ? "기본 분석 준비 완료" : "정보 부족"}</strong>
          </div>
          <div className="chip-list readiness-chip-list">
            {missingBasicFields.map((field) => (
              <span className="badge badge-missing" key={field}>
                {healthFieldLabels[field] ?? field}
              </span>
            ))}
            {missingBasicFields.length === 0 && <span className="badge badge-saved">부족 항목 없음</span>}
          </div>
          <div className="state-box readiness-note">
            <p>검진/혈액검사 수치를 입력하면 정밀 분석 정확도가 높아집니다.</p>
            <div className="chip-list readiness-chip-list">
              {missingPrecisionFields.map((field) => (
                <span className="badge badge-reference" key={field}>
                  {healthFieldLabels[field] ?? field}
                </span>
              ))}
              {missingPrecisionFields.length === 0 && <span className="badge badge-saved">정밀 보강값 입력 완료</span>}
            </div>
          </div>
        </div>
      </Card>
      <Card title="건강정보 입력">
        {error && <ErrorMessage message={error} />}
        {notice && <div className="state-box">{notice}</div>}
        {activeStep < 4 && (
          <div className="state-box">
            직업군, 가족력, 신장, 체중, 흡연/음주/운동 정보를 입력하면 기본 위험도 분석을 실행할 수 있습니다.
            <p>혈압, 혈당, 콜레스테롤 수치는 정밀 분석 정확도를 높이는 선택 입력입니다.</p>
            <div className="button-row" style={{ marginTop: 12, justifyContent: "flex-end" }}>
              <Link className="button secondary" to="/ocr/exam">
                검진표로 입력
              </Link>
              <Link className="button secondary" to="/health/profile">
                필수 건강정보 관리로 이동
              </Link>
            </div>
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
            <button className="secondary" onClick={() => navigate(-1)} type="button">
              이전
            </button>
            <button className="secondary" type="submit">
              저장
            </button>
            <button disabled={activeStep === 4 && !readiness?.is_ready} onClick={runAnalysis} type="button">
              분석 실행
            </button>
          </div>
        </form>
      </Card>
      <div style={{ gridColumn: "2" }}>
      <Card title="최근 건강정보">
        <div className="card-list">
          {records.length === 0 && <div className="state-box">최근 건강정보가 없습니다.</div>}
          {records.slice(0, 3).map((record, index) => (
            <div className="health-record-summary-card" key={String(record.id ?? index)}>
              <div className="health-record-summary-header">
                <div>
                  <span className="muted">측정일</span>
                  <strong>{formatDate(record.measured_at ?? record.created_at)}</strong>
                </div>
                <span className="badge badge-reference">최근 기록 {index + 1}</span>
              </div>
              <div className="health-record-summary-grid">
                <div>
                  <span>키/몸무게/BMI</span>
                  <strong>
                    {getValue(record, "height_cm", "cm")} / {getValue(record, "weight_kg", "kg")} /{" "}
                    {getValue(record, "bmi")}
                  </strong>
                </div>
                <div>
                  <span>가족력/생활</span>
                  <strong>
                    {getDisplayValue(record, "family_htn")} · {getDisplayValue(record, "smoking_status")} ·{" "}
                    {getValue(record, "walking_days_per_week", "일 걷기")}
                  </strong>
                </div>
                <div>
                  <span>혈압</span>
                  <strong>
                    {getValue(record, "systolic_bp")} / {getValue(record, "diastolic_bp")} mmHg
                  </strong>
                </div>
                <div>
                  <span>콜레스테롤/중성지방</span>
                  <div className="health-record-lipid-list">
                    <span>
                      <em>총콜레스테롤</em>
                      <strong>{getValue(record, "total_cholesterol")}</strong>
                    </span>
                    <span>
                      <em>LDL</em>
                      <strong>{getValue(record, "ldl_cholesterol")}</strong>
                    </span>
                    <span>
                      <em>HDL</em>
                      <strong>{getValue(record, "hdl_cholesterol")}</strong>
                    </span>
                    <span>
                      <em>중성지방</em>
                      <strong>{getValue(record, "triglyceride")}</strong>
                    </span>
                  </div>
                </div>
              </div>
              {Boolean(record.id) && (
                <div className="button-row" style={{ justifyContent: "flex-end" }}>
                  <button
                    className="danger-ghost"
                    disabled={isDeleting}
                    onClick={() => setDeleteTargetId(Number(record.id))}
                    type="button"
                  >
                    삭제
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
      </div>
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
    </div>
  );
}
