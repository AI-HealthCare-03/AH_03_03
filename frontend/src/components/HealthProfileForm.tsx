import { useState } from "react";

import OccupationHelpModal from "./OccupationHelpModal";

export type HealthProfileFormState = {
  gender: "MALE" | "FEMALE";
  birth_date: string;
  occupation: string;
  family_htn: "YES" | "NO" | "UNKNOWN";
  family_dm: "YES" | "NO" | "UNKNOWN";
  family_dyslipidemia: "YES" | "NO" | "UNKNOWN";
  height_cm: string;
  weight_kg: string;
  smoking_status: "NON_SMOKER" | "PAST_SMOKER" | "CURRENT_SMOKER";
  drinking_frequency: "RARE" | "MONTHLY_2_4" | "WEEKLY_2_3" | "WEEKLY_4_PLUS";
  drinking_amount: string;
  walking_days: string;
  strength_days: string;
  systolic_bp: string;
  diastolic_bp: string;
  fasting_glucose: string;
  hba1c: string;
  total_cholesterol: string;
  triglyceride: string;
  hdl_cholesterol: string;
  ldl_cholesterol: string;
  waist_cm: string;
  education_level: string;
  income_level: string;
};

type HealthProfileFormProps = {
  form: HealthProfileFormState;
  bmi: string;
  onChange: (key: keyof HealthProfileFormState, value: string) => void;
  visibleSections?: string[];
};

type FieldConfig = {
  key: keyof HealthProfileFormState;
  label: string;
  type?: "number" | "date" | "select";
  required?: boolean;
  options?: Array<{ value: string; label: string }>;
  placeholder?: string;
};

const familyOptions = [
  { value: "YES", label: "있음" },
  { value: "NO", label: "없음" },
  { value: "UNKNOWN", label: "모름" },
];

export const healthProfileSectionTitles = [
  "기본 정보",
  "가족력/생활정보",
  "신체계측",
  "혈액/검진 정보",
] as const;

const dayOptions = [0, 1, 2, 3, 4, 5, 6, 7];

const sections: Array<{
  title: string;
  description?: string;
  fields: FieldConfig[];
  bmiAfter?: boolean;
}> = [
  {
    title: "기본 정보",
    fields: [
      {
        key: "gender",
        label: "성별",
        type: "select",
        required: true,
        options: [
          { value: "MALE", label: "남성" },
          { value: "FEMALE", label: "여성" },
        ],
      },
      { key: "birth_date", label: "생년월일", type: "date", required: true },
      {
        key: "occupation",
        label: "직업군",
        type: "select",
        required: true,
        options: [
          { value: "", label: "선택" },
          { value: "PROFESSIONAL", label: "관리·전문직" },
          { value: "OFFICE", label: "사무직" },
          { value: "SERVICE", label: "서비스·판매직" },
          { value: "AGRICULTURE", label: "농림어업" },
          { value: "MANUAL", label: "기능·노무직" },
          { value: "STUDENT", label: "학생" },
          { value: "HOMEMAKER", label: "주부" },
          { value: "OTHER", label: "무직/기타" },
        ],
      },
    ],
  },
  {
    title: "가족력/생활정보",
    description: "부/모/형제자매 세부 입력 없이 질병별 있음/없음/모름으로만 입력합니다.",
    fields: [
      { key: "family_htn", label: "고혈압 가족력 여부", type: "select", required: true, options: familyOptions },
      { key: "family_dm", label: "당뇨병 가족력 여부", type: "select", required: true, options: familyOptions },
      {
        key: "family_dyslipidemia",
        label: "이상지질혈증 이상 가족력 여부",
        type: "select",
        required: true,
        options: familyOptions,
      },
      {
        key: "smoking_status",
        label: "현재 흡연 여부",
        type: "select",
        required: true,
        options: [
          { value: "NON_SMOKER", label: "비흡연" },
          { value: "PAST_SMOKER", label: "과거 흡연" },
          { value: "CURRENT_SMOKER", label: "현재 흡연" },
        ],
      },
      {
        key: "drinking_frequency",
        label: "1년간 음주 빈도",
        type: "select",
        required: true,
        options: [
          { value: "RARE", label: "월 1회 미만" },
          { value: "MONTHLY_2_4", label: "월 2-4회" },
          { value: "WEEKLY_2_3", label: "주 2-3회" },
          { value: "WEEKLY_4_PLUS", label: "주 4회 이상" },
        ],
      },
      {
        key: "drinking_amount",
        label: "한 번 음주량",
        type: "select",
        required: true,
        options: [
          { value: "", label: "선택" },
          { value: "NONE", label: "마시지 않음" },
          { value: "ONE_TO_TWO", label: "1-2잔" },
          { value: "THREE_TO_FOUR", label: "3-4잔" },
          { value: "FIVE_TO_SIX", label: "5-6잔" },
          { value: "SEVEN_PLUS", label: "7잔 이상" },
        ],
      },
      { key: "walking_days", label: "1주일간 걷기 일수", required: true, placeholder: "0~7" },
      { key: "strength_days", label: "1주일간 근력운동 일수", required: true, placeholder: "0~7" },
    ],
  },
  {
    title: "신체계측",
    fields: [
      { key: "height_cm", label: "신장", type: "number", required: true, placeholder: "cm" },
      { key: "weight_kg", label: "체중", type: "number", required: true, placeholder: "kg" },
    ],
    bmiAfter: true,
  },
  {
    title: "혈액/검진 정보",
    description: "정밀 분석과 대시보드 추적에 사용하는 검진 수치입니다.",
    fields: [
      { key: "systolic_bp", label: "수축기 혈압", type: "number", placeholder: "mmHg" },
      { key: "diastolic_bp", label: "이완기 혈압", type: "number", placeholder: "mmHg" },
      { key: "fasting_glucose", label: "공복혈당", type: "number", placeholder: "mg/dL" },
      { key: "hba1c", label: "당화혈색소 (선택)", type: "number", placeholder: "%" },
      { key: "total_cholesterol", label: "총콜레스테롤", type: "number", placeholder: "mg/dL" },
      { key: "triglyceride", label: "중성지방", type: "number", placeholder: "mg/dL" },
      { key: "hdl_cholesterol", label: "HDL 콜레스테롤", type: "number", placeholder: "mg/dL" },
      { key: "ldl_cholesterol", label: "LDL 콜레스테롤", type: "number", placeholder: "mg/dL" },
      { key: "waist_cm", label: "허리둘레", type: "number", placeholder: "cm" },
    ],
  },
];

function isDayField(key: keyof HealthProfileFormState) {
  return key === "walking_days" || key === "strength_days";
}

export default function HealthProfileForm({ form, bmi, onChange, visibleSections }: HealthProfileFormProps) {
  const visibleSectionSet = visibleSections ? new Set<string>(visibleSections) : null;
  const visibleItems = visibleSectionSet ? sections.filter((section) => visibleSectionSet.has(section.title)) : sections;
  const [isOccupationHelpOpen, setIsOccupationHelpOpen] = useState(false);

  return (
    <div className="page-stack">
      {visibleItems.map((section) => (
        <section className="profile-section" key={section.title}>
          <div className="section-heading">
            <h3>{section.title}</h3>
            {section.description && <p>{section.description}</p>}
            {section.title === "혈액/검진 정보" && (
              <p>이 단계는 정밀 분석 정확도를 높이는 선택 입력입니다. 비워도 기본 분석은 진행할 수 있습니다.</p>
            )}
          </div>
          <div className="form two-col">
            {section.fields.map((field) => (
              <label key={field.key}>
                <span className="field-label-row">
                  <span>{field.label}</span>
                  <em className={field.required ? "badge badge-required" : "badge badge-optional"}>
                    {field.required ? "필수" : "선택"}
                  </em>
                  {field.key === "occupation" && (
                    <button
                      aria-label="직업군 선택 도움말 열기"
                      className="help-icon-button"
                      onClick={() => setIsOccupationHelpOpen(true)}
                      type="button"
                    >
                      ?
                    </button>
                  )}
                </span>
                {field.type === "select" ? (
                  <select value={form[field.key]} onChange={(event) => onChange(field.key, event.target.value)}>
                    {field.options?.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                ) : isDayField(field.key) ? (
                  <div className="day-button-grid" role="group" aria-label={field.label}>
                    {dayOptions.map((day) => (
                      <button
                        className={form[field.key] === String(day) ? "day-button active" : "day-button"}
                        key={day}
                        onClick={() => onChange(field.key, String(day))}
                        type="button"
                      >
                        {day}일
                      </button>
                    ))}
                  </div>
                ) : (
                  <input
                    min={field.type === "number" ? 0 : undefined}
                    max={undefined}
                    onChange={(event) => onChange(field.key, event.target.value)}
                    placeholder={field.placeholder}
                    step={field.key === "hba1c" ? "0.1" : "1"}
                    type={field.type ?? "text"}
                    value={form[field.key]}
                  />
                )}
              </label>
            ))}
            {section.bmiAfter && (
              <div className="readonly-calculated-field">
                <span>BMI 자동 계산 결과</span>
                <strong>{bmi || "-"}</strong>
                <em className="badge badge-reference">자동 계산</em>
              </div>
            )}
          </div>
        </section>
      ))}
      {isOccupationHelpOpen && <OccupationHelpModal onClose={() => setIsOccupationHelpOpen(false)} />}
    </div>
  );
}
