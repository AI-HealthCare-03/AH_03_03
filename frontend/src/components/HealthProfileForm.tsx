export type HealthProfileFormState = {
  gender: "MALE" | "FEMALE";
  birth_date: string;
  height_cm: string;
  weight_kg: string;
  waist_cm: string;
  systolic_bp: string;
  diastolic_bp: string;
  fasting_glucose: string;
  postprandial_glucose: string;
  hba1c: string;
  total_cholesterol: string;
  triglyceride: string;
  hdl_cholesterol: string;
  ldl_cholesterol: string;
  smoking_status: "never" | "past" | "current";
  drinking_frequency: "rare" | "weekly" | "often";
  exercise_frequency: "low" | "medium" | "high";
  sleep_hours: string;
  education_level: string;
  income_level: string;
};

type HealthProfileFormProps = {
  form: HealthProfileFormState;
  bmi: string;
  onChange: (key: keyof HealthProfileFormState, value: string) => void;
};

const sections: Array<{
  title: string;
  description?: string;
  fields: Array<{
    key: keyof HealthProfileFormState;
    label: string;
    type?: "number" | "date" | "select";
    required?: boolean;
    options?: Array<{ value: string; label: string }>;
    placeholder?: string;
  }>;
}> = [
  {
    title: "기본 건강정보",
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
      { key: "height_cm", label: "키(cm)", type: "number", required: true },
      { key: "weight_kg", label: "몸무게(kg)", type: "number", required: true },
      { key: "waist_cm", label: "허리둘레(cm)", type: "number", required: true },
    ],
  },
  {
    title: "혈압/혈당",
    fields: [
      { key: "systolic_bp", label: "수축기 혈압", type: "number", required: true },
      { key: "diastolic_bp", label: "이완기 혈압", type: "number", required: true },
      { key: "fasting_glucose", label: "공복혈당", type: "number", required: true },
      { key: "postprandial_glucose", label: "식후혈당", type: "number", placeholder: "선택 입력" },
      { key: "hba1c", label: "당화혈색소", type: "number", required: true },
    ],
  },
  {
    title: "지질/혈액검사",
    fields: [
      { key: "total_cholesterol", label: "총콜레스테롤", type: "number", required: true },
      { key: "triglyceride", label: "중성지방", type: "number", required: true },
      { key: "hdl_cholesterol", label: "HDL", type: "number", required: true },
      { key: "ldl_cholesterol", label: "LDL", type: "number", required: true },
    ],
  },
  {
    title: "생활습관",
    fields: [
      {
        key: "smoking_status",
        label: "흡연 여부",
        type: "select",
        required: true,
        options: [
          { value: "never", label: "비흡연" },
          { value: "past", label: "과거 흡연" },
          { value: "current", label: "현재 흡연" },
        ],
      },
      {
        key: "drinking_frequency",
        label: "음주 빈도",
        type: "select",
        required: true,
        options: [
          { value: "rare", label: "주 1회 이하" },
          { value: "weekly", label: "주 2-3회" },
          { value: "often", label: "거의 매일" },
        ],
      },
      {
        key: "exercise_frequency",
        label: "운동 빈도",
        type: "select",
        required: true,
        options: [
          { value: "low", label: "주 0-1회" },
          { value: "medium", label: "주 3회" },
          { value: "high", label: "주 5회 이상" },
        ],
      },
      { key: "sleep_hours", label: "수면 시간", type: "number", required: true },
    ],
  },
  {
    title: "선택 설문",
    description: "선택 항목은 분석 정확도 향상을 위한 참고 정보이며, 입력하지 않아도 서비스 이용이 가능합니다.",
    fields: [
      { key: "education_level", label: "교육수준", placeholder: "선택 입력" },
      { key: "income_level", label: "소득수준", placeholder: "선택 입력" },
    ],
  },
];

export default function HealthProfileForm({ form, bmi, onChange }: HealthProfileFormProps) {
  return (
    <div className="page-stack">
      {sections.map((section) => (
        <section className="profile-section" key={section.title}>
          <div className="section-heading">
            <h3>{section.title}</h3>
            {section.description && <p>{section.description}</p>}
          </div>
          <div className="form two-col">
            {section.fields.map((field) => (
              <label key={field.key}>
                <span>
                  {field.label}{" "}
                  <em className={field.required ? "badge badge-required" : "badge badge-optional"}>
                    {field.required ? "필수" : "선택"}
                  </em>
                </span>
                {field.type === "select" ? (
                  <select value={form[field.key]} onChange={(event) => onChange(field.key, event.target.value)}>
                    {field.options?.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    min={field.type === "number" ? 0 : undefined}
                    onChange={(event) => onChange(field.key, event.target.value)}
                    placeholder={field.placeholder}
                    step={field.key === "hba1c" || field.key === "sleep_hours" ? "0.1" : "1"}
                    type={field.type ?? "text"}
                    value={form[field.key]}
                  />
                )}
              </label>
            ))}
            {section.title === "기본 건강정보" && (
              <div className="state-box">
                <strong>BMI 자동 계산</strong>
                <p>{bmi || "-"}</p>
              </div>
            )}
          </div>
        </section>
      ))}
    </div>
  );
}
