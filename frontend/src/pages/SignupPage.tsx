import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { createHealthRecord, type HealthRecordPayload } from "../api/health";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

export default function SignupPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [loginId, setLoginId] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [birthDate, setBirthDate] = useState("");
  const [gender, setGender] = useState<"MALE" | "FEMALE">("MALE");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [lifestyle, setLifestyle] = useState({ smoking: "비흡연", drinking: "주 1회 이하", exercise: "주 3회" });
  const [extraHealth, setExtraHealth] = useState({ height: "", weight: "", sleepHours: "", diseaseHistory: "" });
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const steps = ["계정 정보", "기본 정보", "생활 습관", "추가 건강 정보"];
  const bmi =
    extraHealth.height && extraHealth.weight
      ? Number(extraHealth.weight) / (Number(extraHealth.height) / 100) ** 2
      : null;

  const buildInitialHealthPayload = (): HealthRecordPayload => {
    const height = Number(extraHealth.height);
    const weight = Number(extraHealth.weight);
    const sleepHours = Number(extraHealth.sleepHours);
    return {
      measured_at: new Date().toISOString(),
      ...(Number.isFinite(height) && height > 0 ? { height_cm: height } : {}),
      ...(Number.isFinite(weight) && weight > 0 ? { weight_kg: weight } : {}),
      ...(bmi ? { bmi: Number(bmi.toFixed(2)) } : {}),
      is_smoker: lifestyle.smoking === "현재 흡연",
      drinks_alcohol: lifestyle.drinking !== "주 1회 이하",
      exercise_days_per_week: lifestyle.exercise === "주 5회 이상" ? 5 : lifestyle.exercise === "주 3회" ? 3 : 1,
      ...(Number.isFinite(sleepHours) && sleepHours > 0 ? { sleep_hours: sleepHours } : {}),
    };
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setNotice("");
    if (step < steps.length - 1) {
      setStep((prev) => prev + 1);
      return;
    }
    if (password !== passwordConfirm) {
      setError("비밀번호 확인이 일치하지 않습니다.");
      setStep(0);
      return;
    }

    const normalizedPhoneNumber = phoneNumber.replace(/\D/g, "");
    try {
      await signup({
        login_id: loginId.trim(),
        email: email.trim(),
        password,
        name: name.trim(),
        gender,
        birth_date: birthDate,
        phone_number: normalizedPhoneNumber,
        nickname: name.trim(),
        sensitive_data_agreed: true,
      });
      try {
        await createHealthRecord<unknown>(buildInitialHealthPayload());
      } catch {
        setNotice("회원가입은 완료되었습니다. 초기 건강정보 저장은 실패했지만, 나중에 건강정보 화면에서 입력할 수 있습니다.");
        return;
      }
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "회원가입에 실패했습니다.");
    }
  };

  return (
    <div className="auth-page">
      <Card title="회원가입">
        {error && <ErrorMessage message={error} />}
        {notice && (
          <div className="state-box">
            {notice} <Link to="/health">건강정보 입력으로 이동</Link>
          </div>
        )}
        <div className="stepper">
          {steps.map((label, index) => (
            <button
              className={index === step ? "step active" : "step"}
              key={label}
              onClick={() => setStep(index)}
              type="button"
            >
              <span>{index + 1}</span>
              {label}
            </button>
          ))}
        </div>
        <form className="form" onSubmit={submit}>
          {step === 0 && (
            <>
              <label>
                아이디
                <input
                  value={loginId}
                  onChange={(event) => setLoginId(event.target.value)}
                  minLength={6}
                  maxLength={40}
                  required
                />
              </label>
              <button className="secondary" type="button" onClick={() => setError("중복확인 API 연결은 후속 구현 예정입니다.")}>
                아이디 중복확인
              </button>
              <label>
                이메일
                <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" required />
              </label>
              <label>
                비밀번호
                <input
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  type="password"
                  minLength={8}
                  required
                />
                <span className="muted">
                  8자 이상, 대문자/소문자/숫자/특수문자를 각각 1개 이상 포함해야 합니다.
                </span>
              </label>
              <label>
                비밀번호 확인
                <input
                  value={passwordConfirm}
                  onChange={(event) => setPasswordConfirm(event.target.value)}
                  type="password"
                  minLength={8}
                  required
                />
              </label>
            </>
          )}

          {step === 1 && (
            <>
              <label>
                이름
                <input value={name} onChange={(event) => setName(event.target.value)} maxLength={20} required />
              </label>
              <label>
                휴대폰 번호
                <input
                  value={phoneNumber}
                  onChange={(event) => setPhoneNumber(event.target.value)}
                  placeholder="010-1234-5678"
                  required
                />
                <span className="muted">하이픈이나 공백이 있어도 저장 시 숫자 형식으로 정리됩니다.</span>
              </label>
              <label>
                생년월일
                <input value={birthDate} onChange={(event) => setBirthDate(event.target.value)} type="date" required />
                <span className="muted">YYYY-MM-DD 형식으로 전송됩니다.</span>
              </label>
              <label>
                성별
                <select value={gender} onChange={(event) => setGender(event.target.value as "MALE" | "FEMALE")}>
                  <option value="MALE">남성</option>
                  <option value="FEMALE">여성</option>
                </select>
              </label>
            </>
          )}

          {step === 2 && (
            <>
              <label>
                흡연 여부
                <select
                  value={lifestyle.smoking}
                  onChange={(event) => setLifestyle((prev) => ({ ...prev, smoking: event.target.value }))}
                >
                  <option>비흡연</option>
                  <option>과거 흡연</option>
                  <option>현재 흡연</option>
                </select>
              </label>
              <label>
                음주 빈도
                <select
                  value={lifestyle.drinking}
                  onChange={(event) => setLifestyle((prev) => ({ ...prev, drinking: event.target.value }))}
                >
                  <option>주 1회 이하</option>
                  <option>주 2-3회</option>
                  <option>거의 매일</option>
                </select>
              </label>
              <label>
                운동 빈도
                <select
                  value={lifestyle.exercise}
                  onChange={(event) => setLifestyle((prev) => ({ ...prev, exercise: event.target.value }))}
                >
                  <option>주 0-1회</option>
                  <option>주 3회</option>
                  <option>주 5회 이상</option>
                </select>
              </label>
              <p className="placeholder">생활 습관 항목은 회원가입 후 초기 건강정보로 저장됩니다.</p>
            </>
          )}

          {step === 3 && (
            <>
              <label>
                키(cm)
                <input
                  value={extraHealth.height}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, height: event.target.value }))}
                  type="number"
                />
              </label>
              <label>
                몸무게(kg)
                <input
                  value={extraHealth.weight}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, weight: event.target.value }))}
                  type="number"
                />
              </label>
              <div className="state-box">
                BMI 자동 표시: {bmi ? bmi.toFixed(1) : "-"}
              </div>
              <label>
                수면 시간
                <input
                  value={extraHealth.sleepHours}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, sleepHours: event.target.value }))}
                  type="number"
                  step="0.5"
                  min="0"
                  placeholder="7"
                />
              </label>
              <label>
                당뇨/고혈압/이상지질 가족력 여부
                <select>
                  <option>없음</option>
                  <option>있음</option>
                  <option>모름</option>
                </select>
              </label>
              <label>
                추가 건강 메모
                <textarea
                  value={extraHealth.diseaseHistory}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, diseaseHistory: event.target.value }))}
                  placeholder="가족력, 복약, 관리 중인 질환 등을 적어주세요."
                />
              </label>
              <p className="placeholder">키, 몸무게, BMI, 수면 시간은 회원가입 후 초기 건강정보로 저장됩니다.</p>
            </>
          )}

          <div className="button-row">
            {step > 0 && (
              <button className="secondary" onClick={() => setStep((prev) => prev - 1)} type="button">
                이전
              </button>
            )}
            <button type="submit">{step === steps.length - 1 ? "회원가입" : "다음"}</button>
          </div>
        </form>
        <p className="muted">
          이미 계정이 있다면 <Link to="/login">로그인</Link>
        </p>
      </Card>
    </div>
  );
}
