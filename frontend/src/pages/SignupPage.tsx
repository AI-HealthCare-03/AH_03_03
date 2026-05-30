import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import {
  checkEmail,
  checkLoginId,
  checkPhone,
  sendEmailVerification,
  verifyEmailCode,
} from "../api/auth";
import { createHealthRecord, type HealthRecordPayload } from "../api/health";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import OccupationHelpModal from "../components/OccupationHelpModal";

type AvailabilityCheck = {
  checkedValue: string;
  available: boolean;
  message: string;
};

export default function SignupPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [loginId, setLoginId] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [phoneParts, setPhoneParts] = useState({ first: "010", second: "", third: "" });
  const [birthDate, setBirthDate] = useState("");
  const [gender, setGender] = useState<"MALE" | "FEMALE">("MALE");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [privacyConsentAgreed, setPrivacyConsentAgreed] = useState(false);
  const [lifestyle, setLifestyle] = useState({
    smoking_status: "NON_SMOKER",
    drinking_frequency: "RARE",
    drinking_amount: "NONE",
    walking_days_per_week: "3",
    strength_days_per_week: "2",
  });
  const [extraHealth, setExtraHealth] = useState({
    occupation_code: "",
    family_htn: "UNKNOWN",
    family_dm: "UNKNOWN",
    family_dyslipidemia: "UNKNOWN",
    height: "",
    weight: "",
  });
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [signupCompleted, setSignupCompleted] = useState(false);
  const [healthInfoSaved, setHealthInfoSaved] = useState<boolean | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [isOccupationHelpOpen, setIsOccupationHelpOpen] = useState(false);
  const [loginIdCheck, setLoginIdCheck] = useState<AvailabilityCheck | null>(null);
  const [emailCheck, setEmailCheck] = useState<AvailabilityCheck | null>(null);
  const [phoneCheck, setPhoneCheck] = useState<AvailabilityCheck | null>(null);
  const [emailCode, setEmailCode] = useState("");
  const [emailDebugCode, setEmailDebugCode] = useState<string | null>(null);
  const [emailVerification, setEmailVerification] = useState<AvailabilityCheck | null>(null);
  const [checkingField, setCheckingField] = useState<
    "login_id" | "email" | "phone" | "email_send" | "email_verify" | null
  >(null);

  const steps = ["계정 정보", "기본 정보", "생활 습관", "간편 분석 정보"];
  const bmi =
    extraHealth.height && extraHealth.weight
      ? Number(extraHealth.weight) / (Number(extraHealth.height) / 100) ** 2
      : null;

  const buildInitialHealthPayload = (): HealthRecordPayload => {
    const height = Number(extraHealth.height);
    const weight = Number(extraHealth.weight);
    return {
      measured_at: new Date().toISOString(),
      occupation_code: extraHealth.occupation_code,
      family_htn: extraHealth.family_htn,
      family_dm: extraHealth.family_dm,
      family_dyslipidemia: extraHealth.family_dyslipidemia,
      ...(Number.isFinite(height) && height > 0 ? { height_cm: height } : {}),
      ...(Number.isFinite(weight) && weight > 0 ? { weight_kg: weight } : {}),
      ...(bmi ? { bmi: Number(bmi.toFixed(2)) } : {}),
      smoking_status: lifestyle.smoking_status,
      drinking_frequency: lifestyle.drinking_frequency,
      drinking_amount: lifestyle.drinking_amount,
      walking_days_per_week: Number(lifestyle.walking_days_per_week),
      strength_days_per_week: Number(lifestyle.strength_days_per_week),
    };
  };

  const passwordPolicyMessage = "8자 이상, 영문자 1개 이상, 숫자 1개 이상, 특수문자 1개 이상 포함해야 합니다.";

  const isPasswordValid = (value: string) =>
    value.length >= 8 && /[A-Za-z]/.test(value) && /[0-9]/.test(value) && /[^A-Za-z0-9]/.test(value);

  const normalizedPhoneNumber = `${phoneParts.first}${phoneParts.second}${phoneParts.third}`;
  const hasPhoneInput = Boolean(phoneParts.second || phoneParts.third);

  const setOnlyDigitsPhonePart = (key: keyof typeof phoneParts, value: string, maxLength: number) => {
    setPhoneParts((prev) => ({ ...prev, [key]: value.replace(/\D/g, "").slice(0, maxLength) }));
    setPhoneCheck(null);
    setFieldErrors((prev) => {
      const { phone_number, phone_number_check, ...rest } = prev;
      return rest;
    });
  };

  const validateCurrentStep = (targetStep = step): boolean => {
    const nextErrors: Record<string, string> = {};
    const normalizedLoginId = loginId.trim();

    if (targetStep === 0) {
      if (normalizedLoginId.length < 6) {
        nextErrors.login_id = "아이디는 6자 이상 입력해주세요.";
      }
      if (!loginIdCheck || loginIdCheck.checkedValue !== normalizedLoginId || !loginIdCheck.available) {
        nextErrors.login_id_check = "아이디 중복확인을 완료해주세요.";
      }
      if (!email.trim()) {
        nextErrors.email = "이메일을 입력해주세요.";
      }
      if (!emailCheck || emailCheck.checkedValue !== email.trim().toLowerCase() || !emailCheck.available) {
        nextErrors.email_check = "이메일 중복확인을 완료해주세요.";
      }
      if (
        !emailVerification ||
        emailVerification.checkedValue !== email.trim().toLowerCase() ||
        !emailVerification.available
      ) {
        nextErrors.email_verification = "이메일 인증을 완료해야 다음 단계로 이동할 수 있습니다.";
      }
      if (hasPhoneInput && (phoneParts.first.length < 2 || phoneParts.second.length < 3 || phoneParts.third.length !== 4)) {
        nextErrors.phone_number = "휴대폰 번호를 올바르게 입력해주세요.";
      }
      if (
        hasPhoneInput &&
        phoneParts.first.length >= 2 &&
        phoneParts.second.length >= 3 &&
        phoneParts.third.length === 4 &&
        (!phoneCheck || phoneCheck.checkedValue !== normalizedPhoneNumber || !phoneCheck.available)
      ) {
        nextErrors.phone_number_check = "휴대폰 번호 중복확인을 해주세요.";
      }
      if (!isPasswordValid(password)) {
        nextErrors.password = passwordPolicyMessage;
      }
      if (password !== passwordConfirm) {
        nextErrors.password_confirm = "비밀번호 확인이 일치하지 않습니다.";
      }
      if (!privacyConsentAgreed) {
        nextErrors.privacy_consent = "개인정보 수집·이용 안내를 확인하고 동의해주세요.";
      }
    }

    if (targetStep === 1) {
      if (!name.trim()) {
        nextErrors.name = "이름을 입력해주세요.";
      }
      if (!birthDate) {
        nextErrors.birth_date = "생년월일을 입력해주세요.";
      }
      if (!gender) {
        nextErrors.gender = "성별을 선택해주세요.";
      }
    }

    if (targetStep === 2) {
      if (!lifestyle.drinking_amount) {
        nextErrors.drinking_amount = "한 번 음주량을 선택해주세요.";
      }
      if (lifestyle.walking_days_per_week === "") {
        nextErrors.walking_days_per_week = "걷기 일수를 선택해주세요.";
      }
      if (lifestyle.strength_days_per_week === "") {
        nextErrors.strength_days_per_week = "근력운동 일수를 선택해주세요.";
      }
    }

    if (targetStep === 3) {
      if (!extraHealth.occupation_code) {
        nextErrors.occupation_code = "직업군을 선택해주세요.";
      }
      if (!extraHealth.height) {
        nextErrors.height = "키를 입력해주세요.";
      }
      if (!extraHealth.weight) {
        nextErrors.weight = "몸무게를 입력해주세요.";
      }
    }

    setFieldErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const validateAllSteps = (): boolean => {
    for (let index = 0; index < steps.length; index += 1) {
      if (!validateCurrentStep(index)) {
        setStep(index);
        return false;
      }
    }
    return true;
  };

  const handleCheckLoginId = async () => {
    setError("");
    const normalizedLoginId = loginId.trim();
    if (normalizedLoginId.length < 6) {
      setFieldErrors((prev) => ({ ...prev, login_id: "아이디는 6자 이상 입력해주세요." }));
      return;
    }
    try {
      setCheckingField("login_id");
      const result = await checkLoginId(normalizedLoginId);
      setLoginIdCheck({
        checkedValue: normalizedLoginId,
        available: result.available,
        message: result.message ?? (result.available ? "사용 가능한 아이디입니다." : "이미 사용 중인 아이디입니다."),
      });
      setFieldErrors((prev) => {
        const { login_id, login_id_check, ...rest } = prev;
        return rest;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "아이디 중복확인에 실패했습니다.");
    } finally {
      setCheckingField(null);
    }
  };

  const handleCheckEmail = async () => {
    setError("");
    const normalizedEmail = email.trim().toLowerCase();
    if (!normalizedEmail) {
      setFieldErrors((prev) => ({ ...prev, email: "이메일을 입력해주세요." }));
      return;
    }
    try {
      setCheckingField("email");
      const result = await checkEmail(normalizedEmail);
      setEmailCheck({
        checkedValue: normalizedEmail,
        available: result.available,
        message: result.message ?? (result.available ? "사용 가능한 이메일입니다." : "이미 사용 중인 이메일입니다."),
      });
      setEmailVerification(null);
      setEmailCode("");
      setEmailDebugCode(null);
      setFieldErrors((prev) => {
        const { email: _email, email_check, email_verification, ...rest } = prev;
        return rest;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "이메일 중복확인에 실패했습니다.");
    } finally {
      setCheckingField(null);
    }
  };

  const handleCheckPhone = async () => {
    setError("");
    if (!hasPhoneInput) {
      setFieldErrors((prev) => ({ ...prev, phone_number: "휴대폰 번호를 입력한 뒤 중복확인해주세요." }));
      return;
    }
    if (phoneParts.first.length < 2 || phoneParts.second.length < 3 || phoneParts.third.length !== 4) {
      setFieldErrors((prev) => ({ ...prev, phone_number: "휴대폰 번호를 올바르게 입력해주세요." }));
      return;
    }

    try {
      setCheckingField("phone");
      const result = await checkPhone(normalizedPhoneNumber);
      setPhoneCheck({
        checkedValue: normalizedPhoneNumber,
        available: result.available,
        message:
          result.message ?? (result.available ? "사용 가능한 휴대폰 번호입니다." : "이미 사용중인 휴대폰 번호입니다."),
      });
      setFieldErrors((prev) => {
        const { phone_number, phone_number_check, ...rest } = prev;
        return rest;
      });
    } catch (err) {
      setPhoneCheck(null);
      setError(err instanceof Error ? err.message : "휴대폰 번호 중복확인에 실패했습니다.");
    } finally {
      setCheckingField(null);
    }
  };

  const handleSendEmailVerification = async () => {
    setError("");
    const normalizedEmail = email.trim().toLowerCase();
    if (!emailCheck || emailCheck.checkedValue !== normalizedEmail || !emailCheck.available) {
      setFieldErrors((prev) => ({ ...prev, email_check: "이메일 중복확인을 먼저 완료해주세요." }));
      return;
    }
    try {
      setCheckingField("email_send");
      const result = await sendEmailVerification(normalizedEmail);
      setEmailVerification(null);
      setEmailCode(result.debug_code ?? "");
      setEmailDebugCode(result.debug_code ?? null);
      setNotice(
        result.debug_code
          ? "인증코드를 생성했습니다. 아래 개발/시연용 인증코드로 확인해주세요."
          : "인증코드를 발송했습니다. 이메일을 확인해주세요.",
      );
      setFieldErrors((prev) => {
        const { email_verification, ...rest } = prev;
        return rest;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "인증코드 발송에 실패했습니다.");
    } finally {
      setCheckingField(null);
    }
  };

  const handleVerifyEmailCode = async () => {
    setError("");
    const normalizedEmail = email.trim().toLowerCase();
    if (!emailCode.trim()) {
      setFieldErrors((prev) => ({ ...prev, email_verification: "인증코드를 입력해주세요." }));
      return;
    }
    try {
      setCheckingField("email_verify");
      const result = await verifyEmailCode(normalizedEmail, emailCode.trim());
      setEmailVerification({
        checkedValue: normalizedEmail,
        available: result.verified,
        message: result.verified ? "이메일 인증이 완료되었습니다." : "인증코드를 확인해주세요.",
      });
      setFieldErrors((prev) => {
        const { email_verification, ...rest } = prev;
        return rest;
      });
    } catch (err) {
      setEmailVerification({
        checkedValue: normalizedEmail,
        available: false,
        message: err instanceof Error ? err.message : "인증코드를 확인해주세요.",
      });
    } finally {
      setCheckingField(null);
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setNotice("");
    if (step < steps.length - 1) {
      if (!validateCurrentStep(step)) {
        return;
      }
      setStep((prev) => prev + 1);
      return;
    }
    if (!validateAllSteps()) {
      return;
    }

    try {
      await signup({
        login_id: loginId.trim(),
        email: email.trim(),
        password,
        name: name.trim(),
        gender,
        birth_date: birthDate,
        ...(hasPhoneInput ? { phone_number: normalizedPhoneNumber } : {}),
        nickname: name.trim(),
        privacy_consent_agreed: privacyConsentAgreed,
        sensitive_data_agreed: true,
      });
      try {
        await createHealthRecord<unknown>(buildInitialHealthPayload());
        setHealthInfoSaved(true);
      } catch {
        setHealthInfoSaved(false);
      }
      setSignupCompleted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "회원가입에 실패했습니다.");
    }
  };

  if (signupCompleted) {
    return (
      <div className="auth-page">
        <Card title="회원가입 완료">
          {healthInfoSaved === false && (
            <div className="state-box">
              회원가입은 완료되었습니다. 간편 분석 정보 저장은 나중에 건강정보 화면에서 다시 입력할 수 있습니다.
            </div>
          )}
          <div className="signup-complete-panel">
            <span className="badge badge-saved">가입 완료</span>
            <h2>간편 건강 분석을 시작할 준비가 되었습니다.</h2>
            <p>
              입력하신 x1 기본 정보는 간편 건강 분석과 맞춤 챌린지 추천에 사용됩니다. 검진표 수치나
              혈액검사 결과는 정밀 분석에서 추가로 반영됩니다.
            </p>
          </div>
          <div className="signup-ocr-choice">
            <div>
              <strong>건강검진표로 정밀 분석 정보 추가</strong>
              <p>
                건강검진 결과지가 있다면 혈압, 혈당, 콜레스테롤 등 정밀 분석 정보를 자동 입력할 수
                있습니다. 검진표가 없어도 간편 분석은 바로 이용할 수 있습니다.
              </p>
            </div>
            <div className="button-row">
              <Link className="button" to="/ocr/exam">
                건강검진표로 정밀 분석 정보 추가
              </Link>
              <button className="secondary" onClick={() => navigate("/")} type="button">
                나중에 입력하기
              </button>
            </div>
          </div>
        </Card>
      </div>
    );
  }

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
              disabled={index > step + 1}
              key={label}
              onClick={() => {
                if (index <= step) {
                  setStep(index);
                  return;
                }
                if (validateCurrentStep(step)) {
                  setStep(index);
                }
              }}
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
                  onChange={(event) => {
                    setLoginId(event.target.value);
                    setLoginIdCheck(null);
                  }}
                  minLength={6}
                  maxLength={40}
                  required
                />
                {fieldErrors.login_id && <span className="field-error">{fieldErrors.login_id}</span>}
              </label>
              <button
                className="secondary"
                disabled={checkingField === "login_id"}
                type="button"
                onClick={() => void handleCheckLoginId()}
              >
                {checkingField === "login_id" ? "확인 중..." : "아이디 중복확인"}
              </button>
              {loginIdCheck && (
                <div className={loginIdCheck.available ? "success-text" : "warning-text"}>{loginIdCheck.message}</div>
              )}
              {fieldErrors.login_id_check && <span className="field-error">{fieldErrors.login_id_check}</span>}
              <label>
                이메일
                <input
                  value={email}
                  onChange={(event) => {
                    setEmail(event.target.value);
                    setEmailCheck(null);
                    setEmailVerification(null);
                    setEmailCode("");
                    setEmailDebugCode(null);
                  }}
                  type="email"
                  required
                />
                {fieldErrors.email && <span className="field-error">{fieldErrors.email}</span>}
              </label>
              <button
                className="secondary"
                disabled={checkingField === "email"}
                type="button"
                onClick={() => void handleCheckEmail()}
              >
                {checkingField === "email" ? "확인 중..." : "이메일 중복확인"}
              </button>
              {emailCheck && <div className={emailCheck.available ? "success-text" : "warning-text"}>{emailCheck.message}</div>}
              {fieldErrors.email_check && <span className="field-error">{fieldErrors.email_check}</span>}
              <div className="signup-verification-panel">
                <button
                  className="secondary"
                  disabled={checkingField === "email_send"}
                  type="button"
                  onClick={() => void handleSendEmailVerification()}
                >
                  {checkingField === "email_send" ? "발송 중..." : "인증코드 발송"}
                </button>
                <label>
                  인증코드
                  <input
                    inputMode="numeric"
                    maxLength={6}
                    value={emailCode}
                    onChange={(event) => {
                      setEmailCode(event.target.value.replace(/\D/g, "").slice(0, 6));
                      setEmailVerification(null);
                    }}
                    placeholder="6자리 코드"
                  />
                </label>
                <button
                  className="secondary"
                  disabled={checkingField === "email_verify"}
                  type="button"
                  onClick={() => void handleVerifyEmailCode()}
                >
                  {checkingField === "email_verify" ? "확인 중..." : "인증코드 확인"}
                </button>
                {emailVerification && (
                  <div className={emailVerification.available ? "success-text" : "warning-text"}>
                    {emailVerification.message}
                  </div>
                )}
                {fieldErrors.email_verification && (
                  <span className="field-error">{fieldErrors.email_verification}</span>
                )}
                {emailDebugCode && (
                  <div className="state-box">
                    개발/시연용 인증코드: <strong>{emailDebugCode}</strong>
                  </div>
                )}
              </div>
              <label>
                휴대폰 번호 <span className="muted">(선택)</span>
                <div className="phone-input-grid">
                  <input
                    inputMode="numeric"
                    maxLength={3}
                    value={phoneParts.first}
                    onChange={(event) => setOnlyDigitsPhonePart("first", event.target.value, 3)}
                  />
                  <input
                    inputMode="numeric"
                    maxLength={4}
                    value={phoneParts.second}
                    onChange={(event) => setOnlyDigitsPhonePart("second", event.target.value, 4)}
                  />
                  <input
                    inputMode="numeric"
                    maxLength={4}
                    value={phoneParts.third}
                    onChange={(event) => setOnlyDigitsPhonePart("third", event.target.value, 4)}
                  />
                </div>
                <span className="muted">MVP 시연에서는 이메일 인증만 사용합니다. 휴대폰 번호는 선택 입력입니다.</span>
                {fieldErrors.phone_number && <span className="field-error">{fieldErrors.phone_number}</span>}
              </label>
              <button
                className="secondary"
                disabled={checkingField === "phone"}
                type="button"
                onClick={() => void handleCheckPhone()}
              >
                {checkingField === "phone" ? "확인 중..." : "휴대폰 번호 중복확인"}
              </button>
              {phoneCheck && <div className={phoneCheck.available ? "success-text" : "warning-text"}>{phoneCheck.message}</div>}
              {fieldErrors.phone_number_check && <span className="field-error">{fieldErrors.phone_number_check}</span>}
              <label>
                비밀번호
                <input
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  type="password"
                  minLength={8}
                  required
                />
                <span className="muted">{passwordPolicyMessage}</span>
                {fieldErrors.password && <span className="field-error">{fieldErrors.password}</span>}
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
                {fieldErrors.password_confirm && <span className="field-error">{fieldErrors.password_confirm}</span>}
              </label>
              <div className="state-box signup-privacy-consent">
                <strong>개인정보 수집·이용 안내</strong>
                <p>
                  AI HealthCare는 회원가입 및 건강관리 기능 제공을 위해 이름, 이메일, 휴대폰 번호, 닉네임 등
                  계정 정보를 수집할 수 있습니다. 또한 건강검진 결과, 복약 정보, 식단 기록, 챌린지 수행 기록 등
                  사용자가 직접 입력하거나 업로드한 건강 관련 정보를 분석 및 맞춤 안내 제공 목적으로 사용할 수 있습니다.
                </p>
                <p>
                  입력한 건강정보는 본인 계정의 건강관리 참고 정보를 제공하기 위한 목적으로 사용되며, 본인 외 다른
                  사용자에게 자동 공개되지 않습니다. 본 서비스의 분석 결과는 의료기관의 진단을 대체하지 않으며,
                  건강관리 참고용 정보입니다.
                </p>
                <label className="checkbox-row">
                  <input
                    checked={privacyConsentAgreed}
                    onChange={(event) => {
                      setPrivacyConsentAgreed(event.target.checked);
                      setFieldErrors((prev) => {
                        const { privacy_consent, ...rest } = prev;
                        return rest;
                      });
                    }}
                    type="checkbox"
                  />
                  위 내용을 확인하고 개인정보 수집·이용에 동의합니다.
                </label>
                {fieldErrors.privacy_consent && <span className="field-error">{fieldErrors.privacy_consent}</span>}
              </div>
            </>
          )}

          {step === 1 && (
            <>
              <label>
                이름
                <input value={name} onChange={(event) => setName(event.target.value)} maxLength={20} required />
                {fieldErrors.name && <span className="field-error">{fieldErrors.name}</span>}
              </label>
              <label>
                생년월일
                <input value={birthDate} onChange={(event) => setBirthDate(event.target.value)} type="date" required />
                <span className="muted">YYYY-MM-DD 형식으로 전송됩니다.</span>
                {fieldErrors.birth_date && <span className="field-error">{fieldErrors.birth_date}</span>}
              </label>
              <label>
                성별
                <select value={gender} onChange={(event) => setGender(event.target.value as "MALE" | "FEMALE")}>
                  <option value="MALE">남성</option>
                  <option value="FEMALE">여성</option>
                </select>
                {fieldErrors.gender && <span className="field-error">{fieldErrors.gender}</span>}
              </label>
            </>
          )}

          {step === 2 && (
            <>
              <div className="state-box signup-analysis-guide">
                <strong>간편 분석에 필요한 최소 생활습관 정보입니다.</strong>
                <p>입력하신 정보는 가입 직후 간편 건강 분석과 맞춤 챌린지 추천을 제공하기 위한 최소 항목입니다.</p>
                <p>검진표 수치나 혈액검사 결과는 정밀 분석에서 추가로 반영됩니다.</p>
              </div>
              <label>
                흡연 여부
                <select
                  value={lifestyle.smoking_status}
                  onChange={(event) => setLifestyle((prev) => ({ ...prev, smoking_status: event.target.value }))}
                >
                  <option value="NON_SMOKER">비흡연</option>
                  <option value="PAST_SMOKER">과거 흡연</option>
                  <option value="CURRENT_SMOKER">현재 흡연</option>
                </select>
              </label>
              <label>
                음주 빈도
                <select
                  value={lifestyle.drinking_frequency}
                  onChange={(event) => setLifestyle((prev) => ({ ...prev, drinking_frequency: event.target.value }))}
                >
                  <option value="RARE">월 1회 미만</option>
                  <option value="MONTHLY_2_4">월 2-4회</option>
                  <option value="WEEKLY_2_3">주 2-3회</option>
                  <option value="WEEKLY_4_PLUS">주 4회 이상</option>
                </select>
              </label>
              <label>
                한 번 음주량
                <select
                  value={lifestyle.drinking_amount}
                  onChange={(event) => setLifestyle((prev) => ({ ...prev, drinking_amount: event.target.value }))}
                >
                  <option value="NONE">마시지 않음</option>
                  <option value="ONE_TO_TWO">1-2잔</option>
                  <option value="THREE_TO_FOUR">3-4잔</option>
                  <option value="FIVE_TO_SIX">5-6잔</option>
                  <option value="SEVEN_PLUS">7잔 이상</option>
                </select>
              </label>
              <label>
                1주일간 걷기 일수
                <DaySelector
                  value={lifestyle.walking_days_per_week}
                  onChange={(value) => setLifestyle((prev) => ({ ...prev, walking_days_per_week: value }))}
                />
                {fieldErrors.walking_days_per_week && (
                  <span className="field-error">{fieldErrors.walking_days_per_week}</span>
                )}
              </label>
              <label>
                1주일간 근력운동 일수
                <DaySelector
                  value={lifestyle.strength_days_per_week}
                  onChange={(value) => setLifestyle((prev) => ({ ...prev, strength_days_per_week: value }))}
                />
                {fieldErrors.strength_days_per_week && (
                  <span className="field-error">{fieldErrors.strength_days_per_week}</span>
                )}
              </label>
              <p className="placeholder">생활 습관 항목은 간편 분석 정보로 저장됩니다.</p>
            </>
          )}

          {step === 3 && (
            <>
              <div className="state-box signup-analysis-guide">
                <strong>간편 분석에 필요한 신체계측과 가족력 정보입니다.</strong>
                <p>혈압, 혈당, 콜레스테롤, 허리둘레 같은 검진값은 회원가입에서 직접 입력하지 않습니다.</p>
                <p>가입 완료 후 건강검진표 등록을 선택하면 정밀 분석 정보를 추가할 수 있습니다.</p>
              </div>
              <label>
                <span className="field-label-row">
                  <span>직업군</span>
                  <em className="badge badge-required">필수</em>
                  <button
                    aria-label="직업군 선택 도움말 열기"
                    className="help-icon-button"
                    onClick={() => setIsOccupationHelpOpen(true)}
                    type="button"
                  >
                    ?
                  </button>
                </span>
                <select
                  value={extraHealth.occupation_code}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, occupation_code: event.target.value }))}
                >
                  <option value="">선택</option>
                  <option value="PROFESSIONAL">관리·전문직</option>
                  <option value="OFFICE">사무직</option>
                  <option value="SERVICE">서비스·판매직</option>
                  <option value="AGRICULTURE">농림어업</option>
                  <option value="MANUAL">기능·노무직</option>
                  <option value="STUDENT">학생</option>
                  <option value="HOMEMAKER">주부</option>
                  <option value="OTHER">무직/기타</option>
                </select>
                {fieldErrors.occupation_code && <span className="field-error">{fieldErrors.occupation_code}</span>}
              </label>
              <label>
                키(cm)
                <input
                  value={extraHealth.height}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, height: event.target.value }))}
                  type="number"
                />
                {fieldErrors.height && <span className="field-error">{fieldErrors.height}</span>}
              </label>
              <label>
                몸무게(kg)
                <input
                  value={extraHealth.weight}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, weight: event.target.value }))}
                  type="number"
                />
                {fieldErrors.weight && <span className="field-error">{fieldErrors.weight}</span>}
              </label>
              <div className="state-box">
                BMI 자동 표시: {bmi ? bmi.toFixed(1) : "-"}
              </div>
              <label>
                고혈압 가족력 여부
                <select
                  value={extraHealth.family_htn}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, family_htn: event.target.value }))}
                >
                  <option value="YES">있음</option>
                  <option value="NO">없음</option>
                  <option value="UNKNOWN">모름</option>
                </select>
              </label>
              <label>
                당뇨병 가족력 여부
                <select
                  value={extraHealth.family_dm}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, family_dm: event.target.value }))}
                >
                  <option value="YES">있음</option>
                  <option value="NO">없음</option>
                  <option value="UNKNOWN">모름</option>
                </select>
              </label>
              <label>
                콜레스테롤·중성지방 이상 가족력 여부
                <select
                  value={extraHealth.family_dyslipidemia}
                  onChange={(event) => setExtraHealth((prev) => ({ ...prev, family_dyslipidemia: event.target.value }))}
                >
                  <option value="YES">있음</option>
                  <option value="NO">없음</option>
                  <option value="UNKNOWN">모름</option>
                </select>
              </label>
              <p className="placeholder">
                키, 몸무게, BMI 자동 계산값, 직업군, 질병별 가족력은 간편 분석 정보로 저장됩니다.
              </p>
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
      {isOccupationHelpOpen && <OccupationHelpModal onClose={() => setIsOccupationHelpOpen(false)} />}
    </div>
  );
}

function DaySelector({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return (
    <div className="day-button-grid" role="group" aria-label="요일 선택">
      {[0, 1, 2, 3, 4, 5, 6, 7].map((day) => (
        <button
          className={value === String(day) ? "day-button active" : "day-button"}
          key={day}
          onClick={() => onChange(String(day))}
          type="button"
        >
          {day}일
        </button>
      ))}
    </div>
  );
}
