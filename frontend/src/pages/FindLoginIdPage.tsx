import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";

import { findLoginId } from "../api/auth";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

function onlyDigits(value: string): string {
  return value.replace(/\D/g, "");
}

export default function FindLoginIdPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [phone1, setPhone1] = useState("");
  const [phone2, setPhone2] = useState("");
  const [phone3, setPhone3] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<{ found: boolean; maskedLoginId: string | null; message: string } | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setResult(null);

    const trimmedName = name.trim();
    const trimmedEmail = email.trim();
    const phoneNumber = `${phone1}${phone2}${phone3}`;
    const hasPhone = phone1 || phone2 || phone3;

    if (!trimmedName) {
      setError("이름을 입력해주세요.");
      return;
    }
    if (!trimmedEmail && !hasPhone) {
      setError("이메일 또는 휴대폰 번호 중 하나를 입력해주세요.");
      return;
    }
    if (hasPhone && (phone1.length < 2 || phone2.length < 3 || phone3.length !== 4)) {
      setError("휴대폰 번호를 올바르게 입력해주세요.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await findLoginId({
        name: trimmedName,
        ...(trimmedEmail ? { email: trimmedEmail } : {}),
        ...(hasPhone ? { phone_number: phoneNumber } : {}),
      });
      setResult({
        found: response.found,
        maskedLoginId: response.masked_login_id,
        message: response.message,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "아이디 찾기에 실패했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <Card>
        <div className="page-header">
          <div>
            <span className="eyebrow">Health Ladder</span>
            <h1>아이디 찾기</h1>
            <p>가입 시 입력한 이름과 이메일 또는 휴대폰 번호를 입력해주세요.</p>
          </div>
        </div>

        {error ? <ErrorMessage message={error} /> : null}

        <form className="form" onSubmit={submit}>
          <label>
            이름
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="이름" required />
          </label>
          <label>
            이메일
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="가입한 이메일"
              type="email"
            />
          </label>
          <div className="field-group">
            <span className="field-label">휴대폰 번호</span>
            <div className="phone-input-grid">
              <input
                inputMode="numeric"
                maxLength={3}
                placeholder="010"
                value={phone1}
                onChange={(event) => setPhone1(onlyDigits(event.target.value).slice(0, 3))}
              />
              <input
                inputMode="numeric"
                maxLength={4}
                placeholder="1234"
                value={phone2}
                onChange={(event) => setPhone2(onlyDigits(event.target.value).slice(0, 4))}
              />
              <input
                inputMode="numeric"
                maxLength={4}
                placeholder="5678"
                value={phone3}
                onChange={(event) => setPhone3(onlyDigits(event.target.value).slice(0, 4))}
              />
            </div>
            <small className="muted">이메일 또는 휴대폰 번호 중 하나만 입력해도 됩니다.</small>
          </div>

          <button type="submit" disabled={isLoading}>
            {isLoading ? "확인 중..." : "아이디 찾기"}
          </button>
        </form>

        {result ? (
          <div className={result.found ? "result-card success" : "result-card"}>
            <strong>{result.message}</strong>
            {result.found && result.maskedLoginId ? <p>가입된 아이디: {result.maskedLoginId}</p> : null}
            {!result.found ? <p>입력 정보를 다시 확인해주세요.</p> : null}
            <Link className="button secondary" to="/login">
              로그인으로 돌아가기
            </Link>
          </div>
        ) : null}

        <p className="muted">
          비밀번호가 기억나지 않는다면 <Link to="/auth/password-reset">비밀번호 찾기</Link>
        </p>
      </Card>
    </div>
  );
}
