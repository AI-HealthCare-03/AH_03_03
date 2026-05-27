import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";

import { requestPasswordReset } from "../api/auth";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

export default function PasswordResetRequestPage() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setMessage("");
    try {
      await requestPasswordReset(email.trim());
      setMessage("비밀번호 재설정 안내가 발송되었습니다. 이메일을 확인해주세요.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "비밀번호 재설정 요청에 실패했습니다.");
    }
  };

  return (
    <div className="auth-page">
      <Card>
        <div className="page-header">
          <div>
            <span className="eyebrow">Account Recovery</span>
            <h1>비밀번호 찾기</h1>
            <p>가입한 이메일을 입력하면 비밀번호 재설정 안내를 보내드립니다.</p>
          </div>
        </div>
        {error && <ErrorMessage message={error} />}
        {message && (
          <div className="state-box">
            {message}
          </div>
        )}
        <form className="form" onSubmit={submit}>
          <label>
            이메일
            <input
              autoComplete="email"
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              required
              type="email"
              value={email}
            />
          </label>
          <button type="submit">요청하기</button>
        </form>
        <p className="muted">
          기억났다면 <Link to="/login">로그인으로 돌아가기</Link>
        </p>
      </Card>
    </div>
  );
}
