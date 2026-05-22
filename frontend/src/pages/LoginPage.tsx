import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    try {
      await login(email, password);
      navigate("/");
    } catch (err) {
      const message = err instanceof Error ? err.message : "";
      setError(
        message.includes("Authenticate Failed") || message.includes("인증")
          ? "이메일 또는 비밀번호를 확인해주세요."
          : message || "로그인에 실패했습니다.",
      );
    }
  };

  return (
    <div className="auth-page">
      <Card>
        <div className="page-header">
          <div>
            <span className="eyebrow">HealthCare</span>
            <h1>다시 만나 반가워요!</h1>
            <p>아이디 또는 이메일로 로그인하고 오늘의 건강 상태를 확인하세요.</p>
          </div>
        </div>
        {error && <ErrorMessage message={error} />}
        <form className="form" onSubmit={submit}>
          <label>
            아이디 또는 이메일
            <input value={email} onChange={(event) => setEmail(event.target.value)} required />
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
          </label>
          <button type="submit">로그인</button>
        </form>
        <div className="button-row" style={{ marginTop: 12 }}>
          <Link className="muted" to="/auth/find-login-id">
            아이디 찾기
          </Link>
          <Link className="muted" to="/auth/password-reset">
            비밀번호 찾기
          </Link>
        </div>
        <p className="muted">
          계정이 없다면 <Link to="/signup">회원가입</Link>
        </p>
      </Card>
    </div>
  );
}
