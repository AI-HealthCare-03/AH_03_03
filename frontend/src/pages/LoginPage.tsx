import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

function formatLoginError(message: string): string {
  const normalizedMessage = message.trim();
  if (!normalizedMessage || normalizedMessage.startsWith("API 요청 실패")) {
    return "아이디 또는 이메일과 비밀번호를 확인해주세요.";
  }

  if (normalizedMessage.includes("Authenticate Failed") || normalizedMessage.includes("인증")) {
    return "아이디 또는 이메일과 비밀번호를 확인해주세요.";
  }

  return normalizedMessage;
}

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
      setError(formatLoginError(message));
    }
  };

  return (
    <div className="auth-page">
      <Card>
        <div className="page-header">
          <div>
            <span className="eyebrow">Health Ladder</span>
            <h1>다시 만나 반가워요!</h1>
            <p>아이디 또는 이메일로 로그인하여 건강 상태를 확인하세요.</p>
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
          계정이 없다면{" "}
          <Link className="signup-link" to="/signup">
            회원가입
          </Link>
        </p>
      </Card>
    </div>
  );
}
