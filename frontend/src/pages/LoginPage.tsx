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
      setError(err instanceof Error ? err.message : "로그인에 실패했습니다.");
    }
  };

  return (
    <div className="auth-page">
      <Card title="로그인">
        {error && <ErrorMessage message={error} />}
        <form className="form" onSubmit={submit}>
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
              minLength={6}
              required
            />
          </label>
          <button type="submit">로그인</button>
        </form>
        <p className="muted">
          계정이 없다면 <Link to="/signup">회원가입</Link>
        </p>
      </Card>
    </div>
  );
}
