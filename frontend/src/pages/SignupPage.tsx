import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

export default function SignupPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    try {
      await signup(email, password);
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "회원가입에 실패했습니다.");
    }
  };

  return (
    <div className="auth-page">
      <Card title="회원가입">
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
          <button type="submit">Firebase로 가입</button>
        </form>
        <p className="muted">
          이미 계정이 있다면 <Link to="/login">로그인</Link>
        </p>
      </Card>
    </div>
  );
}
