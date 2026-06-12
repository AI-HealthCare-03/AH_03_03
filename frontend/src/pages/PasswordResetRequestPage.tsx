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

    const trimmedEmail = email.trim();

    if (!trimmedEmail) {
      setError("이메일을 입력해주세요.");
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmedEmail) || trimmedEmail.includes("..")) {
      setError("올바른 이메일 주소를 입력해주세요.");
      return;
    }

    try {
      await requestPasswordReset(trimmedEmail);
      setMessage("비밀번호 재설정 안내를 이메일로 발송했습니다.\n3분 이내에 메일이 도착하지 않으면 스팸메일함 또는 입력한 이메일 주소를 확인해주세요.");
    } catch {
      setError("입력한 이메일 정보를 확인해주세요.");
    }
  };

  return (
    <div className="auth-page">
      <Card>
        <div className="page-header">
          <div>
            <span className="eyebrow">HEALTH LADDER</span>
            <h1>비밀번호 찾기</h1>
            <p>가입 시 등록한 이메일을 입력하면 비밀번호 재설정 링크를 이메일로 보내드립니다.</p>
          </div>
        </div>
        {error && <ErrorMessage message={error} />}
        {message && (
          <div className="state-box">
            {message.split("\n").map((line, i) => (
              <span key={i}>{line}<br /></span>
            ))}
          </div>
        )}
        <form className="form" onSubmit={submit}>
          <label>
            이메일
            <input
              autoComplete="email"
              onChange={(event) => setEmail(event.target.value)}
              onFocus={() => setError("")}
              placeholder="example@example.com"
              required
              type="email"
              value={email}
            />
          </label>
          <button type="submit">재설정 링크 받기</button>
        </form>
        <p className="muted">
          <Link className={"signup-link"} to="/login">
            로그인
          </Link>{" "}
          페이지로 돌아가기
        </p>
      </Card>
    </div>
  );
}
