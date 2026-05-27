import { FormEvent, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { confirmPasswordReset } from "../api/auth";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

export default function PasswordResetConfirmPage() {
  const [searchParams] = useSearchParams();
  const token = useMemo(() => searchParams.get("token") ?? "", [searchParams]);
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    if (!token) {
      setError("유효하지 않은 접근입니다.");
      return;
    }
    if (password !== passwordConfirm) {
      setError("새 비밀번호 확인이 일치하지 않습니다.");
      return;
    }
    try {
      await confirmPasswordReset({ token, new_password: password });
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "비밀번호 변경에 실패했습니다.");
    }
  };

  return (
    <div className="auth-page">
      <Card>
        <div className="page-header">
          <div>
            <span className="eyebrow">Password Reset</span>
            <h1>새 비밀번호 설정</h1>
            <p>새 비밀번호를 입력해 계정 보안을 다시 설정하세요.</p>
          </div>
        </div>
        {!token && <ErrorMessage message="유효하지 않은 접근입니다. 비밀번호 찾기 화면에서 다시 요청해주세요." />}
        {error && <ErrorMessage message={error} />}
        {success ? (
          <div className="state-box">
            비밀번호가 변경되었습니다. <Link to="/login">로그인 페이지로 이동</Link>
          </div>
        ) : (
          <form className="form" onSubmit={submit}>
            <label>
              새 비밀번호
              <input
                autoComplete="new-password"
                minLength={8}
                onChange={(event) => setPassword(event.target.value)}
                required
                type="password"
                value={password}
              />
              <span className="muted">8자 이상, 대문자/소문자/숫자/특수문자를 각각 1개 이상 포함해야 합니다.</span>
            </label>
            <label>
              새 비밀번호 확인
              <input
                autoComplete="new-password"
                minLength={8}
                onChange={(event) => setPasswordConfirm(event.target.value)}
                required
                type="password"
                value={passwordConfirm}
              />
            </label>
            <button disabled={!token} type="submit">
              변경하기
            </button>
          </form>
        )}
        <p className="muted">
          <Link to="/auth/password-reset">비밀번호 재설정 다시 요청</Link>
        </p>
      </Card>
    </div>
  );
}
