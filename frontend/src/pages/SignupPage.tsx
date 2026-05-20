import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

export default function SignupPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [loginId, setLoginId] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [birthDate, setBirthDate] = useState("");
  const [gender, setGender] = useState<"MALE" | "FEMALE">("MALE");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    try {
      await signup({
        login_id: loginId,
        email,
        password,
        name,
        gender,
        birth_date: birthDate,
        phone_number: phoneNumber,
        nickname: name,
        sensitive_data_agreed: true,
      });
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
            아이디
            <input
              value={loginId}
              onChange={(event) => setLoginId(event.target.value)}
              minLength={6}
              maxLength={40}
              required
            />
          </label>
          <label>
            이름
            <input value={name} onChange={(event) => setName(event.target.value)} maxLength={20} required />
          </label>
          <label>
            이메일
            <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" required />
          </label>
          <label>
            휴대폰 번호
            <input
              value={phoneNumber}
              onChange={(event) => setPhoneNumber(event.target.value)}
              placeholder="01012345678"
              required
            />
          </label>
          <label>
            생년월일
            <input value={birthDate} onChange={(event) => setBirthDate(event.target.value)} type="date" required />
          </label>
          <label>
            성별
            <select value={gender} onChange={(event) => setGender(event.target.value as "MALE" | "FEMALE")}>
              <option value="MALE">남성</option>
              <option value="FEMALE">여성</option>
            </select>
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
          <button type="submit">회원가입</button>
        </form>
        <p className="muted">
          이미 계정이 있다면 <Link to="/login">로그인</Link>
        </p>
      </Card>
    </div>
  );
}
