import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { changePassword, deactivateMe, updateMe } from "../api/auth";
import { getLatestAnalysisResults } from "../api/analysis";
import { listChallenges, listMyChallenges } from "../api/challenges";
import { listHealthRecords } from "../api/health";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type Item = Record<string, unknown>;

type ProfileDraft = {
  nickname: string;
  phoneNumber: string;
};

type PasswordDraft = {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
};

const analysisTypeLabels: Record<string, string> = {
  DIABETES: "당뇨",
  HYPERTENSION: "고혈압",
  OBESITY: "비만",
  DYSLIPIDEMIA: "이상지질혈증",
};

const riskFallbackScores: Record<string, number> = {
  HIGH: 80,
  MEDIUM: 55,
  LOW: 25,
};

const myPageMenuItems = [
  { label: "프로필", status: "active" },
  { label: "기본 건강정보", to: "/health/profile" },
  { label: "복약/영양제", to: "/medications" },
  { label: "챌린지 현황", to: "/challenges" },
  { label: "내 가족", to: "/family" },
  { label: "알림 설정", to: "/settings" },
  { label: "개인정보", to: "/settings", badge: "설정에서 관리" },
  { label: "회원탈퇴", action: "deactivate", danger: true },
];

function getText(item: Item | undefined | null, key: string, fallback = "-"): string {
  const value = item?.[key];
  if (value === undefined || value === null || value === "") {
    return fallback;
  }
  return String(value);
}

function getDateLabel(value: unknown): string {
  if (!value) {
    return "-";
  }
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleDateString("ko-KR");
}

function getRiskLevel(result: Item): string {
  return String(result.risk_level ?? "").toUpperCase();
}

function getRiskScore(result: Item): number {
  const score = Number(result.risk_score);
  if (Number.isFinite(score) && score > 0) {
    return Math.round(score <= 1 ? score * 100 : score);
  }
  return riskFallbackScores[getRiskLevel(result)] ?? 0;
}

function normalizeStatus(value: unknown): string {
  return String(value ?? "").toUpperCase();
}

function getChallengeId(item: Item): number | null {
  const nested = item.challenge as Item | undefined;
  const raw = item.challenge_id ?? nested?.id;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function getChallengeTitle(item: Item, challengeList: Item[]): string {
  const nested = item.challenge as Item | undefined;
  const challengeId = getChallengeId(item);
  const master = challengeList.find((challenge) => Number(challenge.id) === challengeId);
  return String(item.title ?? nested?.title ?? master?.title ?? "참여 중인 챌린지");
}

function getChallengeProgress(item: Item): number {
  const explicit = Number(item.progress ?? item.progress_rate);
  if (Number.isFinite(explicit)) {
    return Math.max(0, Math.min(explicit > 1 ? explicit : explicit * 100, 100));
  }

  const completedDays = Number(item.completed_days ?? item.completed_count);
  const durationDays = Number(item.duration_days ?? 7);
  if (Number.isFinite(completedDays) && completedDays >= 0 && Number.isFinite(durationDays) && durationDays > 0) {
    return Math.max(0, Math.min(Math.round((completedDays / durationDays) * 100), 100));
  }

  const status = normalizeStatus(item.status);
  if (status === "COMPLETED") {
    return 100;
  }
  if (["ACTIVE", "IN_PROGRESS", "JOINED"].includes(status)) {
    return 40;
  }
  return 0;
}

export default function MyPage() {
  const { backendUser, logout, refreshBackendUser } = useAuth();
  const [health, setHealth] = useState<Item[]>([]);
  const [analysis, setAnalysis] = useState<Item[]>([]);
  const [challenges, setChallenges] = useState<Item[]>([]);
  const [challengeMasters, setChallengeMasters] = useState<Item[]>([]);
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [profileDraft, setProfileDraft] = useState<ProfileDraft>({
    nickname: "",
    phoneNumber: "",
  });
  const [passwordDraft, setPasswordDraft] = useState<PasswordDraft>({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  });

  useEffect(() => {
    setProfileDraft({
      nickname: backendUser?.nickname ?? "",
      phoneNumber: backendUser?.phone_number ?? "",
    });
  }, [backendUser]);

  useEffect(() => {
    const load = async () => {
      const [healthResult, analysisResult, myChallengesResult, challengeMastersResult] = await Promise.allSettled([
        listHealthRecords<Item[]>(),
        getLatestAnalysisResults<Item[]>(),
        listMyChallenges<Item[]>(),
        listChallenges<Item[]>({ limit: 100, offset: 0 }),
      ]);

      if (healthResult.status === "fulfilled") {
        setHealth(healthResult.value);
      }
      if (analysisResult.status === "fulfilled") {
        setAnalysis(analysisResult.value);
      }
      if (myChallengesResult.status === "fulfilled") {
        setChallenges(myChallengesResult.value);
      }
      if (challengeMastersResult.status === "fulfilled") {
        setChallengeMasters(challengeMastersResult.value);
      }
    };
    void load();
  }, []);

  const latestHealth = health[0];
  const displayName = backendUser?.nickname ?? backendUser?.name ?? backendUser?.login_id ?? "사용자";
  const profileInitial = displayName.slice(0, 1).toUpperCase();

  const profileRows = useMemo(
    () => [
      ["생년월일", backendUser?.birthday],
      ["성별", backendUser?.gender === "FEMALE" ? "여성" : backendUser?.gender === "MALE" ? "남성" : "-"],
      ["키/몸무게", `${getText(latestHealth, "height_cm")}cm / ${getText(latestHealth, "weight_kg")}kg`],
      ["BMI", getText(latestHealth, "bmi")],
      ["휴대폰", backendUser?.phone_number ?? "-"],
      ["이메일", backendUser?.email ?? "-"],
    ],
    [backendUser, latestHealth],
  );

  const saveProfileDraft = async () => {
    setError("");
    setNotice("");
    try {
      await updateMe({
        nickname: profileDraft.nickname.trim() || undefined,
        phone_number: profileDraft.phoneNumber.replace(/\D/g, "") || undefined,
      });
      await refreshBackendUser();
      setNotice("프로필 정보가 수정되었습니다.");
      setIsEditingProfile(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "프로필 수정에 실패했습니다.");
    }
  };

  const submitPasswordChange = async () => {
    setError("");
    setNotice("");
    if (passwordDraft.newPassword !== passwordDraft.confirmPassword) {
      setError("새 비밀번호와 확인 비밀번호가 일치하지 않습니다.");
      return;
    }
    try {
      await changePassword({
        current_password: passwordDraft.currentPassword,
        new_password: passwordDraft.newPassword,
      });
      setPasswordDraft({ currentPassword: "", newPassword: "", confirmPassword: "" });
      setShowPasswordForm(false);
      setNotice("비밀번호가 변경되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "비밀번호 변경에 실패했습니다.");
    }
  };

  const deactivateAccount = async () => {
    setError("");
    setNotice("");
    if (!window.confirm("회원탈퇴를 진행하면 계정이 비활성화됩니다. 계속하시겠습니까?")) {
      return;
    }
    if (!window.confirm("탈퇴 후에는 현재 계정으로 서비스 이용이 제한됩니다. 정말 탈퇴하시겠습니까?")) {
      return;
    }
    try {
      await deactivateMe();
      await logout();
    } catch (err) {
      setError(err instanceof Error ? err.message : "회원탈퇴 처리에 실패했습니다.");
    }
  };

  return (
    <div className="dashboard-grid">
      <Card title="마이페이지">
        <div className="mypage-menu">
          {myPageMenuItems.map((item) => {
            if (item.to) {
              return (
                <Link className="mypage-menu-item" key={item.label} to={item.to}>
                  <span>{item.label}</span>
                  {item.badge && <span className="badge badge-reference">{item.badge}</span>}
                </Link>
              );
            }

            return (
              <button
                className={[
                  "mypage-menu-item",
                  item.status === "active" ? "active" : "",
                  item.danger ? "danger-ghost" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                key={item.label}
                type="button"
                onClick={() => {
                  if (item.action === "deactivate") {
                    void deactivateAccount();
                  }
                }}
              >
                <span>{item.label}</span>
                {item.badge && <span className="badge badge-reference">{item.badge}</span>}
              </button>
            );
          })}
        </div>
      </Card>

      <div className="page-stack">
        {notice && <div className="state-box">{notice}</div>}
        {error && <ErrorMessage message={error} />}

        <Card title="프로필/기본 내역">
          <div className="profile-card-row">
            <span className="avatar avatar-large">{profileInitial}</span>
            <div className="profile-card-main">
              {isEditingProfile ? (
                <div className="form two-col">
                  <label>
                    닉네임
                    <input
                      value={profileDraft.nickname}
                      onChange={(event) => setProfileDraft((prev) => ({ ...prev, nickname: event.target.value }))}
                    />
                  </label>
                  <label>
                    휴대폰
                    <input
                      value={profileDraft.phoneNumber}
                      onChange={(event) => setProfileDraft((prev) => ({ ...prev, phoneNumber: event.target.value }))}
                    />
                  </label>
                  <label>
                    이메일
                    <input readOnly value={backendUser?.email ?? ""} />
                  </label>
                </div>
              ) : (
                <>
                  <strong>{displayName}</strong>
                  <p className="muted">{backendUser?.email ?? "이메일 정보 없음"}</p>
                  <div className="chip-list">
                    <span className="badge badge-saved">{backendUser?.role ?? "USER"}</span>
                    <span className="badge badge-reference">{backendUser?.login_id ?? "로그인 ID 미등록"}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="button-row" style={{ marginTop: 16 }}>
            {isEditingProfile ? (
              <>
                <button onClick={() => void saveProfileDraft()} type="button">
                  저장
                </button>
                <button
                  className="secondary"
                  onClick={() => {
                    setIsEditingProfile(false);
                    setProfileDraft({
                      nickname: backendUser?.nickname ?? "",
                      phoneNumber: backendUser?.phone_number ?? "",
                    });
                  }}
                  type="button"
                >
                  취소
                </button>
              </>
            ) : (
              <button className="secondary" onClick={() => setIsEditingProfile(true)} type="button">
                수정
              </button>
            )}
            <button className="secondary" onClick={() => setShowPasswordForm((prev) => !prev)} type="button">
              비밀번호 변경
            </button>
            <button className="danger-ghost" onClick={() => void deactivateAccount()} type="button">
              회원탈퇴
            </button>
            <Link className="button secondary" to="/settings">
              설정으로 이동
            </Link>
          </div>
        </Card>

        {showPasswordForm && (
          <Card title="비밀번호 변경">
            <div className="form three-col">
              <label>
                현재 비밀번호
                <input
                  type="password"
                  value={passwordDraft.currentPassword}
                  onChange={(event) => setPasswordDraft((prev) => ({ ...prev, currentPassword: event.target.value }))}
                />
              </label>
              <label>
                새 비밀번호
                <input
                  type="password"
                  value={passwordDraft.newPassword}
                  onChange={(event) => setPasswordDraft((prev) => ({ ...prev, newPassword: event.target.value }))}
                />
              </label>
              <label>
                새 비밀번호 확인
                <input
                  type="password"
                  value={passwordDraft.confirmPassword}
                  onChange={(event) => setPasswordDraft((prev) => ({ ...prev, confirmPassword: event.target.value }))}
                />
              </label>
            </div>
            <div className="button-row" style={{ marginTop: 16 }}>
              <button onClick={() => void submitPasswordChange()} type="button">
                변경하기
              </button>
              <button className="secondary" onClick={() => setShowPasswordForm(false)} type="button">
                취소
              </button>
            </div>
          </Card>
        )}

        <Card title="기본 건강정보">
          <div className="table-list">
            {profileRows.map(([label, value]) => (
              <div className="table-row" key={String(label)}>
                <span>{String(label)}</span>
                <strong>{String(value ?? "-")}</strong>
                <span className={value && value !== "-" ? "badge badge-saved" : "badge badge-missing"}>
                  {value && value !== "-" ? "저장됨" : "미입력"}
                </span>
              </div>
            ))}
          </div>
          <div className="button-row" style={{ marginTop: 16 }}>
            <Link className="button secondary" to="/health/profile">
              건강정보 수정
            </Link>
            <Link className="button secondary" to="/health">
              건강 분석 입력
            </Link>
          </div>
        </Card>

        <div className="page-grid">
          <Card title="현재 상태">
            <div className="metric-grid mypage-metric-grid">
              <div>
                <span>BMI</span>
                <strong>{getText(latestHealth, "bmi", "기록 없음")}</strong>
              </div>
              <div>
                <span>혈압</span>
                <strong>
                  {getText(latestHealth, "systolic_bp", "-")}/{getText(latestHealth, "diastolic_bp", "-")}
                </strong>
              </div>
              <div>
                <span>공복혈당</span>
                <strong>{getText(latestHealth, "fasting_glucose", "기록 없음")}</strong>
              </div>
              <div>
                <span>수면</span>
                <strong>{getText(latestHealth, "sleep_hours", "기록 없음")}</strong>
              </div>
            </div>
          </Card>

          <Card title="건강 목표">
            <div className="card-list">
              <div className="mini-card">
                <strong>기본 건강정보 꾸준히 관리</strong>
                <p className="muted">신장, 체중, 생활습관을 최신 상태로 유지하면 분석 결과를 더 안정적으로 확인할 수 있습니다.</p>
              </div>
              <div className="mini-card">
                <strong>생활습관 챌린지 실천</strong>
                <p className="muted">걷기, 식단, 복약 기록을 함께 관리해 추적 대시보드에서 변화를 확인해보세요.</p>
              </div>
            </div>
          </Card>
        </div>

        <div className="page-grid">
          <Card title="최근 분석 결과">
            <div className="card-list">
              {analysis.length === 0 && <div className="state-box">최근 분석 결과가 없습니다.</div>}
              {analysis.map((result) => {
                const level = getRiskLevel(result);
                const score = getRiskScore(result);
                const resultId = Number(result.id);
                return (
                  <div className="mini-card result-summary-card" key={String(result.id ?? result.analysis_type)}>
                    <div>
                      <span className="muted">분석 유형</span>
                      <strong>{analysisTypeLabels[String(result.analysis_type)] ?? String(result.analysis_type ?? "-")}</strong>
                    </div>
                    <div className="button-row">
                      <span className={`badge risk-${level.toLowerCase()}`}>{level || "-"}</span>
                      <span className="badge badge-reference">{score}/100</span>
                      <span className="badge badge-reference">{getDateLabel(result.created_at)}</span>
                    </div>
                    {Number.isFinite(resultId) && (
                      <Link className="button secondary" to={`/analysis/${resultId}`}>
                        상세보기
                      </Link>
                    )}
                  </div>
                );
              })}
            </div>
          </Card>

          <Card title="진행 중 챌린지">
            <div className="card-list">
              {challenges.length === 0 && <div className="state-box">진행 중인 챌린지가 없습니다.</div>}
              {challenges.map((challenge) => {
                const progress = getChallengeProgress(challenge);
                return (
                  <div className="mini-card" key={String(challenge.id)}>
                    <div>
                      <strong>{getChallengeTitle(challenge, challengeMasters)}</strong>
                      <p className="muted">{normalizeStatus(challenge.status) || "JOINED"}</p>
                    </div>
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: `${progress}%` }} />
                    </div>
                    <div className="button-row">
                      <span className="badge badge-reference">진행률 {progress}%</span>
                      {getChallengeId(challenge) && (
                        <Link className="button secondary" to={`/challenges/${String(getChallengeId(challenge))}`}>
                          상세보기
                        </Link>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>

        <Card title="복용중인 약물/영양제 목록">
          <div className="empty-state">
            <strong>복약/영양제 정보는 복약 관리 화면에서 확인할 수 있습니다.</strong>
            <p>등록된 약물, 영양제, 복약 기록을 한 곳에서 관리해보세요.</p>
            <Link className="button secondary" to="/medications">
              복약 관리로 이동
            </Link>
          </div>
        </Card>

        <Card title="가족 요약">
          <div className="family-summary-card">
            <div>
              <span className="badge badge-reference">내 가족</span>
              <h3>가족을 연결하면 건강정보 공유와 보호자 알림을 설정할 수 있습니다.</h3>
              <p className="muted">현재 연결된 가족 0명, 공유 알림 0건입니다.</p>
            </div>
            <Link className="button" to="/family">
              가족 관리로 이동
            </Link>
          </div>
        </Card>
      </div>
    </div>
  );
}
