import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { changePassword, updateMe } from "../api/auth";
import { getLatestAnalysisResults } from "../api/analysis";
import { listChallenges, listMyChallenges } from "../api/challenges";
import { listHealthRecords } from "../api/health";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import RiskStageBoard, { type DiseaseRiskItem } from "../components/RiskStageBoard";
import {
  getAnalysisSourceBadgeLabel,
  getAnalysisTypeLabel,
  getDisplayRiskLabel,
  getLatestResultsByAnalysisType,
  getRiskClassName,
  isKnownAnalysisType,
} from "../utils/riskDisplay";

import { Mail, Phone } from 'lucide-react';
import { Activity, Gauge, Droplet, Moon } from "lucide-react";

type Item = Record<string, unknown>;

type MyPageMenuItem = {
  label: string;
  to?: string;
  status?: "active";
  badge?: string;
  danger?: boolean;
  action?: "deactivate";
};

type ProfileDraft = {
  nickname: string;
  phoneNumber: string;
};

type PasswordDraft = {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
};

const challengeStatusLabels: Record<string, string> = {
  ACTIVE: "진행 중",
  IN_PROGRESS: "진행 중",
  JOINED: "진행 중",
  COMPLETED: "완료",
  CANCELED: "참여 전",
  CANCELLED: "참여 전",
  FAILED: "종료",
  GIVE_UP: "참여 전",
  GIVEN_UP: "참여 전",
};

const myPageMenuItems: MyPageMenuItem[] = [
  { label: "프로필", status: "active" },
  { label: "기본 건강정보", to: "/health/profile" },
  { label: "복약/영양제", to: "/medications" },
  { label: "챌린지 현황", to: "/challenges" },
  { label: "내 가족", to: "/family" },
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

function normalizeStatus(value: unknown): string {
  return String(value ?? "").toUpperCase();
}

function hasMetCompletionCondition(item: Item): boolean {
  if (item.has_met_completion_condition !== undefined) {
    return Boolean(item.has_met_completion_condition);
  }
  const completedDays = Number(item.completed_days ?? item.completed_count ?? 0);
  const totalDays = Number(item.total_days ?? item.duration_days ?? 7);
  return Number.isFinite(completedDays) && Number.isFinite(totalDays) && completedDays >= Math.ceil(totalDays * 0.8);
}

function isFinalizedChallenge(item: Item): boolean {
  if (Boolean(item.is_finalized) || Boolean(item.completed_at)) {
    return true;
  }
  return ["COMPLETED", "EXPIRED", "FAILED"].includes(normalizeStatus(item.status));
}

function getChallengeStatusLabel(item: Item): string {
  if (["GIVE_UP", "GIVEN_UP", "CANCELED", "CANCELLED"].includes(normalizeStatus(item.status))) {
    return "참여 전";
  }
  if (isFinalizedChallenge(item)) {
    return hasMetCompletionCondition(item) || Boolean(item.completed_at) ? "완료" : "미달성";
  }
  if (hasMetCompletionCondition(item)) {
    return "완료 조건 충족";
  }
  return challengeStatusLabels[normalizeStatus(item.status)] ?? "진행 중";
}

function isVisibleMyChallenge(item: Item): boolean {
  return (
    !["GIVE_UP", "GIVEN_UP", "CANCELED", "CANCELLED"].includes(normalizeStatus(item.status)) &&
    !isFinalizedChallenge(item)
  );
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

  if (Boolean(item.completed_at) || (isFinalizedChallenge(item) && hasMetCompletionCondition(item))) {
    return 100;
  }

  const rate = Number(item.completion_rate);
  if (Number.isFinite(rate)) {
    return Math.max(0, Math.min(rate, 100));
  }

  const completedDays = Number(item.completed_days ?? item.completed_count);
  const durationDays = Number(item.total_days ?? item.duration_days ?? 7);
  if (Number.isFinite(completedDays) && completedDays >= 0 && Number.isFinite(durationDays) && durationDays > 0) {
    return Math.max(0, Math.min(Math.round((completedDays / durationDays) * 100), 100));
  }

  if (["ACTIVE", "IN_PROGRESS", "JOINED"].includes(normalizeStatus(item.status))) {
    return 40;
  }
  return 0;
}

function validateNickname(value: string): string | null {
  const trimmed = value.trim();
  if (trimmed.length < 2 || trimmed.length > 20) {
    return "닉네임은 2자 이상 20자 이하로 입력해주세요.";
  }
  return null;
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
  const visibleChallenges = challenges.filter(isVisibleMyChallenge);
  const latestDiseaseAnalysis = getLatestResultsByAnalysisType(
    analysis.filter((result) => isKnownAnalysisType(result.analysis_type)),
  );
  const diseaseRiskItems: DiseaseRiskItem[] = latestDiseaseAnalysis
    .map((result) => ({
      analyzed_at: result.analyzed_at,
      created_at: result.created_at,
      diseaseName: getAnalysisTypeLabel(result.analysis_type),
      id: result.id,
      risk_level: result.risk_level,
      service_band: result.service_band,
      service_band_label: result.service_band_label,
    }));
  const displayName = backendUser?.nickname ?? backendUser?.name ?? backendUser?.login_id ?? "사용자";
  const profileInitial = displayName.slice(0, 1).toUpperCase();

  const profileRows = useMemo(
    () => [
      ["생년월일", backendUser?.birthday],
      ["성별", backendUser?.gender === "FEMALE" ? "여성" : backendUser?.gender === "MALE" ? "남성" : "-"],
      ["키/몸무게", `${getText(latestHealth, "height_cm")}cm / ${getText(latestHealth, "weight_kg")}kg`],
      ["BMI", getText(latestHealth, "bmi")],
    ],
    [backendUser, latestHealth],
  );

  const saveProfileDraft = async () => {
    setError("");
    setNotice("");
    const nextNickname = profileDraft.nickname.trim();
    const nicknameError = validateNickname(nextNickname);
    if (nicknameError) {
      setError(nicknameError);
      return;
    }

    try {
      const currentPhoneNumber = backendUser?.phone_number ?? "";
      const nextPhoneNumber = profileDraft.phoneNumber.replace(/\D/g, "");
      const payload = {
        nickname: nextNickname,
        ...(nextPhoneNumber && nextPhoneNumber !== currentPhoneNumber ? { phone_number: nextPhoneNumber } : {}),
      };
      const updatedUser = await updateMe(payload);
      await refreshBackendUser();
      setProfileDraft({
        nickname: updatedUser.nickname ?? nextNickname,
        phoneNumber: updatedUser.phone_number ?? nextPhoneNumber,
      });
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

  const deactivateAccount = () => {
    window.alert("회원 탈퇴 기능은 준비 중입니다.");
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
                </Link>
              );
            }

            return (
              <button
                className="mypage-menu-item"
                key={item.label}
                type="button"
              >
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </Card>

      <div className="page-stack">
        {notice && <div className="state-box">{notice}</div>}
        {error && <ErrorMessage message={error} />}

        <Card title="프로필/기본 내역" actions={
          !isEditingProfile ? (
            <button className="secondary" onClick={() => setIsEditingProfile(true)} type="button">
              수정
            </button>
          ) : undefined
        }>
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
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <strong style={{ fontSize: "40px" }}>{displayName}</strong>
                    <span className="badge badge-reference">{backendUser?.login_id ?? "로그인 ID 미등록"}</span>
                  </div>
                  <p className="muted" style={{ marginBottom: "4px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <Mail size={14} /> {backendUser?.email ?? "이메일 정보 없음"}
                  </p>
                  <p className="muted" style={{ margin: 0, display: "flex", alignItems: "center", gap: "6px" }}>
                    <Phone size={14} /> {backendUser?.phone_number ?? "미등록"}
                  </p>
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
            ) : null}

          </div>
        </Card>

        {(showPasswordForm || isEditingProfile) && (
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

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
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

        <Card title="현재 상태">
            <div className="metric-grid mypage-metric-grid">
              <div>
                <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <Gauge size={14} /> BMI
                </span>
                <strong>{getText(latestHealth, "bmi", "기록 없음")}</strong>
              </div>
              <div>
                <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <Activity size={14} /> 혈압
                </span>
                <strong>
                  {getText(latestHealth, "systolic_bp", "-")}/{getText(latestHealth, "diastolic_bp", "-")}
                </strong>
              </div>
              <div>
                <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <Droplet size={14} /> 공복혈당
                </span>
                <strong>{getText(latestHealth, "fasting_glucose", "기록 없음")}</strong>
              </div>
              <div>
                <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <Moon size={14} /> 수면
                </span>
                <strong>{getText(latestHealth, "sleep_hours", "기록 없음")}</strong>
              </div>
            </div>
          </Card>
          </div>

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

        <div className="page-grid">
          <Card title="최근 분석 결과">
            <div className="card-list">
              {analysis.length === 0 && <div className="state-box">최근 분석 결과가 없습니다.</div>}
              {diseaseRiskItems.length > 0 && <RiskStageBoard items={diseaseRiskItems} />}
              {latestDiseaseAnalysis.map((result) => {
                const resultId = Number(result.id);
                const sourceBadgeLabel = getAnalysisSourceBadgeLabel(result);
                return (
                  <div className="mini-card result-summary-card" key={String(result.id ?? result.analysis_type)}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <strong>{getAnalysisTypeLabel(result.analysis_type, "-")}</strong>
                      {Number.isFinite(resultId) && (
                        <Link className="muted" style={{ fontSize: "13px" }} to={`/analysis/${resultId}`}>
                          상세보기 →
                        </Link>
                      )}
                    </div>
                    <div className="button-row" style={{ marginTop: "6px" }}>
                      <span className={`badge ${getRiskClassName(result)}`}>{getDisplayRiskLabel(result)}</span>
                      {sourceBadgeLabel && <span className="badge badge-reference">{sourceBadgeLabel}</span>}
                      <span className="badge badge-reference">{getDateLabel(result.created_at)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          <Card title="진행 중 챌린지">
            <div className="card-list">
              {visibleChallenges.length === 0 && <div className="state-box">진행 중인 챌린지가 없습니다.</div>}
              {visibleChallenges.map((challenge) => {
                const progress = getChallengeProgress(challenge);
                return (
                  <div className="mini-card" key={String(challenge.id)}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <strong>{getChallengeTitle(challenge, challengeMasters)}</strong>
                      {getChallengeId(challenge) && (
                        <Link className="muted" style={{ fontSize: "13px" }} to={`/challenges/${String(getChallengeId(challenge))}`}>
                          상세보기 →
                        </Link>
                      )}
                    </div>
                    <div style={{ marginTop: "6px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                        <span className="muted" style={{ fontSize: "13px" }}>진행률</span>
                        <span className="muted" style={{ fontSize: "13px" }}>{progress}%</span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
