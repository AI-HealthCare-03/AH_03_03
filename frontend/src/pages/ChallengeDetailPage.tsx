import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import {
  completeToday,
  getChallenge,
  giveUpChallenge,
  joinChallenge,
  listChallengeLogs,
  listMyChallenges,
} from "../api/challenges";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type Challenge = Record<string, unknown>;
type ChallengeLog = Record<string, unknown>;

const categoryIcon: Record<string, string> = {
  DIET: "🥗",
  EXERCISE: "🚶",
  SLEEP: "🌙",
  MEDICATION: "💊",
  WATER: "💧",
  BLOOD_SUGAR: "🩸",
  BLOOD_GLUCOSE: "🩸",
  BLOOD_PRESSURE: "🩺",
  HABIT: "✅",
  COMMON: "🌿",
  WEIGHT: "⚖️",
};

const categoryLabel: Record<string, string> = {
  BLOOD_PRESSURE: "혈압 관리",
  BLOOD_SUGAR: "혈당 관리",
  BLOOD_GLUCOSE: "혈당 관리",
  DIET: "식단",
  EXERCISE: "운동",
  MEDICATION: "복약",
  HABIT: "생활습관",
  WATER: "수분섭취",
  SLEEP: "수면",
  COMMON: "공통",
  WEIGHT: "체중 관리",
};

const targetDiseaseLabel: Record<string, string> = {
  COMMON: "공통 건강관리",
  GENERAL: "일반 건강관리",
  DIABETES: "당뇨 관리",
  HYPERTENSION: "고혈압 관리",
  DYSLIPIDEMIA: "콜레스테롤·중성지방 관리",
  OBESITY: "비만 관리",
};

const challengeTypeLabel: Record<string, string> = {
  SPECIAL: "특수 챌린지",
  COMMON: "공통 챌린지",
  GENERAL: "일반 챌린지",
};

const difficultyLabel: Record<string, string> = {
  EASY: "쉬움",
  MEDIUM: "보통",
  NORMAL: "보통",
  HARD: "어려움",
};

const statusLabel: Record<string, string> = {
  ACTIVE: "진행 중",
  IN_PROGRESS: "진행 중",
  JOINED: "진행 중",
  COMPLETED: "완료",
  GIVEN_UP: "포기",
  GIVE_UP: "포기",
  CANCELED: "포기",
  CANCELLED: "포기",
  PENDING: "대기",
};

const metricLabel: Record<string, string> = {
  exercise_minutes: "운동 시간",
  post_meal_walk_minutes: "식후 산책 시간",
  water_replacement_count: "물 마시기 실천",
  low_sugar_meal_count: "저당 식단 실천",
  late_night_snack_count: "야식 횟수",
  sleep_hours: "수면 시간",
  medication_record_count: "복약 기록",
  morning_health_record_count: "아침 건강 지표 기록",
};

function normalizeStatus(value: unknown): string {
  return String(value ?? "").toUpperCase();
}

function getDisplayStatus(value: unknown): string {
  const status = normalizeStatus(value);
  return statusLabel[status] ?? "진행 중";
}

function getCategory(challenge: Challenge | null): string {
  return String(challenge?.category ?? "COMMON").toUpperCase();
}

function getDisplayCategory(challenge: Challenge | null): string {
  return categoryLabel[getCategory(challenge)] ?? "공통";
}

function getDisplayChallengeType(challenge: Challenge | null): string {
  const raw = String(challenge?.challenge_type ?? "GENERAL").toUpperCase();
  return challengeTypeLabel[raw] ?? "일반 챌린지";
}

function getDifficulty(challenge: Challenge | null): string {
  const raw = String(challenge?.difficulty ?? "NORMAL").toUpperCase();
  return difficultyLabel[raw] ?? "보통";
}

function getTargetDiseaseFromDescription(description: unknown): string {
  const text = String(description ?? "");
  const matched = text.match(/\[target_disease=([A-Z_]+)\]/);
  if (matched?.[1]) {
    return matched[1];
  }
  if (/혈당|당뇨|저당/.test(text)) {
    return "DIABETES";
  }
  if (/혈압/.test(text)) {
    return "HYPERTENSION";
  }
  if (/비만|체중|야식/.test(text)) {
    return "OBESITY";
  }
  if (/콜레스테롤|지질/.test(text)) {
    return "DYSLIPIDEMIA";
  }
  return "COMMON";
}

function getDisplayTargetDisease(challenge: Challenge | null): string {
  const raw = String(challenge?.target_disease ?? getTargetDiseaseFromDescription(challenge?.description));
  return targetDiseaseLabel[raw.toUpperCase()] ?? "공통 건강관리";
}

function getCleanDescription(value: unknown): string {
  return String(value ?? "").replace(/\[target_disease=[A-Z_]+\]\s*/g, "");
}

function getDisplayMetric(value: unknown): string {
  const raw = String(value ?? "");
  return metricLabel[raw] ?? "생활습관";
}

function getDurationDays(challenge: Challenge | null, userChallenge?: Challenge | null): number {
  const parsed = Number(userChallenge?.duration_days ?? challenge?.duration_days);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 7;
}

function getProgress(userChallenge: Challenge | null, challenge: Challenge | null, logs: ChallengeLog[]): number {
  if (!userChallenge) {
    return 0;
  }
  const explicit = Number(userChallenge.progress ?? userChallenge.progress_rate);
  if (Number.isFinite(explicit)) {
    return Math.max(0, Math.min(explicit > 1 ? explicit : explicit * 100, 100));
  }
  if (logs.length > 0) {
    const completedCount = logs.filter((log) => Boolean(log.is_completed)).length;
    return Math.max(0, Math.min(Math.round((completedCount / getDurationDays(challenge, userChallenge)) * 100), 100));
  }
  const completedDays = Number(userChallenge.completed_days ?? userChallenge.completed_count);
  if (Number.isFinite(completedDays) && completedDays >= 0) {
    return Math.max(0, Math.min(Math.round((completedDays / getDurationDays(challenge, userChallenge)) * 100), 100));
  }
  const status = normalizeStatus(userChallenge.status);
  if (status === "COMPLETED") {
    return 100;
  }
  if (["ACTIVE", "IN_PROGRESS", "JOINED"].includes(status)) {
    return 40;
  }
  if (["GIVE_UP", "GIVEN_UP", "FAILED", "CANCELED", "CANCELLED"].includes(status)) {
    return 0;
  }
  return 0;
}

function getDailyGoalCount(challenge: Challenge | null, userChallenge?: Challenge | null): number {
  const direct = Number(userChallenge?.daily_goal_count ?? userChallenge?.daily_goal);
  if (Number.isFinite(direct) && direct > 0) {
    return Math.max(1, Math.min(Math.round(direct), 10));
  }
  const metric = String(challenge?.target_metric ?? "").toLowerCase();
  if (!["count", "times", "횟수", "회"].some((token) => metric.includes(token))) {
    return 1;
  }
  const matched = String(challenge?.target_value ?? "").match(/\d+/);
  return matched ? Math.max(1, Math.min(Number(matched[0]), 10)) : 1;
}

function getTodayKey(): string {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

function getCompletedCount(logs: ChallengeLog[]): number {
  return logs.filter((log) => Boolean(log.is_completed)).length;
}

function getTodayCompletedCount(logs: ChallengeLog[]): number {
  const today = getTodayKey();
  return logs.filter((log) => String(log.log_date ?? "").slice(0, 10) === today && Boolean(log.is_completed)).length;
}

function isTodayCompleted(logs: ChallengeLog[], dailyGoalCount: number): boolean {
  return getTodayCompletedCount(logs) >= dailyGoalCount;
}

function isActiveUserChallenge(userChallenge: Challenge | null): boolean {
  if (!userChallenge) {
    return false;
  }
  return ["ACTIVE", "IN_PROGRESS", "JOINED"].includes(normalizeStatus(userChallenge.status));
}

function isEndedUserChallenge(userChallenge: Challenge | null): boolean {
  if (!userChallenge) {
    return false;
  }
  return ["COMPLETED", "GIVE_UP", "GIVEN_UP", "FAILED", "CANCELED", "CANCELLED"].includes(normalizeStatus(userChallenge.status));
}

function getStatusBadgeLabel(userChallenge: Challenge | null, logs: ChallengeLog[]): string {
  if (!userChallenge) {
    return "참여 전";
  }
  const dailyGoalCount = Number(userChallenge.daily_goal_count ?? 1);
  if (isTodayCompleted(logs, dailyGoalCount) && isActiveUserChallenge(userChallenge)) {
    return "오늘 완료";
  }
  return getDisplayStatus(userChallenge.status);
}

function buildHowToItems(challenge: Challenge | null): string[] {
  const category = getCategory(challenge);
  const duration = getDurationDays(challenge);
  const dailyGoalCount = getDailyGoalCount(challenge);
  const metric = getDisplayMetric(challenge?.target_metric);
  const targetValue = String(challenge?.target_value ?? "").trim();
  const base = [
    `${duration}일 동안 하루 ${dailyGoalCount}회 목표를 실천한 뒤 완료 버튼을 눌러 기록하세요.`,
    targetValue ? `하루 수행 기준은 ${metric} ${targetValue}입니다.` : `하루 수행 기준은 ${metric} 실천 여부입니다.`,
  ];

  if (["EXERCISE", "WATER", "WEIGHT"].includes(category)) {
    return [...base, "무리하지 않는 강도로 시작하고, 컨디션이 좋지 않으면 강도를 낮춰주세요."];
  }
  if (category === "DIET") {
    return [...base, "식사 후 실천 내용을 확인하고, 가능한 경우 식단 기록도 함께 남겨보세요."];
  }
  if (category === "MEDICATION") {
    return [...base, "복약 또는 영양제 섭취 후 완료 버튼을 눌러 기록하세요."];
  }
  if (["BLOOD_PRESSURE", "BLOOD_GLUCOSE"].includes(category)) {
    return [...base, "측정값이 있다면 건강정보나 검진 기록에 함께 남기면 추적에 도움이 됩니다."];
  }
  return [...base, "생활 패턴에 맞는 시간대를 정해 반복하면 기록을 이어가기 쉽습니다."];
}

function buildExpectedEffects(challenge: Challenge | null): string[] {
  const category = getCategory(challenge);
  const disease = getDisplayTargetDisease(challenge);
  const effects = ["작은 실천을 반복해 생활습관을 점검하는 데 도움이 될 수 있습니다."];

  if (category === "DIET") {
    effects.push("식사 패턴을 기록하면 건강관리 관점에서 식습관을 돌아보기 쉽습니다.");
  } else if (category === "EXERCISE") {
    effects.push("규칙적인 활동량을 쌓아 체중과 혈압 관리 습관을 만드는 데 도움이 될 수 있습니다.");
  } else if (category === "MEDICATION") {
    effects.push("복약 기록을 남기면 빠뜨린 날을 확인하고 관리 흐름을 유지하기 쉽습니다.");
  } else if (["BLOOD_PRESSURE", "BLOOD_GLUCOSE"].includes(category)) {
    effects.push("반복 기록을 통해 수치 변화 흐름을 확인하는 데 도움이 될 수 있습니다.");
  } else {
    effects.push(`${disease} 관점에서 무리 없는 건강관리 습관을 확인하는 데 참고할 수 있습니다.`);
  }
  effects.push("개인 상태에 따라 부담이 될 수 있으니 불편감이 있으면 중단하고 필요한 경우 의료진과 상담하세요.");
  return effects;
}

export default function ChallengeDetailPage() {
  const { challengeId } = useParams();
  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [userChallenge, setUserChallenge] = useState<Challenge | null>(null);
  const [logs, setLogs] = useState<ChallengeLog[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<"join" | "complete" | "give-up" | null>(null);

  const load = async () => {
    if (!challengeId) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const challengeItem = await getChallenge<Challenge>(Number(challengeId));
      setChallenge(challengeItem);
    } catch {
      setUserChallenge(null);
      setLogs([]);
      setError("챌린지 상세를 불러오지 못했습니다.");
      setLoading(false);
      return;
    }
    try {
      const myItems = await listMyChallenges<Challenge[]>({ limit: 100, offset: 0 });
      const matched = myItems.find((item) => Number(item.challenge_id) === Number(challengeId)) ?? null;
      setUserChallenge(matched);
      if (matched?.id) {
        setLogs(await listChallengeLogs<ChallengeLog[]>(Number(matched.id)));
      } else {
        setLogs([]);
      }
    } catch {
      setUserChallenge(null);
      setLogs([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, [challengeId]);

  const join = async () => {
    if (!challengeId) {
      return;
    }
    setError("");
    setMessage("");
    setActionLoading("join");
    try {
      await joinChallenge(Number(challengeId));
      setMessage("챌린지에 참여했습니다. 오늘 수행 기록을 남길 수 있습니다.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "챌린지 참여 처리에 실패했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  const complete = async () => {
    if (!userChallenge?.id) {
      setMessage("먼저 챌린지에 참여해주세요.");
      return;
    }
    if (isTodayCompleted(logs, getDailyGoalCount(challenge, userChallenge))) {
      setMessage("오늘은 이미 완료한 챌린지입니다.");
      return;
    }
    setError("");
    setMessage("");
    setActionLoading("complete");
    try {
      await completeToday(Number(userChallenge.id));
      setMessage("오늘 수행을 완료했습니다.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "오늘 수행 완료 처리에 실패했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  const giveUp = async () => {
    if (!userChallenge?.id) {
      setMessage("참여 중인 챌린지가 없습니다.");
      return;
    }
    const confirmed = window.confirm("챌린지를 포기하면 현재 진행 상태가 중단됩니다. 계속하시겠습니까?");
    if (!confirmed) {
      return;
    }
    setError("");
    setMessage("");
    setActionLoading("give-up");
    try {
      await giveUpChallenge(Number(userChallenge.id));
      setMessage("챌린지를 포기 처리했습니다.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "챌린지 포기 처리에 실패했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  const category = getCategory(challenge);
  const progress = getProgress(userChallenge, challenge, logs);
  const durationDays = getDurationDays(challenge, userChallenge);
  const dailyGoalCount = getDailyGoalCount(challenge, userChallenge);
  const completedCount = getCompletedCount(logs);
  const todayCompletedCount = getTodayCompletedCount(logs);
  const active = isActiveUserChallenge(userChallenge);
  const todayCompleted = isTodayCompleted(logs, dailyGoalCount);
  const ended = isEndedUserChallenge(userChallenge);
  const nextActionLabel =
    dailyGoalCount > 1 ? `${Math.min(todayCompletedCount + 1, dailyGoalCount)}/${dailyGoalCount} 수행하기` : "오늘 수행 완료";

  return (
    <div className="page-stack">
      <Card
        title="챌린지 상세"
        actions={
          <Link className="button secondary" to="/challenges">
            목록
          </Link>
        }
      >
        {error && <ErrorMessage message={error} />}
        {message && <p className="success-text">{message}</p>}
        {loading && <div className="state-box">챌린지 상세를 불러오는 중입니다.</div>}
        <div className="detail-hero">
          <span className="eyebrow">{getDisplayCategory(challenge)}</span>
          <h1>{String(challenge?.title ?? "챌린지")}</h1>
          <div className="challenge-icon-large" aria-label={`${category} challenge icon`}>
            {categoryIcon[category] ?? "🌿"}
          </div>
          <p>{getCleanDescription(challenge?.description) || "건강 습관을 작게 시작해보세요."}</p>
          <div className="metric-grid">
            <div>
              <span>기간</span>
              <strong>{getDurationDays(challenge, userChallenge)}일</strong>
            </div>
            <div>
              <span>하루 목표</span>
              <strong>{dailyGoalCount}회</strong>
            </div>
            <div>
              <span>목표 지표</span>
              <strong>{getDisplayMetric(challenge?.target_metric)}</strong>
            </div>
            <div>
              <span>분류</span>
              <strong>{getDisplayChallengeType(challenge)}</strong>
            </div>
            <div>
              <span>난이도</span>
              <strong>{getDifficulty(challenge)}</strong>
            </div>
            <div>
              <span>대상</span>
              <strong>{getDisplayTargetDisease(challenge)}</strong>
            </div>
            <div>
              <span>상태</span>
              <strong>{getStatusBadgeLabel(userChallenge, logs)}</strong>
            </div>
          </div>
          {Boolean(challenge?.caution_message) && <div className="state-box warning-card">{String(challenge?.caution_message)}</div>}
          {Boolean(challenge?.contraindication_message) && (
            <div className="state-box warning-card">{String(challenge?.contraindication_message)}</div>
          )}
          <div>
            <span className="muted">진행률</span>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <p className="muted">
              {userChallenge ? `${Math.min(completedCount, durationDays)}일 / ${durationDays}일 완료 (${progress}%)` : "아직 참여 전입니다."}
            </p>
          </div>
          <div className="challenge-log-grid">
            {logs.length === 0 && (
              <div className="state-box">
                {userChallenge ? "아직 수행 기록이 없습니다. 오늘 완료를 눌러 첫 기록을 남겨보세요." : "참여하면 오늘 수행 기록을 남길 수 있습니다."}
              </div>
            )}
            {logs.map((log) => (
              <span className={log.is_completed ? "badge risk-low" : "badge badge-missing"} key={String(log.id)}>
                {String(log.log_date ?? "기록")} {log.is_completed ? "완료" : "미완료"}
              </span>
            ))}
          </div>
          <div className="challenge-detail-actions">
            {!userChallenge && (
              <button disabled={actionLoading !== null} onClick={join} type="button">
                {actionLoading === "join" ? "참여 처리 중..." : "참여하기"}
              </button>
            )}
            {active && (
              <>
                <button className="secondary" disabled={todayCompleted || actionLoading !== null} onClick={complete} type="button">
                  {actionLoading === "complete" ? "저장 중..." : todayCompleted ? "오늘 완료됨" : nextActionLabel}
                </button>
                <button className="danger" disabled={actionLoading !== null} onClick={giveUp} type="button">
                  {actionLoading === "give-up" ? "처리 중..." : "포기하기"}
                </button>
              </>
            )}
            {ended && <span className="badge badge-reference">{getDisplayStatus(userChallenge?.status)}</span>}
          </div>
        </div>
      </Card>
      <Card title="챌린지 하는 방법">
        <div className="timeline-list">
          {buildHowToItems(challenge).map((item) => (
            <div key={item}>{item}</div>
          ))}
        </div>
      </Card>
      <Card title="기대 효과">
        <div className="timeline-list">
          {buildExpectedEffects(challenge).map((item) => (
            <div key={item}>{item}</div>
          ))}
        </div>
      </Card>
      <Card title="주의사항">
        <div className="timeline-list">
          {challenge?.caution_message ? <div>{String(challenge.caution_message)}</div> : null}
          {challenge?.contraindication_message ? <div>{String(challenge.contraindication_message)}</div> : null}
          {!challenge?.caution_message && !challenge?.contraindication_message ? (
            <div>컨디션이 좋지 않거나 불편감이 있으면 강도를 낮추고, 필요한 경우 의료진과 상담하세요.</div>
          ) : null}
        </div>
      </Card>
      <Card title="추천 이유">
        <div className="timeline-list">
          <div>혈당/혈압 추적 결과와 생활습관 입력을 바탕으로 추천되는 챌린지입니다.</div>
          <div>오늘 수행 완료와 포기 처리는 내 챌린지 기록에 바로 반영됩니다.</div>
        </div>
      </Card>
    </div>
  );
}
