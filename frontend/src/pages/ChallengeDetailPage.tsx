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
  DIABETES: "당뇨 관리",
  HYPERTENSION: "고혈압 관리",
  DYSLIPIDEMIA: "이상지질혈증 관리",
  OBESITY: "비만 관리",
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

export default function ChallengeDetailPage() {
  const { challengeId } = useParams();
  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [userChallenge, setUserChallenge] = useState<Challenge | null>(null);
  const [logs, setLogs] = useState<ChallengeLog[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const load = async () => {
    if (!challengeId) {
      return;
    }
    const challengeItem = await getChallenge<Challenge>(Number(challengeId));
    setChallenge(challengeItem);
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
    }
  };

  useEffect(() => {
    void load().catch(() => setError("챌린지 상세를 불러오지 못했습니다."));
  }, [challengeId]);

  const join = async () => {
    if (!challengeId) {
      return;
    }
    await joinChallenge(Number(challengeId));
    setMessage("챌린지에 참여했습니다. 내 챌린지 목록에서 오늘 완료를 기록할 수 있습니다.");
    await load();
  };

  const complete = async () => {
    if (!userChallenge?.id) {
      setMessage("먼저 챌린지에 참여해주세요.");
      return;
    }
    await completeToday(Number(userChallenge.id));
    setMessage("오늘 수행을 완료했습니다.");
    await load();
  };

  const giveUp = async () => {
    if (!userChallenge?.id) {
      setMessage("참여 중인 챌린지가 없습니다.");
      return;
    }
    await giveUpChallenge(Number(userChallenge.id));
    setMessage("챌린지를 포기 처리했습니다.");
    await load();
  };

  const category = getCategory(challenge);
  const progress = getProgress(userChallenge, challenge, logs);

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
              <span>목표 지표</span>
              <strong>{getDisplayMetric(challenge?.target_metric)}</strong>
            </div>
            <div>
              <span>대상</span>
              <strong>{getDisplayTargetDisease(challenge)}</strong>
            </div>
            <div>
              <span>상태</span>
              <strong>{getDisplayStatus(userChallenge?.status ?? challenge?.status ?? "ACTIVE")}</strong>
            </div>
          </div>
          <div>
            <span className="muted">진행률</span>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <p className="muted">{progress}% 완료</p>
          </div>
          <div className="challenge-log-grid">
            {logs.length === 0 && <div className="state-box">아직 수행 기록이 없습니다. 오늘 완료를 눌러 첫 기록을 남겨보세요.</div>}
            {logs.map((log) => (
              <span className={log.is_completed ? "badge risk-low" : "badge badge-missing"} key={String(log.id)}>
                {String(log.log_date ?? "기록")} {log.is_completed ? "완료" : "미완료"}
              </span>
            ))}
          </div>
          <button onClick={join}>참여하기</button>
          <button className="secondary" onClick={complete}>오늘 수행 완료</button>
          <button className="danger" onClick={giveUp}>포기하기</button>
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
