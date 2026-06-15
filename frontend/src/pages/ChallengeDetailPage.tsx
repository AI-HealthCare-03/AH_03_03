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

import {
  UtensilsCrossed,
  SportShoe,
  Bed,
  Pill,
  GlassWater,
  Droplet,
  HeartPulse,
  ListTodo,
  Leaf,
  Gauge,
} from "lucide-react";

type Challenge = Record<string, unknown>;
type ChallengeLog = Record<string, unknown>;

const categoryIcon: Record<string, React.ReactNode> = {
  DIET: <UtensilsCrossed size={40} />,
  EXERCISE: <SportShoe size={40} />,
  SLEEP: <Bed size={40} />,
  MEDICATION: <Pill size={40} />,
  WATER: <GlassWater size={40} />,
  BLOOD_SUGAR: <Droplet size={40} />,
  BLOOD_GLUCOSE: <Droplet size={40} />,
  BLOOD_PRESSURE: <HeartPulse size={40} />,
  HABIT: <ListTodo size={40} />,
  COMMON: <Leaf size={40} />,
  WEIGHT: <Gauge size={40} />,
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
  DYSLIPIDEMIA: "이상지질혈증 관리",
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
  GIVEN_UP: "참여 전",
  GIVE_UP: "참여 전",
  CANCELED: "참여 전",
  CANCELLED: "참여 전",
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

const challengeImageMap: Record<string, string> = {
  "감각을 깨워 챌린지": "/images/challenges/sensory-awakening.jpg",
  "경치보며 지루함 달래는 등산 챌린지": "/images/challenges/hiking-beginner.jpg",
  "경치보며 지루함 달래는 찐등산 챌린지": "/images/challenges/hiking-advanced.jpg",
  "계단 타고 지방 타고 챌린지": "/images/challenges/stair-climbing-general.jpg",
  "계단 타고 지방 타고 챌린지(초급)": "/images/challenges/stair-climbing-beginner.jpg",
  "굳지마 발목 챌린지": "/images/challenges/ankle-stretch.jpg",
  "굳지마 어깨 챌린지": "/images/challenges/shoulder-stretch.jpg",
  "굳지마 척추 챌린지": "/images/challenges/spine-stretch.jpg",
  "굽은 어깨 펴고 당당어깨 챌린지": "/images/challenges/shoulder-posture.jpg",
  "나비처럼 날아볼까 챌린지": "/images/challenges/butterfly-walk.jpg",
  "나에게도 탄탄한 가슴이? 챌린지": "/images/challenges/chest-workout.jpg",
  "나에게도 탄탄한 어깨가? 챌린지": "/images/challenges/shoulder-workout.jpg",
  "나에게도 탄탄한 팔뚝이? 챌린지": "/images/challenges/arm-workout.jpg",
  "내 몸을 더 가볍게! 챌린지": "/images/challenges/body-lightening.jpg",
  "달리기 입문을 위한 인터벌 입문 챌린지": "/images/challenges/interval-running-beginner.jpg",
  "당신을 위한 입문 스쿼트 챌린지": "/images/challenges/squat-beginner.jpg",
  "당신을 위한 초급 스쿼트 챌린지": "/images/challenges/squat-intermediate.jpg",
  "데드리프트 입문 챌린지": "/images/challenges/deadlift-beginner.jpg",
  "두 번째 심장 키우기 챌린지(서서)": "/images/challenges/heart-pump-standing.jpg",
  "두 번째 심장 키우기 챌린지(앉아서)": "/images/challenges/heart-pump-sitting.jpg",
  "런지 입문 챌린지": "/images/challenges/lunge-beginner.jpg",
  "맨 몸 런지보다 더 강하게 챌린지": "/images/challenges/lunge-advanced.jpg",
  "맨 몸 스쿼트보다 더 강하게 챌린지": "/images/challenges/squat-advanced.jpg",
  "명상 호흡 챌린지": "/images/challenges/meditation-breathing.jpg",
  "무릎 대고 푸쉬업 챌린지": "/images/challenges/knee-pushup.jpg",
  "벽 짚고 푸쉬업 챌린지": "/images/challenges/wall-pushup.jpg",
  "브릿지 심화 챌린지": "/images/challenges/bridge-advanced.jpg",
  "브릿지 입문 챌린지": "/images/challenges/bridge-beginner.jpg",
  "브릿지 챌린지 입문 챌린지": "/images/challenges/bridge-beginner.jpg",
  "비가오나 눈이오나 유산소는 가능해 챌린지": "/images/challenges/indoor-cardio.jpg",
  "산뜻하게 몸 깨우기 챌린지": "/images/challenges/morning-warmup.jpg",
  "수영 많이 좋아하세요? 챌린지": "/images/challenges/swimming-advanced.jpg",
  "수영 좋아하세요? 챌린지": "/images/challenges/swimming-beginner.jpg",
  "수퍼맨 챌린지": "/images/challenges/superman.jpg",
  "스쿼트 챌린지": "/images/challenges/squat.jpg",
  "심폐는 곧 체력 챌린지": "/images/challenges/cardio-stamina.jpg",
  "싸이클 입문 챌린지": "/images/challenges/cycling-beginner.jpg",
  "싸이클 중급 챌린지": "/images/challenges/cycling-advanced.jpg",
  "싸이클 초급 챌린지": "/images/challenges/cycling-intro.jpg",
  "아쿠아 워킹 챌린지": "/images/challenges/aqua-walking.jpg",
  "앉아서도 건강한 다리 만들기 챌린지": "/images/challenges/leg-exercise-sitting.jpg",
  "어깨 볼륨 up 챌린지": "/images/challenges/shoulder-volume.jpg",
  "유산소 입문을 위한 제자리 걷기 챌린지": "/images/challenges/cardio-stepping.jpg",
  "유산소 초급 챌린지": "/images/challenges/cardio-beginner.jpg",
  "이제 달려볼까 챌린지": "/images/challenges/running-beginner.jpg",
  "인터벌 초급 챌린지": "/images/challenges/interval-beginner.jpg",
  "자! 모두 주먹! 챌린지": "/images/challenges/fist-exercise.jpg",
  "정석 플랭크 챌린지": "/images/challenges/plank-advanced.jpg",
  "조금 더 힘차게 걸어볼까 챌린지": "/images/challenges/brisk-walking.jpg",
  "줄넘기로 지방 태우기 챌린지": "/images/challenges/jump-rope.jpg",
  "차분히 걷기 챌린지": "/images/challenges/calm-walking.jpg",
  "최고의 운동! 데드리프트 입문 챌린지": "/images/challenges/deadlift-advanced.jpg",
  "푸쉬업 챌린지": "/images/challenges/pushup.jpg",
  "푸쉬업 입문 챌린지": "/images/challenges/pushup-beginner.jpg",
  "플랭크 입문 챌린지": "/images/challenges/plank-beginner.jpg",
  "혈당 밟아주기 챌린지(서서)": "/images/challenges/blood-sugar-standing.jpg",
  "혈당 밟아주기 챌린지(앉아서)": "/images/challenges/blood-sugar-sitting.jpg",
  "혈당 싸이클로 밟아주기 챌린지": "/images/challenges/blood-sugar-cycling.jpg",
  "홀로 서기 챌린지": "/images/challenges/solo-standing.jpg",
};

function getChallengeImagePath(title: string): string {
  const normalized = title
    .replace(/챌린지/g, '')
    .replace(/[!?()· \s]/g, '');
  return `/images/challenges/${normalized}.jpg`;
}

function normalizeStatus(value: unknown): string {
  return String(value ?? "").toUpperCase();
}

function getDisplayStatus(value: unknown): string {
  const status = normalizeStatus(value);
  return statusLabel[status] ?? "진행 중";
}

function isRejoinableChallengeStatus(value: unknown): boolean {
  return ["GIVE_UP", "GIVEN_UP", "FAILED", "CANCELED", "CANCELLED"].includes(normalizeStatus(value));
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
  const parsed = Number(userChallenge?.total_days ?? userChallenge?.duration_days ?? challenge?.duration_days);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 7;
}

function getRequiredDays(challenge: Challenge | null, userChallenge?: Challenge | null): number {
  const parsed = Number(userChallenge?.required_days ?? userChallenge?.required_count);
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.round(parsed);
  }
  return Math.ceil(getDurationDays(challenge, userChallenge) * 0.8);
}

function getCompletedDays(userChallenge: Challenge | null, logs: ChallengeLog[]): number {
  const parsed = Number(userChallenge?.completed_days ?? userChallenge?.completed_count);
  if (Number.isFinite(parsed) && parsed >= 0) {
    return Math.round(parsed);
  }
  const completedDates = new Set(
    logs
      .filter((log) => Boolean(log.is_completed) && Boolean(log.completed_at ?? log.completed_date))
      .map((log) => getDateKey(log.completed_date ?? log.completed_at ?? log.log_date)),
  );
  return completedDates.size;
}

function hasMetCompletionCondition(challenge: Challenge | null, userChallenge: Challenge | null, logs: ChallengeLog[]): boolean {
  if (!userChallenge) {
    return false;
  }
  if (userChallenge.has_met_completion_condition !== undefined) {
    return Boolean(userChallenge.has_met_completion_condition);
  }
  return getCompletedDays(userChallenge, logs) >= getRequiredDays(challenge, userChallenge);
}

function isFinalizedUserChallenge(userChallenge: Challenge | null): boolean {
  if (!userChallenge) {
    return false;
  }
  if (Boolean(userChallenge.is_finalized) || Boolean(userChallenge.completed_at)) {
    return true;
  }
  return ["COMPLETED", "EXPIRED", "FAILED"].includes(normalizeStatus(userChallenge.status));
}

function isFinalCompletedChallenge(challenge: Challenge | null, userChallenge: Challenge | null, logs: ChallengeLog[]): boolean {
  if (!userChallenge) {
    return false;
  }
  if (Boolean(userChallenge.completed_at)) {
    return true;
  }
  return isFinalizedUserChallenge(userChallenge) && hasMetCompletionCondition(challenge, userChallenge, logs);
}

function getProgress(userChallenge: Challenge | null, challenge: Challenge | null, logs: ChallengeLog[]): number {
  if (!userChallenge) {
    return 0;
  }
  if (isRejoinableChallengeStatus(userChallenge.status)) {
    return 0;
  }
  if (isFinalCompletedChallenge(challenge, userChallenge, logs)) {
    return 100;
  }
  const explicit = Number(userChallenge.progress ?? userChallenge.progress_rate);
  if (Number.isFinite(explicit)) {
    return Math.max(0, Math.min(explicit > 1 ? explicit : explicit * 100, 100));
  }
  const completionRate = Number(userChallenge.completion_rate);
  if (Number.isFinite(completionRate)) {
    return Math.max(0, Math.min(completionRate, 100));
  }
  const completedDays = getCompletedDays(userChallenge, logs);
  return Math.max(0, Math.min(Math.round((completedDays / getDurationDays(challenge, userChallenge)) * 100), 100));
}

function getProgressSummary(challenge: Challenge | null, userChallenge: Challenge | null, logs: ChallengeLog[]): string {
  if (!userChallenge) {
    return "아직 참여 전입니다.";
  }
  const durationDays = getDurationDays(challenge, userChallenge);
  const completedDays = Math.min(getCompletedDays(userChallenge, logs), durationDays);
  const requiredDays = getRequiredDays(challenge, userChallenge);
  if (isFinalCompletedChallenge(challenge, userChallenge, logs)) {
    return `${durationDays}/${durationDays}일 완료`;
  }
  if (isFinalizedUserChallenge(userChallenge)) {
    return `${completedDays}/${durationDays}일 완료 · 목표 ${requiredDays}일 미달성`;
  }
  if (hasMetCompletionCondition(challenge, userChallenge, logs)) {
    return `${completedDays}/${durationDays}일 완료 · 목표 ${requiredDays}일 충족, 진행 중`;
  }
  return `${completedDays}/${durationDays}일 완료 · 목표 ${requiredDays}일`;
}

function getStatusLabelForUserChallenge(challenge: Challenge | null, userChallenge: Challenge | null, logs: ChallengeLog[]): string {
  if (!userChallenge || isRejoinableChallengeStatus(userChallenge.status)) {
    return "참여 전";
  }
  if (isFinalCompletedChallenge(challenge, userChallenge, logs)) {
    return "완료";
  }
  if (isFinalizedUserChallenge(userChallenge)) {
    return "미달성";
  }
  if (hasMetCompletionCondition(challenge, userChallenge, logs)) {
    return "완료 조건 충족";
  }
  const status = normalizeStatus(userChallenge.status);
  return getDisplayStatus(status);
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

function getDateKey(value: unknown): string {
  const raw = String(value ?? "").trim();
  if (!raw) {
    return "";
  }
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
    return raw;
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return raw.slice(0, 10);
  }
  const month = String(parsed.getMonth() + 1).padStart(2, "0");
  const day = String(parsed.getDate()).padStart(2, "0");
  return `${parsed.getFullYear()}-${month}-${day}`;
}

function formatDateLabel(value: unknown): string {
  const key = getDateKey(value);
  if (!key) {
    return "-";
  }
  const [year, month, day] = key.split("-");
  return `${year}. ${Number(month)}. ${Number(day)}.`;
}

function getTodayCompletedCount(logs: ChallengeLog[]): number {
  const today = getTodayKey();
  return logs.filter((log) => getDateKey(log.completed_date ?? log.completed_at ?? log.log_date) === today && Boolean(log.is_completed)).length;
}

function isTodayCompleted(logs: ChallengeLog[], dailyGoalCount: number): boolean {
  return getTodayCompletedCount(logs) >= dailyGoalCount;
}

function isActiveUserChallenge(userChallenge: Challenge | null): boolean {
  if (!userChallenge) {
    return false;
  }
  if (userChallenge.completed_at || userChallenge.canceled_at) {
    return false;
  }
  if (isFinalizedUserChallenge(userChallenge)) {
    return false;
  }
  return ["ACTIVE", "IN_PROGRESS", "JOINED"].includes(normalizeStatus(userChallenge.status));
}

function isEndedUserChallenge(userChallenge: Challenge | null): boolean {
  return isFinalizedUserChallenge(userChallenge);
}

function getStatusBadgeLabel(challenge: Challenge | null, userChallenge: Challenge | null, logs: ChallengeLog[]): string {
  if (!userChallenge) {
    return "참여 전";
  }
  const dailyGoalCount = Number(userChallenge.daily_goal_count ?? 1);
  if (isTodayCompleted(logs, dailyGoalCount) && isActiveUserChallenge(userChallenge)) {
    return "오늘 완료";
  }
  return getStatusLabelForUserChallenge(challenge, userChallenge, logs);
}

function buildHowToItems(challenge: Challenge | null): string[] {
  const category = getCategory(challenge);
  const duration = getDurationDays(challenge);
  const requiredDays = Math.ceil(duration * 0.8);
  const dailyGoalCount = getDailyGoalCount(challenge);
  const metric = getDisplayMetric(challenge?.target_metric);
  const targetValue = String(challenge?.target_value ?? "").trim();
  const base = [
    `${duration}일 동안 하루 ${dailyGoalCount}회 목표를 실천한 뒤 완료 버튼을 눌러 기록하세요.`,
    `${duration}일 기간이 끝난 뒤 ${requiredDays}일 이상 이행하면 챌린지 완료로 판정됩니다.`,
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
      const matched =
        myItems.find(
          (item) => Number(item.challenge_id) === Number(challengeId) && !isRejoinableChallengeStatus(item.status),
        ) ?? null;
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
    const confirmed = window.confirm("챌린지 참여를 취소하면 현재 진행 상태가 중단됩니다. 계속하시겠습니까?");
    if (!confirmed) {
      return;
    }
    setError("");
    setMessage("");
    setActionLoading("give-up");
    try {
      await giveUpChallenge(Number(userChallenge.id));
      setUserChallenge(null);
      setLogs([]);
      setMessage("챌린지 참여를 취소했습니다. 다시 시작할 수 있습니다.");
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "챌린지 참여 취소 처리에 실패했습니다.");
    } finally {
      setActionLoading(null);
    }
  };

  const category = getCategory(challenge);
  const progress = getProgress(userChallenge, challenge, logs);
  const durationDays = getDurationDays(challenge, userChallenge);
  const dailyGoalCount = getDailyGoalCount(challenge, userChallenge);
  const completedCount = getCompletedDays(userChallenge, logs);
  const todayCompletedCount = getTodayCompletedCount(logs);
  const active = isActiveUserChallenge(userChallenge);
  const todayCompleted = isTodayCompleted(logs, dailyGoalCount);
  const ended = isEndedUserChallenge(userChallenge);
  const nextActionLabel =
    dailyGoalCount > 1 ? `${Math.min(todayCompletedCount + 1, dailyGoalCount)}/${dailyGoalCount} 수행하기` : "오늘 수행 완료";
  const startedLabel = formatDateLabel(userChallenge?.started_date ?? userChallenge?.started_at);
  const expectedDoneLabel = formatDateLabel(userChallenge?.end_date ?? userChallenge?.expected_done_date ?? userChallenge?.expected_done_at);
  const challengeTitle = String(challenge?.title ?? "챌린지");
  const challengeImageSrc = challengeImageMap[challengeTitle] ?? getChallengeImagePath(challengeTitle);

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
          <h1>{challengeTitle}</h1>
          <div
            className="challenge-icon-large"
            aria-label={`${category} challenge icon`}
            style={category === "EXERCISE" ? {} : { background: "none", border: "none" }}
          >
            {category === "EXERCISE" && challenge?.title ? (
              <img
                src={challengeImageSrc}
                alt={`${challengeTitle} 이미지`}
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                }}
                style={{
                  width: "100%",
                  height: "auto",
                  maxHeight: "600px",
                  objectFit: "contain",
                  borderRadius: "inherit",
                }}
              />
            ) : (
              categoryIcon[category] ?? "🌿"
            )}
          </div>
          {category !== "EXERCISE" && (
            <p>{getCleanDescription(challenge?.description) || "건강 습관을 작게 시작해보세요."}</p>
          )}
          <div className="metric-grid" style={{ marginTop: "16px" }}>
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
              <strong>{getStatusBadgeLabel(challenge, userChallenge, logs)}</strong>
            </div>
            {userChallenge && (
              <>
                <div>
                  <span>실행일</span>
                  <strong>{startedLabel}</strong>
                </div>
                <div>
                  <span>완료 예정일</span>
                  <strong>{expectedDoneLabel}</strong>
                </div>
              </>
            )}
          </div>
          {Boolean(challenge?.caution_message) && <div className="state-box warning-card" style={{ marginTop: "16px" }}>{String(challenge?.caution_message)}</div>}
          {Boolean(challenge?.contraindication_message) && (
            <div className="state-box warning-card">{String(challenge?.contraindication_message)}</div>
          )}
          <div>
            <span className="muted">진행률</span>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <p className="muted">
              {getProgressSummary(challenge, userChallenge, logs)} {userChallenge ? `(${Math.round(progress)}%)` : ""}
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
                {actionLoading === "join" ? "시작 중..." : "지금 수행하기"}
              </button>
            )}
            {active && (
              <>
                <button className="secondary" disabled={todayCompleted || actionLoading !== null} onClick={complete} type="button">
                  {actionLoading === "complete" ? "저장 중..." : todayCompleted ? "오늘 완료됨" : nextActionLabel}
                </button>
                <button className="danger" disabled={actionLoading !== null} onClick={giveUp} type="button">
                  {actionLoading === "give-up" ? "처리 중..." : "참여 취소하기"}
                </button>
              </>
            )}
            {ended && (
              <>
                <span className="badge badge-reference">{getStatusBadgeLabel(challenge, userChallenge, logs)}</span>
                {/* TODO: 재도전은 기존 기록과 새 시도 분리를 위한 attempt 구조가 정리되면 활성화한다. */}
                <button className="secondary" disabled type="button">
                  다시 도전하기 준비 중
                </button>
              </>
            )}
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
    </div>
  );
}
