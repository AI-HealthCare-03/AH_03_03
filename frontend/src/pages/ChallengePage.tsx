import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  completeToday,
  giveUpChallenge,
  joinChallenge,
  listChallengeLogs,
  listChallenges,
  listMyChallenges,
} from "../api/challenges";
import Card from "../components/Card";

import { type ReactNode } from "react";
import { Salad, Dumbbell, Moon, Pill, Droplets, Droplet, Activity, Medal, Leaf, Gauge, ListChecks } from "lucide-react";

type Challenge = Record<string, unknown>;
type ChallengeLog = Record<string, unknown>;

const PAGE_SIZE = 4;

const tabToCategory: Record<string, string | null> = {
  전체: null,
  식단: "DIET",
  운동: "EXERCISE",
  수면: "SLEEP",
  복약: "MEDICATION",
  수분섭취: "WATER",
  혈압: "BLOOD_PRESSURE",
  혈당: "BLOOD_GLUCOSE",
  생활습관: "HABIT",
};

export const categoryIcon: Record<string, ReactNode> = {
  DIET: <Salad size={20} />,
  EXERCISE: <Dumbbell size={20} />,
  SLEEP: <Moon size={20} />,
  MEDICATION: <Pill size={20} />,
  WATER: <Droplets size={20} />,
  BLOOD_SUGAR: <Droplet size={20} />,
  BLOOD_GLUCOSE: <Droplet size={20} />,
  BLOOD_PRESSURE: <Activity size={20} />,
  HABIT: <ListChecks size={20} />,
  COMMON: <Leaf size={20} />,
  WEIGHT: <Gauge size={20} />,
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

const challengeTypeLabel: Record<string, string> = {
  SPECIAL: "특수 챌린지",
  COMMON: "공통 챌린지",
  GENERAL: "일반 챌린지",
};

const targetDiseaseLabel: Record<string, string> = {
  COMMON: "공통 건강관리",
  GENERAL: "일반 건강관리",
  DIABETES: "당뇨 관리",
  HYPERTENSION: "고혈압 관리",
  DYSLIPIDEMIA: "이상지질혈증 관리",
  OBESITY: "비만 관리",
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

function getCategory(challenge: Challenge): string {
  return String(challenge.category ?? "").toUpperCase();
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

function getDisplayCategory(value: unknown): string {
  const category = String(value ?? "COMMON").toUpperCase();
  return categoryLabel[category] ?? "공통";
}

function getDisplayChallengeType(challenge: Challenge): string {
  const raw = String(challenge.challenge_type ?? "GENERAL").toUpperCase();
  return challengeTypeLabel[raw] ?? "일반 챌린지";
}

function getDifficulty(value: unknown): string {
  const difficulty = String(value ?? "EASY").toUpperCase();
  return difficultyLabel[difficulty] ?? "쉬움";
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

function getDisplayTargetDisease(challenge: Challenge): string {
  const raw = String(challenge.target_disease ?? getTargetDiseaseFromDescription(challenge.description));
  return targetDiseaseLabel[raw.toUpperCase()] ?? "공통 건강관리";
}

function getCleanDescription(value: unknown): string {
  return String(value ?? "").replace(/\[target_disease=[A-Z_]+\]\s*/g, "");
}

function getChallengeId(item: Challenge): number | null {
  const nested = item.challenge as Challenge | undefined;
  const raw = item.challenge_id ?? nested?.id ?? item.id;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function findMasterChallenge(item: Challenge, challengeList: Challenge[]): Challenge | undefined {
  const challengeId = getChallengeId(item);
  return challengeList.find((challenge) => Number(challenge.id) === challengeId);
}

function getChallengeTitle(item: Challenge, challengeList: Challenge[]): string {
  const nested = item.challenge as Challenge | undefined;
  return String(
    item.title ??
      nested?.title ??
      findMasterChallenge(item, challengeList)?.title ??
      "참여 중인 챌린지",
  );
}

function getChallengeDescription(item: Challenge, challengeList: Challenge[]): string {
  const nested = item.challenge as Challenge | undefined;
  return getCleanDescription(item.description ?? nested?.description ?? findMasterChallenge(item, challengeList)?.description ?? "");
}

function getDurationDays(item: Challenge, challengeList: Challenge[]): number {
  const nested = item.challenge as Challenge | undefined;
  const parsed = Number(item.total_days ?? item.duration_days ?? nested?.duration_days ?? findMasterChallenge(item, challengeList)?.duration_days);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 7;
}

function getRequiredDays(item: Challenge, challengeList: Challenge[] = []): number {
  const parsed = Number(item.required_days ?? item.required_count);
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.round(parsed);
  }
  return Math.ceil(getDurationDays(item, challengeList) * 0.8);
}

function getCompletedDays(item: Challenge, challengeList: Challenge[] = []): number | null {
  const completedDays = Number(item.completed_days ?? item.completed_count);
  if (Number.isFinite(completedDays) && completedDays >= 0) {
    return Math.round(completedDays);
  }
  const progress = getProgress(item, challengeList);
  if (progress > 0) {
    return Math.round((progress / 100) * getDurationDays(item, challengeList));
  }
  return null;
}

function hasMetCompletionCondition(item: Challenge, challengeList: Challenge[] = []): boolean {
  if (item.has_met_completion_condition !== undefined) {
    return Boolean(item.has_met_completion_condition);
  }
  return (getCompletedDays(item, challengeList) ?? 0) >= getRequiredDays(item, challengeList);
}

function isFinalizedChallenge(item: Challenge | undefined): boolean {
  if (!item) {
    return false;
  }
  if (Boolean(item.is_finalized) || Boolean(item.completed_at)) {
    return true;
  }
  return ["COMPLETED", "EXPIRED", "FAILED"].includes(normalizeStatus(item.status));
}

function isFinalCompletedChallenge(item: Challenge | undefined, challengeList: Challenge[] = []): boolean {
  if (!item) {
    return false;
  }
  if (Boolean(item.completed_at)) {
    return true;
  }
  return isFinalizedChallenge(item) && hasMetCompletionCondition(item, challengeList);
}

function isMissedChallenge(item: Challenge | undefined, challengeList: Challenge[] = []): boolean {
  return Boolean(item) && isFinalizedChallenge(item) && !isFinalCompletedChallenge(item, challengeList) && !isRejoinableChallengeStatus(item?.status);
}

function getChallengeStatusLabel(item: Challenge | undefined, challengeList: Challenge[] = []): string {
  if (!item || isRejoinableChallengeStatus(item.status)) {
    return "참여 전";
  }
  if (isFinalCompletedChallenge(item, challengeList)) {
    return "완료";
  }
  if (isMissedChallenge(item, challengeList)) {
    return "미달성";
  }
  if (hasMetCompletionCondition(item, challengeList)) {
    return "완료 조건 충족";
  }
  return getDisplayStatus(item.status);
}

function getProgressLabel(item: Challenge, challengeList: Challenge[] = []): string {
  const status = normalizeStatus(item.status);
  const durationDays = getDurationDays(item, challengeList);
  const completedDays = Math.min(getCompletedDays(item, challengeList) ?? 0, durationDays);
  const requiredDays = getRequiredDays(item, challengeList);
  if (isFinalCompletedChallenge(item, challengeList)) {
    return `${durationDays}/${durationDays}일 완료`;
  }
  if (isRejoinableChallengeStatus(status)) {
    return "참여 전";
  }
  if (isMissedChallenge(item, challengeList)) {
    return `${completedDays}/${durationDays}일 미달성`;
  }
  if (hasMetCompletionCondition(item, challengeList)) {
    return `${completedDays}/${durationDays}일 완료 조건 충족 · 진행 중`;
  }
  return `${completedDays}/${durationDays}일 진행 중 · 목표 ${requiredDays}일`;
}

function isTodayCompleted(item: Challenge): boolean {
  return Boolean(item.today_completed ?? item.is_today_completed ?? item.completed_today);
}

function getDailyGoalCount(item: Challenge | undefined, challengeList: Challenge[] = []): number {
  if (!item) {
    return 1;
  }
  const direct = Number(item.daily_goal_count ?? item.daily_goal);
  if (Number.isFinite(direct) && direct > 0) {
    return Math.max(1, Math.min(Math.round(direct), 10));
  }
  const master = findMasterChallenge(item, challengeList);
  const metric = String(item.target_metric ?? master?.target_metric ?? "").toLowerCase();
  if (!["count", "times", "횟수", "회"].some((token) => metric.includes(token))) {
    return 1;
  }
  const matched = String(item.target_value ?? master?.target_value ?? "").match(/\d+/);
  return matched ? Math.max(1, Math.min(Number(matched[0]), 10)) : 1;
}

function getTodayCompletedCount(item: Challenge | undefined): number {
  if (!item) {
    return 0;
  }
  const count = Number(item.today_completed_count ?? item.today_count);
  if (Number.isFinite(count) && count >= 0) {
    return Math.round(count);
  }
  return isTodayCompleted(item) ? 1 : 0;
}

function getTodayActionLabel(item: Challenge | undefined, challengeList: Challenge[] = []): string {
  const dailyGoal = getDailyGoalCount(item, challengeList);
  const current = getTodayCompletedCount(item);
  if (current >= dailyGoal) {
    return "오늘 완료됨";
  }
  if (dailyGoal > 1) {
    return `${current + 1}/${dailyGoal} 수행하기`;
  }
  return "오늘 수행 완료";
}

function getLocalDateKey(date: Date): string {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
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
  return getLocalDateKey(parsed);
}

function formatDateLabel(value: unknown): string {
  const key = getDateKey(value);
  if (!key) {
    return "-";
  }
  const [year, month, day] = key.split("-");
  return `${year}. ${Number(month)}. ${Number(day)}.`;
}

function getStartedDate(item: Challenge | undefined): string {
  return getDateKey(item?.started_date ?? item?.started_at);
}

function getExpectedDoneDate(item: Challenge | undefined): string {
  return getDateKey(item?.end_date ?? item?.expected_done_date ?? item?.due_date ?? item?.expected_done_at ?? item?.due_at);
}

function getCompletedDate(log: ChallengeLog): string {
  return getDateKey(log.completed_date ?? log.completed_at ?? log.log_date);
}

function isChallengeScheduledForDate(item: Challenge, dateKey: string): boolean {
  const startedDate = getStartedDate(item);
  if (!startedDate) {
    return false;
  }
  const expectedDoneDate = getExpectedDoneDate(item);
  if (!expectedDoneDate) {
    return startedDate === dateKey;
  }
  return startedDate <= dateKey && dateKey < expectedDoneDate;
}

function getProgress(item: Challenge, challengeList: Challenge[] = []): number {
  if (isRejoinableChallengeStatus(item.status)) {
    return 0;
  }
  if (isFinalCompletedChallenge(item, challengeList)) {
    return 100;
  }
  const explicit = Number(item.progress ?? item.progress_rate);
  if (Number.isFinite(explicit)) {
    return Math.max(0, Math.min(explicit > 1 ? explicit : explicit * 100, 100));
  }
  const completedDays = Number(item.completed_days ?? item.completed_count);
  const durationDays = getDurationDays(item, challengeList);
  if (Number.isFinite(completedDays) && completedDays >= 0 && durationDays > 0) {
    return Math.max(0, Math.min(Math.round((completedDays / durationDays) * 100), 100));
  }
  const rate = Number(item.completion_rate);
  if (Number.isFinite(rate)) {
    return Math.max(0, Math.min(rate, 100));
  }
  return ["ACTIVE", "IN_PROGRESS", "JOINED"].includes(normalizeStatus(item.status)) ? 40 : 0;
}

function isJoinedStatus(item: Challenge | undefined): boolean {
  if (!item) {
    return false;
  }
  const status = normalizeStatus(item.status);
  return !["", "PENDING"].includes(status) && !isRejoinableChallengeStatus(status);
}

function isActiveChallengeStatus(item: Challenge | undefined): boolean {
  if (!item) {
    return false;
  }
  if (item.completed_at || item.canceled_at) {
    return false;
  }
  if (isFinalizedChallenge(item)) {
    return false;
  }
  return ["ACTIVE", "IN_PROGRESS", "JOINED"].includes(normalizeStatus(item.status));
}

export default function ChallengePage() {
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [myChallenges, setMyChallenges] = useState<Challenge[]>([]);
  const [logsByUserChallengeId, setLogsByUserChallengeId] = useState<Record<number, ChallengeLog[]>>({});
  const [activeTab, setActiveTab] = useState("전체");
  const [activeDifficulty, setActiveDifficulty] = useState("전체");
  const [page, setPage] = useState(1);
  const [calendarMonth, setCalendarMonth] = useState(() => new Date());
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const [actionId, setActionId] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const [challengeItems, myChallengeItems] = await Promise.all([
        listChallenges<Challenge[]>({ limit: 200, offset: 0 }),
        listMyChallenges<Challenge[]>({ limit: 100, offset: 0 }),
      ]);
      setChallenges(challengeItems);
      setMyChallenges(myChallengeItems);
      const logEntries = await Promise.all(
        myChallengeItems.map(async (item) => {
          const id = Number(item.id);
          if (!Number.isFinite(id)) {
            return null;
          }
          try {
            const logs = await listChallengeLogs<ChallengeLog[]>(id);
            return [id, logs] as const;
          } catch {
            return [id, []] as const;
          }
        }),
      );
      setLogsByUserChallengeId(Object.fromEntries(logEntries.filter((entry): entry is readonly [number, ChallengeLog[]] => Boolean(entry))));
    } catch (err) {
      setError(err instanceof Error ? err.message : "챌린지 목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);
  const filteredChallenges = useMemo(() => {
    const category = tabToCategory[activeTab];
    let result: Challenge[];
    if (activeTab === "생활습관") {
      result = challenges.filter((challenge) => getCategory(challenge) === "HABIT" || getCategory(challenge) === "WEIGHT");
    } else if (!category) {
      result = challenges;
    } else {
      result = challenges.filter((challenge) => getCategory(challenge) === category);
    }
    if (activeDifficulty !== "전체") {
      result = result.filter((challenge) => String(challenge.difficulty ?? "").toUpperCase() === activeDifficulty);
    }
    return result;
  }, [activeTab, activeDifficulty, challenges]);

  const myChallengeByMasterId = useMemo(() => {
    const mapped = new Map<number, Challenge>();
    myChallenges.forEach((item) => {
      if (isRejoinableChallengeStatus(item.status)) {
        return;
      }
      const challengeId = getChallengeId(item);
      if (challengeId !== null) {
        mapped.set(challengeId, item);
      }
    });
    return mapped;
  }, [myChallenges]);
  const myChallengeById = useMemo(() => {
    const mapped = new Map<number, Challenge>();
    myChallenges.forEach((item) => {
      const id = Number(item.id);
      if (Number.isFinite(id)) {
        mapped.set(id, item);
      }
    });
    return mapped;
  }, [myChallenges]);

  const pageCount = Math.max(1, Math.ceil(filteredChallenges.length / PAGE_SIZE));
  const pagedChallenges = filteredChallenges.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  useEffect(() => {
    setPage(1);
  }, [activeTab]);

  const activeMyChallenges = myChallenges.filter(isActiveChallengeStatus);
  const todayDoneCount = activeMyChallenges.filter((item) => getTodayCompletedCount(item) >= getDailyGoalCount(item, challenges)).length;
  const averageProgress =
    activeMyChallenges.length > 0
      ? Math.round(activeMyChallenges.reduce((sum, item) => sum + getProgress(item, challenges), 0) / activeMyChallenges.length)
      : 0;
  const weekStart = new Date();
  weekStart.setHours(0, 0, 0, 0);
  weekStart.setDate(weekStart.getDate() - 6);
  const weeklyLogCount = Object.values(logsByUserChallengeId)
    .flat()
    .filter((log) => {
      const rawDate = String(log.log_date ?? "");
      const parsed = rawDate ? new Date(`${rawDate.slice(0, 10)}T00:00:00`) : null;
      return Boolean(log.is_completed) && parsed !== null && parsed >= weekStart;
    }).length;

  const firstOfMonth = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth(), 1);
  const monthDays = new Date(calendarMonth.getFullYear(), calendarMonth.getMonth() + 1, 0).getDate();
  const leadingBlankDays = firstOfMonth.getDay();
  const todayKey = getLocalDateKey(new Date());
  const calendarCells = [
    ...Array.from({ length: leadingBlankDays }, (_, index) => ({ key: `blank-${index}`, day: null as number | null })),
    ...Array.from({ length: monthDays }, (_, index) => ({ key: `day-${index + 1}`, day: index + 1 })),
  ];

  const getCalendarDayState = (day: number) => {
    const key = `${calendarMonth.getFullYear()}-${String(calendarMonth.getMonth() + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
    const scheduledIds = new Set<number>();
    activeMyChallenges.forEach((item) => {
      const id = Number(item.id);
      if (Number.isFinite(id) && isChallengeScheduledForDate(item, key)) {
        scheduledIds.add(id);
      }
    });

    const completedCountByChallengeId = new Map<number, number>();
    Object.entries(logsByUserChallengeId).forEach(([rawId, logs]) => {
      const id = Number(rawId);
      const count = logs.filter((log) => Boolean(log.is_completed) && getCompletedDate(log) === key).length;
      if (Number.isFinite(id) && count > 0) {
        completedCountByChallengeId.set(id, count);
        scheduledIds.add(id);
      }
    });

    let total = 0;
    let done = 0;
    scheduledIds.forEach((id) => {
      const challenge = myChallengeById.get(id);
      const dailyGoal = getDailyGoalCount(challenge, challenges);
      total += dailyGoal;
      done += Math.min(completedCountByChallengeId.get(id) ?? 0, dailyGoal);
    });
    return { done, total, isToday: key === todayKey, label: total > 0 ? `${done}/${total}` : "" };
  };

  const handleJoin = async (challengeId: number) => {
    setActionId(`join-${challengeId}`);
    setError("");
    setMessage("");
    try {
      await joinChallenge(challengeId);
      await load();
      setMessage("챌린지에 참여했습니다. 오늘 수행 기록을 남길 수 있습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "챌린지 참여 처리에 실패했습니다.");
    } finally {
      setActionId(null);
    }
  };

  const handleComplete = async (userChallenge: Challenge) => {
    const id = Number(userChallenge.id);
    if (!Number.isFinite(id)) {
      return;
    }
    setActionId(`complete-${id}`);
    setError("");
    setMessage("");
    try {
      await completeToday(id);
      await load();
      setMessage("오늘 수행을 완료했습니다. 달력과 진행률에 반영되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "오늘 수행 완료 처리에 실패했습니다.");
    } finally {
      setActionId(null);
    }
  };

  const handleGiveUp = async (userChallenge: Challenge) => {
    const id = Number(userChallenge.id);
    if (!Number.isFinite(id)) {
      return;
    }
    if (!window.confirm("챌린지 참여를 취소하면 현재 진행 상태가 중단됩니다. 계속하시겠습니까?")) {
      return;
    }
    setActionId(`give-up-${id}`);
    setError("");
    setMessage("");
    try {
      await giveUpChallenge(id);
      await load();
      setMessage("챌린지 참여를 취소했습니다. 다시 시작할 수 있습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "챌린지 참여 취소 처리에 실패했습니다.");
    } finally {
      setActionId(null);
    }
  };

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>챌린지</h1>
          <p>위험도 분석 결과에 맞는 생활습관 챌린지를 시작해보세요.</p>
        </div>
      </div>
      <div className="filter-tabs">
        {["전체", "식단", "운동", "수면", "복약", "수분섭취", "혈압", "혈당", "생활습관"].map((tab) => (
          <button
            className={activeTab === tab ? "filter-tab active" : "filter-tab"}
            key={tab}
            onClick={() => {
              setActiveTab(tab);
              setPage(1);
            }}
          >
            {tab}
          </button>
        ))}
      </div>
      {error && <div className="state-box">{error}</div>}
      {message && <div className="state-box">{message}</div>}
      <div className="challenge-page-layout">
        <main className="challenge-main-list">
          <Card
            title="챌린지 목록"
            actions={
              <select
                value={activeDifficulty}
                onChange={(e) => {
                  setActiveDifficulty(e.target.value);
                  setPage(1);
                }}
                style={{ padding: "4px 8px", borderRadius: "8px", border: "1px solid var(--color-border)", background: "var(--color-surface)", cursor: "pointer", fontSize: "14px", width: "fit-content" }}
              >
                <option value="전체">난이도 전체</option>
                <option value="EASY">쉬움</option>
                <option value="NORMAL">보통</option>
                <option value="HARD">어려움</option>
              </select>
            }
          >
            <div className="challenge-list">
              {loading && <div className="state-box">챌린지 목록을 불러오는 중입니다.</div>}
              {!loading && filteredChallenges.length === 0 && (
                <div className="state-box">현재 선택한 카테고리에 표시할 챌린지가 없습니다.</div>
              )}
              {pagedChallenges.map((challenge) => {
                const category = getCategory(challenge);
                const challengeId = Number(challenge.id);
                const joinedChallenge = Number.isFinite(challengeId) ? myChallengeByMasterId.get(challengeId) : undefined;
                const joined = isJoinedStatus(joinedChallenge);
                const activeJoined = isActiveChallengeStatus(joinedChallenge);
                const progress = joinedChallenge ? getProgress(joinedChallenge, challenges) : 0;
                const todayCount = getTodayCompletedCount(joinedChallenge);
                const dailyGoal = getDailyGoalCount(joinedChallenge ?? challenge, challenges);
                const todayComplete = todayCount >= dailyGoal;
                return (
                  <article className="challenge-list-card" key={String(challenge.id)}>
                    <div className="challenge-card-header">
                      <span className="challenge-icon">{categoryIcon[category] ?? "🌿"}</span>
                      <div>
                        <strong>{String(challenge.title)}</strong>
                        <div className="challenge-card-meta">
                          <span className="badge risk-low">난이도 {getDifficulty(challenge.difficulty)}</span>
                          <span className="badge risk-medium">{String(challenge.duration_days ?? 7)}일</span>
                          <span className="badge badge-reference">{getDisplayChallengeType(challenge)}</span>
                          <span className="badge badge-reference">{getDisplayCategory(category)}</span>
                          <span className="badge badge-reference">대상: {getDisplayTargetDisease(challenge)}</span>
                          <span className="badge badge-reference">
                            참여 상태: {joinedChallenge ? getChallengeStatusLabel(joinedChallenge, challenges) : "참여 전"}
                          </span>
                        </div>
                      </div>
                    </div>
                    {joinedChallenge ? (
                      <>
                        <div className="challenge-date-meta">
                          <span>실행일: {formatDateLabel(joinedChallenge.started_date ?? joinedChallenge.started_at)}</span>
                          <span>완료 예정일: {formatDateLabel(joinedChallenge.end_date ?? joinedChallenge.expected_done_date ?? joinedChallenge.expected_done_at)}</span>
                        </div>
                        <div className="challenge-progress">
                          <span className="muted">{getProgressLabel(joinedChallenge, challenges)}</span>
                          <div className="progress-bar">
                            <div className="progress-fill" style={{ width: `${progress}%` }} />
                          </div>
                        </div>
                      </>
                    ) : (
                      <p className="muted" style={{ overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
                        {getCleanDescription(challenge.description) || "상세 버튼을 눌러 챌린지 내용을 확인하세요."}
                      </p>
                    )}
                    <div className="challenge-card-actions">
                      <Link className="button secondary" to={`/challenges/${String(challenge.id)}`}>
                        상세
                      </Link>
                      {activeJoined && joinedChallenge ? (
                        <button
                          disabled={todayComplete || actionId === `complete-${String(joinedChallenge.id)}`}
                          onClick={() => void handleComplete(joinedChallenge)}
                          type="button"
                        >
                          {actionId === `complete-${String(joinedChallenge.id)}` ? "저장 중..." : getTodayActionLabel(joinedChallenge, challenges)}
                        </button>
                      ) : joined ? (
                        <button disabled type="button">
                          {getChallengeStatusLabel(joinedChallenge, challenges)}
                        </button>
                      ) : (
                        <button disabled={actionId === `join-${String(challenge.id)}`} onClick={() => void handleJoin(Number(challenge.id))} type="button">
                          {actionId === `join-${String(challenge.id)}` ? "시작 중..." : "지금 수행하기"}
                        </button>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
            <div className="pagination-row">
              <button className="secondary compact-button" disabled={page <= 1} onClick={() => { setPage((prev) => Math.max(1, prev - 1)); window.scrollTo(0, 0); }}>
                이전
              </button>
              <span>
                {page} / {pageCount}
              </span>
              <button className="secondary compact-button" disabled={page >= pageCount} onClick={() => { setPage((prev) => Math.min(pageCount, prev + 1)); window.scrollTo(0, 0); }}>
                다음
              </button>
            </div>
          </Card>
        </main>
        <aside className="my-challenge-panel">
          <Card title="내 챌린지 요약">
            <div className="challenge-summary-metrics">
              <div>
                <span>참여 중</span>
                <strong>{activeMyChallenges.length}</strong>
              </div>
              <div>
                <span>오늘 완료</span>
                <strong>{todayDoneCount}</strong>
              </div>
              <div>
                <span>이번 주 기록</span>
                <strong>{weeklyLogCount}</strong>
              </div>
              <div>
                <span>평균 진행률</span>
                <strong>{averageProgress}%</strong>
              </div>
            </div>
            <div className="my-challenge-summary-list">
              {!loading && activeMyChallenges.length === 0 && (
                <div className="compact-empty-state">아직 참여 중인 챌린지가 없습니다. 관심 있는 챌린지를 시작해보세요.</div>
              )}
              {activeMyChallenges.slice(0, 4).map((challenge) => {
                const master = findMasterChallenge(challenge, challenges);
                const category = getCategory(master ?? challenge);
                const progress = getProgress(challenge, challenges);
                return (
                  <div className="my-challenge-summary-card" key={String(challenge.id)}>
                    <div className="challenge-card-header compact">
                      <span className="challenge-icon">{categoryIcon[category] ?? "🌿"}</span>
                      <div>
                        <strong>{getChallengeTitle(challenge, challenges)}</strong>
                        <p>{getChallengeStatusLabel(challenge, challenges)}</p>
                      </div>
                    </div>
                    <div className="challenge-progress">
                      <span className="muted">{getProgressLabel(challenge, challenges)}</span>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                      </div>
                    </div>
                    <div className="challenge-date-meta compact">
                      <span>실행일: {formatDateLabel(challenge.started_date ?? challenge.started_at)}</span>
                      <span>완료 예정일: {formatDateLabel(challenge.end_date ?? challenge.expected_done_date ?? challenge.expected_done_at)}</span>
                    </div>
                    <div className="button-row">
                      <span className={isTodayCompleted(challenge) ? "badge risk-low" : "badge badge-missing"}>
                        오늘 수행: {getTodayCompletedCount(challenge)}/{getDailyGoalCount(challenge, challenges)}
                      </span>
                      <Link className="button secondary compact-button" to={`/challenges/${String(getChallengeId(challenge) ?? "")}`}>
                        상세
                      </Link>
                    </div>
                    <div className="button-row">
                      {isActiveChallengeStatus(challenge) ? (
                        <>
                          <button
                            className="compact-button"
                            disabled={getTodayCompletedCount(challenge) >= getDailyGoalCount(challenge, challenges) || actionId === `complete-${String(challenge.id)}`}
                            onClick={() => void handleComplete(challenge)}
                            type="button"
                          >
                            {actionId === `complete-${String(challenge.id)}` ? "저장 중..." : getTodayActionLabel(challenge, challenges)}
                          </button>
                          <button
                            className="secondary compact-button"
                            disabled={actionId === `give-up-${String(challenge.id)}`}
                            onClick={() => void handleGiveUp(challenge)}
                            type="button"
                          >
                            참여 취소
                          </button>
                        </>
                      ) : (
                        <span className="badge badge-reference">{getChallengeStatusLabel(challenge, challenges)}</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
          <Card title="챌린지 달력">
            <div className="challenge-calendar-header">
              <button
                className="secondary compact-button"
                onClick={() => setCalendarMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() - 1, 1))}
                type="button"
              >
                이전 달
              </button>
              <strong>
                {calendarMonth.getFullYear()}년 {calendarMonth.getMonth() + 1}월
              </strong>
              <button
                className="secondary compact-button"
                onClick={() => setCalendarMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() + 1, 1))}
                type="button"
              >
                다음 달
              </button>
            </div>
            <div className="challenge-calendar-grid">
              {["일", "월", "화", "수", "목", "금", "토"].map((day) => (
                <span className="challenge-calendar-weekday" key={day}>
                  {day}
                </span>
              ))}
              {calendarCells.map((cell) => {
                if (cell.day === null) {
                  return <span className="challenge-calendar-cell empty" key={cell.key} />;
                }
                const state = getCalendarDayState(cell.day);
                const complete = state.total > 0 && state.done >= state.total;
                const partial = state.total > 0 && !complete;
                return (
                  <span
                    className={`challenge-calendar-cell ${state.isToday ? "today" : ""} ${complete ? "complete" : ""} ${partial ? "partial" : ""}`}
                    key={cell.key}
                  >
                    <strong>{cell.day}</strong>
                    {complete ? <em><Medal size={16} /></em> : null}
                    {partial ? <small>{state.label}</small> : null}
                  </span>
                );
              })}
            </div>
          </Card>
        </aside>
      </div>
    </div>
  );
}
