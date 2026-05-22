import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { completeToday, giveUpChallenge, joinChallenge, listChallenges, listMyChallenges } from "../api/challenges";
import Card from "../components/Card";

type Challenge = Record<string, unknown>;

const tabToCategory: Record<string, string | null> = {
  전체: null,
  식단: "DIET",
  운동: "EXERCISE",
  수면: "SLEEP",
  복약: "MEDICATION",
  수분섭취: "HABIT",
  혈압: "BLOOD_PRESSURE",
  혈당: "BLOOD_GLUCOSE",
  생활습관: "HABIT",
};

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

function getDisplayCategory(value: unknown): string {
  const category = String(value ?? "COMMON").toUpperCase();
  return categoryLabel[category] ?? "공통";
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
  const parsed = Number(item.duration_days ?? nested?.duration_days ?? findMasterChallenge(item, challengeList)?.duration_days);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 7;
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

function getProgressLabel(item: Challenge, challengeList: Challenge[] = []): string {
  const status = normalizeStatus(item.status);
  const durationDays = getDurationDays(item, challengeList);
  const completedDays = getCompletedDays(item, challengeList);
  if (status === "COMPLETED") {
    return `${durationDays}/${durationDays}일 완료`;
  }
  if (["GIVE_UP", "GIVEN_UP", "FAILED", "CANCELED", "CANCELLED"].includes(status)) {
    return "포기";
  }
  if (completedDays !== null) {
    return `${Math.min(completedDays, durationDays)}/${durationDays}일 완료`;
  }
  return getDisplayStatus(status);
}

function isTodayCompleted(item: Challenge): boolean {
  return Boolean(item.today_completed ?? item.is_today_completed ?? item.completed_today);
}

function getProgress(item: Challenge, challengeList: Challenge[] = []): number {
  const explicit = Number(item.progress ?? item.progress_rate);
  if (Number.isFinite(explicit)) {
    return Math.max(0, Math.min(explicit > 1 ? explicit : explicit * 100, 100));
  }
  const completedDays = Number(item.completed_days ?? item.completed_count);
  const durationDays = getDurationDays(item, challengeList);
  if (Number.isFinite(completedDays) && completedDays >= 0 && durationDays > 0) {
    return Math.max(0, Math.min(Math.round((completedDays / durationDays) * 100), 100));
  }
  const status = normalizeStatus(item.status);
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

export default function ChallengePage() {
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [myChallenges, setMyChallenges] = useState<Challenge[]>([]);
  const [activeTab, setActiveTab] = useState("전체");
  const [limit, setLimit] = useState(8);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const category = tabToCategory[activeTab] ?? undefined;
      const [challengeItems, myChallengeItems] = await Promise.all([
        listChallenges<Challenge[]>({ category, limit, offset: 0 }),
        listMyChallenges<Challenge[]>({ limit: 20, offset: 0 }),
      ]);
      setChallenges(challengeItems);
      setMyChallenges(myChallengeItems);
    } catch (err) {
      setError(err instanceof Error ? err.message : "챌린지 목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, [activeTab, limit]);

  const filteredChallenges = useMemo(() => {
    const category = tabToCategory[activeTab];
    if (!category) {
      return challenges;
    }
    return challenges.filter((challenge) => getCategory(challenge) === category);
  }, [activeTab, challenges]);

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
              setLimit(8);
            }}
          >
            {tab}
          </button>
        ))}
      </div>
      {error && <div className="state-box">{error}</div>}
      <div className="challenge-page-layout">
        <main className="challenge-main-list">
          <Card title="챌린지 목록">
            <div className="challenge-list">
              {loading && <div className="state-box">챌린지 목록을 불러오는 중입니다.</div>}
              {!loading && filteredChallenges.length === 0 && (
                <div className="state-box">현재 선택한 카테고리에 표시할 챌린지가 없습니다.</div>
              )}
              {filteredChallenges.map((challenge) => {
                const category = getCategory(challenge);
                return (
                  <article className="challenge-list-card" key={String(challenge.id)}>
                    <div className="challenge-card-header">
                      <span className="challenge-icon">{categoryIcon[category] ?? "🌿"}</span>
                      <div>
                        <strong>{String(challenge.title)}</strong>
                        <p>{getCleanDescription(challenge.description)}</p>
                      </div>
                    </div>
                    <div className="challenge-card-meta">
                      <span className="badge risk-low">난이도 {getDifficulty(challenge.difficulty)}</span>
                      <span className="badge risk-medium">{String(challenge.duration_days ?? 7)}일</span>
                      <span className="badge badge-reference">{getDisplayCategory(category)}</span>
                      <span className="badge badge-reference">대상: {getDisplayTargetDisease(challenge)}</span>
                      <span className="badge badge-reference">참여 상태: 대기</span>
                    </div>
                    <div className="challenge-card-actions">
                      <Link className="button secondary" to={`/challenges/${String(challenge.id)}`}>
                        상세
                      </Link>
                      <button onClick={() => void joinChallenge(Number(challenge.id)).then(load)} type="button">
                        참여하기
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
            <button className="secondary" style={{ marginTop: 16 }} onClick={() => setLimit((prev) => prev + 8)}>
              더 많은 챌린지 보기
            </button>
          </Card>
        </main>
        <aside className="my-challenge-panel">
          <Card title="내 챌린지 요약">
            <div className="my-challenge-summary-list">
              {!loading && myChallenges.length === 0 && (
                <div className="compact-empty-state">아직 참여 중인 챌린지가 없습니다. 관심 있는 챌린지를 시작해보세요.</div>
              )}
              {myChallenges.slice(0, 4).map((challenge) => {
                const master = findMasterChallenge(challenge, challenges);
                const category = getCategory(master ?? challenge);
                const progress = getProgress(challenge, challenges);
                return (
                  <div className="my-challenge-summary-card" key={String(challenge.id)}>
                    <div className="challenge-card-header compact">
                      <span className="challenge-icon">{categoryIcon[category] ?? "🌿"}</span>
                      <div>
                        <strong>{getChallengeTitle(challenge, challenges)}</strong>
                        <p>{getDisplayStatus(challenge.status)}</p>
                      </div>
                    </div>
                    <div className="challenge-progress">
                      <span className="muted">{getProgressLabel(challenge, challenges)}</span>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                      </div>
                    </div>
                    <div className="button-row">
                      <span className={isTodayCompleted(challenge) ? "badge risk-low" : "badge badge-missing"}>
                        오늘 수행: {isTodayCompleted(challenge) ? "완료" : "기록 전"}
                      </span>
                      <Link className="button secondary compact-button" to={`/challenges/${String(getChallengeId(challenge) ?? "")}`}>
                        상세
                      </Link>
                    </div>
                    <div className="button-row">
                      <button className="compact-button" onClick={() => void completeToday(Number(challenge.id)).then(load)} type="button">
                        오늘 완료
                      </button>
                      <button
                        className="secondary compact-button"
                        onClick={() => void giveUpChallenge(Number(challenge.id)).then(load)}
                        type="button"
                      >
                        포기
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </aside>
      </div>
    </div>
  );
}
