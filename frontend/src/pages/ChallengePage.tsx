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
  수분섭취: "WATER",
};

const categoryIcon: Record<string, string> = {
  DIET: "🍽",
  EXERCISE: "🏃",
  SLEEP: "🌙",
  MEDICATION: "💊",
  WATER: "💧",
  BLOOD_SUGAR: "🩸",
  BLOOD_PRESSURE: "🫀",
};

function getCategory(challenge: Challenge): string {
  return String(challenge.category ?? "").toUpperCase();
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
        {["전체", "식단", "운동", "수면", "복약", "수분섭취"].map((tab) => (
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
      <div className="page-grid">
      <Card title="챌린지 목록">
        <div className="card-list">
          {loading && <div className="state-box">챌린지 목록을 불러오는 중입니다.</div>}
          {!loading && filteredChallenges.length === 0 && (
            <div className="state-box">현재 선택한 카테고리에 표시할 챌린지가 없습니다.</div>
          )}
          {filteredChallenges.map((challenge) => {
            const category = getCategory(challenge);
            return (
            <div className="mini-card" key={String(challenge.id)}>
              <div className="upload-box" aria-label={`${category} challenge icon`}>
                <span style={{ fontSize: 32 }}>{categoryIcon[category] ?? "✅"}</span>
              </div>
              <strong>{String(challenge.title)}</strong>
              <p>{String(challenge.description ?? "")}</p>
              <div className="button-row">
                <span className="badge risk-low">난이도 쉬움</span>
                <span className="badge risk-medium">{String(challenge.duration_days ?? 7)}일</span>
                <span className="badge">{category || "COMMON"}</span>
              </div>
              <div className="progress-bar"><div className="progress-fill" style={{ width: "0%" }} /></div>
              <div className="button-row">
                <Link className="button secondary" to={`/challenges/${String(challenge.id)}`}>
                  상세
                </Link>
                <button onClick={() => void joinChallenge(Number(challenge.id)).then(load)}>참여하기</button>
              </div>
            </div>
            );
          })}
        </div>
        <button className="secondary" style={{ marginTop: 16 }} onClick={() => setLimit((prev) => prev + 8)}>
          더 많은 챌린지 보기
        </button>
      </Card>
      <Card title="내 챌린지">
        <div className="card-list">
          {!loading && myChallenges.length === 0 && (
            <div className="state-box">아직 참여 중인 챌린지가 없습니다. 관심 있는 챌린지를 시작해보세요.</div>
          )}
          {myChallenges.map((challenge) => (
            <div className="mini-card" key={String(challenge.id)}>
              <strong>#{String(challenge.id)} {String(challenge.status)}</strong>
              <div className="button-row">
                <button onClick={() => void completeToday(Number(challenge.id)).then(load)}>오늘 완료</button>
                <button className="secondary" onClick={() => void giveUpChallenge(Number(challenge.id)).then(load)}>
                  포기
                </button>
              </div>
            </div>
          ))}
        </div>
      </Card>
      </div>
    </div>
  );
}
