import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { completeToday, giveUpChallenge, joinChallenge, listChallenges, listMyChallenges } from "../api/challenges";
import Card from "../components/Card";

type Challenge = Record<string, unknown>;

export default function ChallengePage() {
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [myChallenges, setMyChallenges] = useState<Challenge[]>([]);
  const [activeTab, setActiveTab] = useState("전체");

  const load = async () => {
    setChallenges(await listChallenges<Challenge[]>());
    setMyChallenges(await listMyChallenges<Challenge[]>());
  };

  useEffect(() => {
    void load().catch(() => undefined);
  }, []);

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
          <button className={activeTab === tab ? "filter-tab active" : "filter-tab"} key={tab} onClick={() => setActiveTab(tab)}>
            {tab}
          </button>
        ))}
      </div>
      <div className="page-grid">
      <Card title="챌린지 목록">
        <div className="card-list">
          {challenges.map((challenge) => (
            <div className="mini-card" key={String(challenge.id)}>
              <div className="upload-box">이미지</div>
              <strong>{String(challenge.title)}</strong>
              <p>{String(challenge.description ?? "")}</p>
              <div className="button-row">
                <span className="badge risk-low">난이도 쉬움</span>
                <span className="badge risk-medium">{String(challenge.duration_days ?? 7)}일</span>
              </div>
              <div className="progress-bar"><div className="progress-fill" style={{ width: "0%" }} /></div>
              <div className="button-row">
                <Link className="button secondary" to={`/challenges/${String(challenge.id)}`}>
                  상세
                </Link>
                <button onClick={() => void joinChallenge(Number(challenge.id)).then(load)}>참여하기</button>
              </div>
            </div>
          ))}
        </div>
        <button className="secondary" style={{ marginTop: 16 }}>더 많은 챌린지 보기</button>
      </Card>
      <Card title="내 챌린지">
        <div className="card-list">
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
