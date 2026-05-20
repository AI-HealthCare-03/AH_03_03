import { useEffect, useState } from "react";

import { completeToday, giveUpChallenge, joinChallenge, listChallenges, listMyChallenges } from "../api/challenges";
import Card from "../components/Card";

type Challenge = Record<string, unknown>;

export default function ChallengePage() {
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [myChallenges, setMyChallenges] = useState<Challenge[]>([]);

  const load = async () => {
    setChallenges(await listChallenges<Challenge[]>());
    setMyChallenges(await listMyChallenges<Challenge[]>());
  };

  useEffect(() => {
    void load().catch(() => undefined);
  }, []);

  return (
    <div className="page-grid">
      <Card title="챌린지 목록">
        <div className="card-list">
          {challenges.map((challenge) => (
            <div className="mini-card" key={String(challenge.id)}>
              <strong>{String(challenge.title)}</strong>
              <p>{String(challenge.description ?? "")}</p>
              <button onClick={() => void joinChallenge(Number(challenge.id)).then(load)}>참여하기</button>
            </div>
          ))}
        </div>
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
  );
}
