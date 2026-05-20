import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getChallenge, joinChallenge } from "../api/challenges";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";

type Challenge = Record<string, unknown>;

export default function ChallengeDetailPage() {
  const { challengeId } = useParams();
  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const load = async () => {
      if (!challengeId) {
        return;
      }
      setChallenge(await getChallenge<Challenge>(Number(challengeId)));
    };
    void load().catch(() => setError("챌린지 상세를 불러오지 못했습니다."));
  }, [challengeId]);

  const join = async () => {
    if (!challengeId) {
      return;
    }
    await joinChallenge(Number(challengeId));
    setMessage("챌린지에 참여했습니다. 내 챌린지 목록에서 오늘 완료를 기록할 수 있습니다.");
  };

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
          <span className="eyebrow">{String(challenge?.category ?? "CHALLENGE")}</span>
          <h1>{String(challenge?.title ?? "챌린지")}</h1>
          <div className="upload-box">챌린지 이미지</div>
          <p>{String(challenge?.description ?? "건강 습관을 작게 시작해보세요.")}</p>
          <div className="metric-grid">
            <div>
              <span>기간</span>
              <strong>{String(challenge?.duration_days ?? "-")}일</strong>
            </div>
            <div>
              <span>목표 지표</span>
              <strong>{String(challenge?.target_metric ?? "생활습관")}</strong>
            </div>
            <div>
              <span>상태</span>
              <strong>{String(challenge?.status ?? "ACTIVE")}</strong>
            </div>
          </div>
          <div>
            <span className="muted">진행률</span>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: "60%" }} />
            </div>
          </div>
          <div className="button-row">
            {["월", "화", "수", "목", "금"].map((day, index) => (
              <span className={index % 2 === 0 ? "badge risk-low" : "badge"} key={day}>
                {day}
              </span>
            ))}
          </div>
          <button onClick={join}>참여하기</button>
          <button className="secondary">오늘 수행 완료</button>
          <button className="danger">포기하기</button>
        </div>
      </Card>
      <Card title="추천 이유">
        <div className="timeline-list">
          <div>혈당/혈압 추적 결과와 생활습관 입력을 바탕으로 추천되는 챌린지입니다.</div>
          <div>수행 기록 API 연결은 기존 내 챌린지 화면을 사용하며, 상세 액션은 후속 보강 예정입니다.</div>
        </div>
      </Card>
    </div>
  );
}
