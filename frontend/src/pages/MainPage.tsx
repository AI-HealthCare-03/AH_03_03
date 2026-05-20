import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { getMainSummary, getPublicMain } from "../api/main";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

type MainData = Record<string, unknown>;

const publicFallback = {
  service_title: "HealthCare",
  service_description: "AI 기반 분석과 맞춤 챌린지로 나의 건강 상태를 관리해보세요.",
  sample_health_cards: ["AI 위험도 분석", "헬스케어 코치", "쉬운 기록", "추적 대시보드", "가족 관리"],
  sample_challenges: ["혈당", "혈압", "체중", "챌린지 수행률", "식단 점수"],
  locked_features: ["개인 대시보드", "맞춤 분석", "챌린지 참여"],
};

export default function MainPage() {
  const { backendUser, isAuthenticated } = useAuth();
  const [data, setData] = useState<MainData>(publicFallback);

  useEffect(() => {
    const load = async () => {
      try {
        setData(isAuthenticated ? await getMainSummary<MainData>() : await getPublicMain<MainData>());
      } catch {
        setData(publicFallback);
      }
    };
    void load();
  }, [isAuthenticated]);

  if (isAuthenticated) {
    return (
      <div className="page-stack">
        <div className="page-header">
          <div>
            <h1>안녕하세요, {backendUser?.nickname ?? backendUser?.name ?? "회원"}님</h1>
            <p>오늘의 건강 기록과 위험도 요약을 확인해보세요.</p>
          </div>
          <Link className="button" to="/health">
            건강 분석 시작하기
          </Link>
        </div>
        <div className="metric-grid">
          {["건강정보 입력", "식단 기록", "챌린지 수행", "복약 기록"].map((task) => (
            <Link className="stat-card" key={task} to={task === "식단 기록" ? "/diets" : "/health"}>
              <span>오늘 할 일</span>
              <strong>{task}</strong>
            </Link>
          ))}
        </div>
        <div className="page-grid">
          <Card title="위험도 요약">
            <div className="metric-grid">
              <div>
                <span>당뇨 위험도</span>
                <strong>MEDIUM</strong>
              </div>
              <div>
                <span>고혈압 위험도</span>
                <strong>MEDIUM</strong>
              </div>
              <div>
                <span>종합 위험도</span>
                <strong>관리 필요</strong>
              </div>
              <div>
                <span>최근 분석일</span>
                <strong>오늘</strong>
              </div>
            </div>
          </Card>
          <Card title="AI 건강 코멘트">
            <p>{String(data.ai_comment ?? "혈당과 혈압을 함께 추적하면 생활습관 변화 효과를 더 쉽게 볼 수 있어요.")}</p>
          </Card>
          <Card title="추천 액션">
            <div className="button-row">
              <Link className="button secondary" to="/challenges">
                식후 10분 걷기
              </Link>
              <Link className="button secondary" to="/diets">
                식단 이미지 분석
              </Link>
              <Link className="button secondary" to="/dashboard">
                추적 대시보드
              </Link>
            </div>
          </Card>
          <Card title="추적 미니 카드">
            <div className="metric-grid">
              {["혈당", "혈압", "체중", "챌린지 수행률", "식단 점수"].map((label) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>-</strong>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <section className="hero-panel">
        <div>
          <span className="eyebrow">HealthCare MVP</span>
          <h1>당뇨와 고혈압 위험도를 쉽게 확인하고 건강한 습관을 만들어가세요</h1>
          <p>{String(data.service_description)}</p>
          <div className="button-row">
            <Link className="button" to="/signup">
              건강 분석 시작하기
            </Link>
            <Link className="button secondary" to="/login">
              로그인 후 이용 가능
            </Link>
          </div>
        </div>
        <div className="mobile-health-card">
          <span className="badge risk-medium">오늘의 건강 요약</span>
          <strong>혈당 108 mg/dL</strong>
          <strong>혈압 132/84 mmHg</strong>
          <strong>챌린지 수행률 72%</strong>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: "72%" }} />
          </div>
          <p className="muted">모바일 건강 카드 예시입니다.</p>
        </div>
      </section>
      <div className="metric-grid">
        {publicFallback.sample_health_cards.map((feature) => (
          <div className="metric-card card" key={feature}>
            <span>주요 기능</span>
            <strong>{feature}</strong>
            <p className="muted">로그인 후 이용 가능</p>
          </div>
        ))}
      </div>
      <Card title="예시 추적 대시보드">
        <div className="metric-grid">
          {publicFallback.sample_challenges.map((label) => (
            <div key={label}>
              <span>{label}</span>
              <strong>-</strong>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
