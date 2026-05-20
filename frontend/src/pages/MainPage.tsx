import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { getMainSummary, getPublicMain } from "../api/main";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

type MainData = Record<string, unknown>;

const publicFallback = {
  service_title: "AI Health MVP",
  service_description: "건강검진 기록을 기반으로 당뇨, 비만, 이상지질혈증 위험도를 관리합니다.",
  sample_health_cards: ["건강정보 입력", "위험도 더미 분석", "챌린지 추천"],
  sample_challenges: ["혈당 기록하기", "식단 점수 올리기", "주 3회 운동"],
  locked_features: ["개인 대시보드", "맞춤 분석", "챌린지 참여"],
  cta_buttons: ["로그인", "회원가입"],
};

export default function MainPage() {
  const { firebaseUser } = useAuth();
  const [data, setData] = useState<MainData>(publicFallback);

  useEffect(() => {
    const load = async () => {
      try {
        setData(firebaseUser ? await getMainSummary<MainData>() : await getPublicMain<MainData>());
      } catch {
        setData(publicFallback);
      }
    };
    void load();
  }, [firebaseUser]);

  return (
    <div className="page-grid">
      <Card title={String(data.service_title ?? "오늘의 건강 홈")}>
        <p>{String(data.service_description ?? "로그인 후 개인 건강 요약을 확인하세요.")}</p>
        {!firebaseUser && (
          <div className="button-row">
            <Link className="button" to="/login">
              로그인
            </Link>
            <Link className="button secondary" to="/signup">
              회원가입
            </Link>
          </div>
        )}
      </Card>
      <Card title="건강 분석 카드">
        <pre>{JSON.stringify(data.latest_analysis_summary ?? data.sample_health_cards ?? [], null, 2)}</pre>
      </Card>
      <Card title="챌린지 / 오늘 할 일">
        <pre>{JSON.stringify(data.today_tasks ?? data.sample_challenges ?? [], null, 2)}</pre>
      </Card>
      <Card title="알림 요약">
        <pre>{JSON.stringify(data.notification_summary ?? data.locked_features ?? [], null, 2)}</pre>
      </Card>
    </div>
  );
}
