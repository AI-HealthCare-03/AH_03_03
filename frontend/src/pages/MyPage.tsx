import { useEffect, useState } from "react";

import { getLatestAnalysisResults } from "../api/analysis";
import { listMyChallenges } from "../api/challenges";
import { listHealthRecords } from "../api/health";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

type Item = Record<string, unknown>;

export default function MyPage() {
  const { backendUser } = useAuth();
  const [health, setHealth] = useState<Item[]>([]);
  const [analysis, setAnalysis] = useState<Item[]>([]);
  const [challenges, setChallenges] = useState<Item[]>([]);

  useEffect(() => {
    const load = async () => {
      setHealth(await listHealthRecords<Item[]>());
      setAnalysis(await getLatestAnalysisResults<Item[]>());
      setChallenges(await listMyChallenges<Item[]>());
    };
    void load().catch(() => undefined);
  }, []);

  return (
    <div className="dashboard-grid">
      <Card title="마이페이지">
        <div className="card-list">
          {["프로필", "기본 건강정보", "복약/영양제", "챌린지 현황", "구독 결제", "알림 설정", "개인정보", "회원탈퇴"].map(
            (item) => (
              <span className={item === "프로필" ? "filter-tab active" : "filter-tab"} key={item}>
                {item}
              </span>
            ),
          )}
        </div>
      </Card>
      <div className="page-stack">
        <Card title="프로필 카드">
          <div className="button-row">
            <span className="avatar">{(backendUser?.nickname ?? backendUser?.name ?? "U").slice(0, 1)}</span>
            <div>
              <strong>{backendUser?.nickname ?? backendUser?.name}</strong>
              <p className="muted">{backendUser?.email}</p>
            </div>
          </div>
          <button className="secondary">수정</button>
        </Card>
        <Card title="건강 정보 테이블">
          <div className="table-list">
            {[
              ["생년월일", backendUser?.birthday],
              ["성별", backendUser?.gender],
              ["키/몸무게", `${String(health[0]?.height_cm ?? "-")}/${String(health[0]?.weight_kg ?? "-")}`],
              ["BMI", health[0]?.bmi],
              ["휴대폰", backendUser?.phone_number],
            ].map(([label, value]) => (
              <div className="table-row" key={String(label)}>
                <span>{String(label)}</span>
                <strong>{String(value ?? "-")}</strong>
                <span />
              </div>
            ))}
          </div>
        </Card>
        <div className="page-grid">
          <Card title="최근 분석 결과">
            <pre>{JSON.stringify(analysis, null, 2)}</pre>
          </Card>
          <Card title="진행 중 챌린지">
            <pre>{JSON.stringify(challenges, null, 2)}</pre>
          </Card>
        </div>
      </div>
    </div>
  );
}
