import { useEffect, useState } from "react";

import { getLatestAnalysisResults } from "../api/analysis";
import { listMyChallenges } from "../api/challenges";
import { listHealthRecords } from "../api/health";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

type Item = Record<string, unknown>;

export default function MyPage() {
  const { backendUser, firebaseUser } = useAuth();
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
    <div className="page-grid">
      <Card title="내 프로필">
        <p>{backendUser?.email ?? firebaseUser?.email}</p>
        <p>role: {backendUser?.role ?? "USER"}</p>
        <button className="danger" onClick={() => window.confirm("회원탈퇴는 후속 구현에서 안전장치를 추가합니다.")}>
          회원탈퇴
        </button>
      </Card>
      <Card title="건강정보 요약">
        <pre>{JSON.stringify(health[0] ?? null, null, 2)}</pre>
      </Card>
      <Card title="최근 분석 결과">
        <pre>{JSON.stringify(analysis, null, 2)}</pre>
      </Card>
      <Card title="진행 중 챌린지">
        <pre>{JSON.stringify(challenges, null, 2)}</pre>
      </Card>
    </div>
  );
}
