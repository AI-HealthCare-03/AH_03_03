import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { getDietRecord, listDietPhotoResults } from "../api/diets";
import Card from "../components/Card";

type Item = Record<string, unknown>;

export default function DietResultPage() {
  const { dietRecordId } = useParams();
  const navigate = useNavigate();
  const [record, setRecord] = useState<Item | null>(null);
  const [photoResults, setPhotoResults] = useState<Item[]>([]);

  useEffect(() => {
    const load = async () => {
      if (!dietRecordId) {
        return;
      }
      setRecord(await getDietRecord<Item>(Number(dietRecordId)));
      setPhotoResults(await listDietPhotoResults<Item[]>(Number(dietRecordId)));
    };
    void load().catch(() => undefined);
  }, [dietRecordId]);

  return (
    <div className="page-grid">
      <Card
        title="식단 분석 결과"
        actions={
          <Link className="button secondary" to="/diets/history">
            전체 기록
          </Link>
        }
      >
        <div className="score-panel">
          <span>식단 점수</span>
          <strong>{String(record?.diet_score ?? "-")}</strong>
          <p>{String(record?.diet_feedback ?? "더미 분석 또는 수동 기록 결과가 여기에 표시됩니다.")}</p>
        </div>
      </Card>
      <Card title="탐지 음식">
        <pre>{JSON.stringify(record?.detected_foods ?? photoResults[0]?.detected_foods ?? [], null, 2)}</pre>
      </Card>
      <Card title="영양 구성">
        {["탄수화물", "단백질", "지방", "나트륨"].map((label, index) => (
          <div className="bar-row" key={label}>
            <span>{label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${[65, 38, 28, 52][index]}%` }} />
            </div>
            <strong>{[65, 38, 28, 52][index]}</strong>
          </div>
        ))}
      </Card>
      <Card title="추천 액션">
        <div className="button-row">
          <button onClick={() => navigate("/diets/history")}>기록 완료</button>
          <Link className="button secondary" to="/dashboard">
            추적 대시보드 이동
          </Link>
          <Link className="button secondary" to="/challenges">
            추천 챌린지
          </Link>
        </div>
      </Card>
      <Card title="분석 요약">
        <pre>{JSON.stringify(photoResults[0] ?? record, null, 2)}</pre>
      </Card>
    </div>
  );
}
