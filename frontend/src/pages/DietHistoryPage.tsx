import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listDietRecords } from "../api/diets";
import Card from "../components/Card";

type DietRecord = Record<string, unknown>;

export default function DietHistoryPage() {
  const [records, setRecords] = useState<DietRecord[]>([]);

  useEffect(() => {
    void listDietRecords<DietRecord[]>().then(setRecords).catch(() => setRecords([]));
  }, []);

  return (
    <Card
      title="식단 추천 결과 전체 리스트"
      actions={
        <Link className="button" to="/diets">
          식단 분석하기
        </Link>
      }
    >
      <div className="filter-tabs">
        {["전체", "식단 업로드", "추천 식단", "직접 입력", "최근 1개월"].map((tab, index) => (
          <span className={index === 0 ? "filter-tab active" : "filter-tab"} key={tab}>
            {tab}
          </span>
        ))}
      </div>
      <div className="table-list">
        {records.map((record) => (
          <Link className="table-row" key={String(record.id)} to={`/diets/${String(record.id)}`}>
            <span>{String(record.meal_time ?? record.created_at ?? "")}</span>
            <strong>{String(record.diet_score ?? "-")}점</strong>
            <span>{String(record.description ?? "분석 요약")}</span>
          </Link>
        ))}
        {records.length === 0 && <p className="placeholder">아직 저장된 식단 분석 결과가 없습니다.</p>}
      </div>
    </Card>
  );
}
