import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listDietRecords } from "../api/diets";
import Card from "../components/Card";
import { formatDateTime, mealTypeLabel, scoreBadgeClass } from "../utils/format";

type DietRecord = Record<string, unknown>;

const TABS = ["전체", "식단 분석", "직접 입력", "추천 식단", "최근 1개월"] as const;

function filterRecords(records: DietRecord[], tab: string): DietRecord[] {
  if (tab === "식단 분석") {
    return records.filter((r) => String(r.analysis_method ?? "").toUpperCase() !== "MANUAL");
  }
  if (tab === "직접 입력") {
    return records.filter((r) => String(r.analysis_method ?? "").toUpperCase() === "MANUAL");
  }
  if (tab === "최근 1개월") {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - 30);
    return records.filter((r) => {
      const d = new Date(String(r.meal_time ?? r.created_at ?? ""));
      return !isNaN(d.getTime()) && d >= cutoff;
    });
  }
  return records;
}

export default function DietHistoryPage() {
  const [records, setRecords] = useState<DietRecord[]>([]);
  const [activeTab, setActiveTab] = useState("전체");

  useEffect(() => {
    void listDietRecords<DietRecord[]>().then(setRecords).catch(() => setRecords([]));
  }, []);

  const filtered = filterRecords(records, activeTab);

  return (
    <Card
      title="식단 분석 결과 전체 리스트"
      actions={
        <Link className="button" to="/diets">
          식단 분석하기
        </Link>
      }
    >
      <div className="filter-tabs">
        {TABS.map((tab) => (
          <span
            className={tab === activeTab ? "filter-tab active" : "filter-tab"}
            key={tab}
            role="button"
            tabIndex={0}
            onClick={() => setActiveTab(tab)}
            onKeyDown={(e) => e.key === "Enter" && setActiveTab(tab)}
          >
            {tab}
          </span>
        ))}
      </div>
      <div className="table-list">
        {filtered.map((record) => {
          const scoreRaw = record.diet_score != null ? Number(record.diet_score) : null;
          const mealName = String(record.description ?? record.meal_name ?? "기록된 식단");
          const summary = String(
            record.diet_feedback ?? record.summary ?? "식단 사진을 기반으로 영양 균형을 확인했습니다.",
          );
          const dateStr = String(record.meal_time ?? record.created_at ?? "");

          return (
            <div className="diet-record-card" key={String(record.id)}>
              <div className="diet-record-header">
                <div className="diet-record-meta">
                  <span className="muted">{formatDateTime(dateStr)}</span>
                  <span className="badge badge-reference">{mealTypeLabel(record.meal_type)}</span>
                </div>
                {scoreRaw !== null ? (
                  <span className={`badge ${scoreBadgeClass(scoreRaw)}`}>{scoreRaw}점</span>
                ) : (
                  <span className="badge badge-reference">-</span>
                )}
              </div>
              <strong>{mealName}</strong>
              <p className="diet-record-summary">{summary}</p>
              <div className="diet-record-footer">
                <Link className="button secondary compact-button" to={`/diets/${String(record.id)}`}>
                  상세보기
                </Link>
              </div>
            </div>
          );
        })}
        {filtered.length === 0 && <p className="placeholder">아직 저장된 식단 분석 결과가 없습니다.</p>}
      </div>
    </Card>
  );
}
