import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listDietRecords } from "../api/diets";
import Card from "../components/Card";
import { formatDateTime, mealTypeLabel } from "../utils/format";

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

function isManualRecord(record: DietRecord): boolean {
  return String(record.analysis_method ?? "").toUpperCase() === "MANUAL";
}

function summarizeFoodItems(value: unknown): string {
  if (!Array.isArray(value)) return "";
  return value
    .map((item) => {
      if (!item || typeof item !== "object") return "";
      const food = item as Record<string, unknown>;
      const name = String(food.name ?? "").trim();
      const quantity = String(food.quantity ?? "").trim();
      return name ? `${name}${quantity ? ` ${quantity}` : ""}` : "";
    })
    .filter(Boolean)
    .join(", ");
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
      <div className="state-box">분석 결과는 건강관리 참고용이며, 실제 식사량과 음식 선택에 따라 달라질 수 있습니다.</div>
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
          const isManual = isManualRecord(record);
          const foodSummary = summarizeFoodItems(record.detected_foods);
          const mealName = String(record.description ?? record.meal_name ?? (foodSummary || "기록된 식단"));
          const summary = isManual
            ? String(record.memo ?? (foodSummary || "직접 입력한 식단 기록입니다."))
            : String(record.diet_feedback ?? record.summary ?? "식단 분석 결과를 확인해보세요.");
          const dateStr = String(record.meal_time ?? record.created_at ?? "");
          return (
            <div className="diet-record-card" key={String(record.id)}>
              <div className="diet-record-header">
                <div className="diet-record-meta">
                  <span className="muted">{formatDateTime(dateStr)}</span>
                  <span className="badge badge-reference">{mealTypeLabel(record.meal_type)}</span>
                  {isManual && <span className="badge badge-reference">직접 기록</span>}
                </div>
                <Link className="button secondary compact-button" to={`/diets/${String(record.id)}`}>
                  상세보기
                </Link>
              </div>
              <strong>{mealName}</strong>
              <p className="diet-record-summary">{summary}</p>
            </div>
          );
        })}
        {filtered.length === 0 && <p className="placeholder">아직 저장된 식단 분석 결과가 없습니다.</p>}
      </div>
    </Card>
  );
}
