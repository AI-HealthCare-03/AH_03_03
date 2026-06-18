import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { listDietRecords } from "../api/diets";
import Card from "../components/Card";
import { formatDateTime, mealTypeLabel } from "../utils/format";

type DietRecord = Record<string, unknown>;

const TABS = ["전체", "식단 분석", "최근 1개월"] as const;
const PAGE_SIZE = 6;

function getRecordDate(record: DietRecord): Date | null {
  const date = new Date(String(record.meal_time ?? record.created_at ?? ""));
  return Number.isNaN(date.getTime()) ? null : date;
}

function filterRecords(records: DietRecord[], tab: string, startDate: string, endDate: string): DietRecord[] {
  let nextRecords = records;
  if (tab === "식단 분석") {
    nextRecords = nextRecords.filter((r) => String(r.analysis_method ?? "").toUpperCase() !== "MANUAL");
  }
  if (tab === "최근 1개월") {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - 30);
    nextRecords = nextRecords.filter((r) => {
      const d = getRecordDate(r);
      return d !== null && d >= cutoff;
    });
  }
  if (startDate) {
    const start = new Date(`${startDate}T00:00:00`);
    nextRecords = nextRecords.filter((r) => {
      const d = getRecordDate(r);
      return d !== null && d >= start;
    });
  }
  if (endDate) {
    const end = new Date(`${endDate}T23:59:59`);
    nextRecords = nextRecords.filter((r) => {
      const d = getRecordDate(r);
      return d !== null && d <= end;
    });
  }
  return nextRecords;
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
      const name = String(
        food.matched_food_name ??
          food.display_name ??
          food.name ??
          food.food_name ??
          food.original_name ??
          food.raw_food_name ??
          food.vision_food_name ??
          "",
      ).trim();
      const quantity = String(food.quantity ?? "").trim();
      return name ? `${name}${quantity ? ` ${quantity}` : ""}` : "";
    })
    .filter(Boolean)
    .join(", ");
}

function getDietRecordTitle(record: DietRecord, foodSummary: string): string {
  if (foodSummary) {
    const names = foodSummary.split(",").map((name) => name.trim()).filter(Boolean);
    return names.length > 2 ? `${names.slice(0, 2).join(", ")} 외 ${names.length - 2}개` : names.join(", ");
  }
  const fallback = String(record.description ?? record.meal_name ?? "").trim();
  return fallback && fallback !== "사진으로 선택한 식단" ? fallback : "분석한 식단";
}

export default function DietHistoryPage() {
  const [records, setRecords] = useState<DietRecord[]>([]);
  const [activeTab, setActiveTab] = useState("전체");
  const [searchParams, setSearchParams] = useSearchParams();
  const [startDate, setStartDate] = useState(searchParams.get("start") ?? "");
  const [endDate, setEndDate] = useState(searchParams.get("end") ?? "");

  const currentPage = Number(searchParams.get("page") ?? "1");

  useEffect(() => {
    void listDietRecords<DietRecord[]>().then(setRecords).catch(() => setRecords([]));
  }, []);

  const filtered = filterRecords(records, activeTab, startDate, endDate);
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const safePage = Math.min(currentPage, Math.max(totalPages, 1));
  const paginated = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    setSearchParams({ page: "1", ...(startDate ? { start: startDate } : {}), ...(endDate ? { end: endDate } : {}) });
  };

  const handlePageChange = (page: number) => {
    setSearchParams({ page: String(page), ...(startDate ? { start: startDate } : {}), ...(endDate ? { end: endDate } : {}) });
  };

  const handleDateFilterChange = (nextStartDate: string, nextEndDate: string) => {
    setStartDate(nextStartDate);
    setEndDate(nextEndDate);
    setSearchParams({ page: "1", ...(nextStartDate ? { start: nextStartDate } : {}), ...(nextEndDate ? { end: nextEndDate } : {}) });
  };

  return (
    <Card
      title="식단 분석 결과 전체 리스트"
      actions={
        <Link className="button" to="/diets">
          식단 분석하기
        </Link>
      }
    >
      <div className="state-box">
        음식 후보와 식단 분석 결과는 건강관리 참고용이며, 실제 식사량과 음식 선택에 따라 달라질 수 있습니다.
      </div>
      <div className="filter-tabs">
        {TABS.map((tab) => (
          <span
            className={tab === activeTab ? "filter-tab active" : "filter-tab"}
            key={tab}
            role="button"
            tabIndex={0}
            onClick={() => handleTabChange(tab)}
            onKeyDown={(e) => e.key === "Enter" && handleTabChange(tab)}
          >
            {tab}
          </span>
        ))}
      </div>
      <div className="history-filter-row">
        <label>
          시작일
          <input
            onChange={(event) => handleDateFilterChange(event.target.value, endDate)}
            type="date"
            value={startDate}
          />
        </label>
        <label>
          종료일
          <input
            onChange={(event) => handleDateFilterChange(startDate, event.target.value)}
            type="date"
            value={endDate}
          />
        </label>
        {(startDate || endDate) && (
          <button className="secondary compact-button" onClick={() => handleDateFilterChange("", "")} type="button">
            날짜 초기화
          </button>
        )}
      </div>
      <div className="table-list">
        {paginated.map((record) => {
          const isManual = isManualRecord(record);
          const foodSummary = summarizeFoodItems(record.detected_foods);
          const mealName = getDietRecordTitle(record, foodSummary);
          const summary = isManual
            ? String(record.memo ?? (foodSummary || "직접 입력한 식단 기록입니다."))
            : String(record.diet_feedback ?? record.summary ?? "음식 후보와 영양성분을 상세보기에서 확인할 수 있습니다.");
          const dateStr = String(record.meal_time ?? record.created_at ?? "");
          return (
            <div className="diet-record-card" key={String(record.id)}>
              <div className="diet-record-header">
                <div className="diet-record-meta">
                  <span className="muted">{formatDateTime(dateStr)}</span>
                  <span className="badge badge-reference">{mealTypeLabel(record.meal_type)}</span>
                  {isManual && <span className="badge badge-reference">직접 기록</span>}
                </div>
                <span className="badge badge-reference">{isManual ? "직접 기록" : "분석 기록"}</span>
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
      {totalPages > 1 && (
        <div style={{ display: "flex", justifyContent: "center", gap: "8px", marginTop: "16px" }}>
          <button
            className="button secondary compact-button"
            disabled={safePage === 1}
            onClick={() => handlePageChange(safePage - 1)}
            type="button"
          >
            이전
          </button>
          {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
            <button
              className={`button compact-button ${page === safePage ? "" : "secondary"}`}
              key={page}
              onClick={() => handlePageChange(page)}
              type="button"
            >
              {page}
            </button>
          ))}
          <button
            className="button secondary compact-button"
            disabled={safePage === totalPages}
            onClick={() => handlePageChange(safePage + 1)}
            type="button"
          >
            다음
          </button>
        </div>
      )}
    </Card>
  );
}
