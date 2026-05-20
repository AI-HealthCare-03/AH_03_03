import { useEffect, useState } from "react";

import { getDashboardSummary, getDashboardTrends } from "../api/dashboard";
import Card from "../components/Card";

type DashboardData = Record<string, unknown>;
type HealthRecord = Record<string, unknown>;

function Bars({ items }: { items: Record<string, unknown>[] }) {
  return (
    <div className="bars">
      {items.slice(0, 8).map((item, index) => {
        const value = Number(item.value ?? item.systolic ?? 0);
        const label =
          item.systolic || item.diastolic
            ? `${String(item.systolic ?? "-")}/${String(item.diastolic ?? "-")}`
            : value || "-";
        return (
          <div className="bar-row" key={`${String(item.date)}-${index}`}>
            <span>{String(item.date)}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${Math.min(value, 100)}%` }} />
            </div>
            <strong>{label}</strong>
          </div>
        );
      })}
    </div>
  );
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardData>({});
  const [trends, setTrends] = useState<Record<string, Record<string, unknown>[]>>({});

  useEffect(() => {
    const load = async () => {
      setSummary(await getDashboardSummary<DashboardData>());
      setTrends(await getDashboardTrends<Record<string, Record<string, unknown>[]>>("week"));
    };
    void load().catch(() => undefined);
  }, []);

  const latest = (summary.latest_health_record ?? {}) as HealthRecord;
  const metrics = [
    ["혈당", latest.fasting_glucose ?? "-"],
    ["혈압", `${String(latest.systolic_bp ?? "-")}/${String(latest.diastolic_bp ?? "-")}`],
    ["체중", latest.weight_kg ?? "-"],
    ["챌린지 수행률", "72%"],
    ["식단 점수", "84"],
    ["복약/영양제 수행률", "80%"],
    ["수면 시간", latest.sleep_hours ?? "-"],
  ];

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <h1>추적 대시보드</h1>
          <p>혈당, 혈압, 체중, 식단, 챌린지 변화를 한 번에 확인합니다.</p>
        </div>
      </div>
      <div className="metric-grid">
        {metrics.map(([label, value]) => (
          <div className="metric-card" key={String(label)}>
            <span>{String(label)}</span>
            <strong>{String(value)}</strong>
          </div>
        ))}
      </div>
      <div className="dashboard-grid">
        <Card title="분석 항목">
          <div className="card-list">
            {["혈당/혈압 변화", "체중/BMI 변화", "수면 변화", "운동 수행률", "식단 점수 변화", "복약/영양제 수행률", "수분 섭취 수행률"].map(
              (item, index) => (
                <span className={index === 0 ? "filter-tab active" : "filter-tab"} key={item}>
                  {item}
                </span>
              ),
            )}
          </div>
        </Card>
        <Card title="혈당/혈압 변화">
          <div className="mock-chart" />
        </Card>
      </div>
      <div className="page-grid">
        <Card title="혈당 추이">
          <Bars items={trends.glucose ?? []} />
        </Card>
        <Card title="혈압 추이">
          <Bars items={trends.blood_pressure ?? []} />
        </Card>
        <Card title="AI 코멘트">
          <p>최근 입력된 건강 지표를 기준으로 혈당, 혈압, 체중 변화를 함께 확인해보세요.</p>
        </Card>
        <Card title="건강 팁">
          <p>식후 10분 걷기와 저녁 나트륨 줄이기는 혈당과 혈압 관리에 도움이 될 수 있습니다.</p>
        </Card>
        <Card title="식단 추천 결과 요약">
          <Bars items={trends.diet_score ?? []} />
        </Card>
        <Card title="추천 챌린지">
          <p>식후 산책, 물 마시기, 혈압 기록하기 같은 기본 챌린지를 먼저 시작해보세요.</p>
        </Card>
      </div>
    </div>
  );
}
