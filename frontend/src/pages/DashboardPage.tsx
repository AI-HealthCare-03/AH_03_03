import { useEffect, useState } from "react";

import { getDashboardSummary, getDashboardTrends } from "../api/dashboard";
import Card from "../components/Card";

type DashboardData = Record<string, unknown>;

function Bars({ items }: { items: Record<string, unknown>[] }) {
  return (
    <div className="bars">
      {items.slice(0, 8).map((item, index) => {
        const value = Number(item.value ?? item.systolic ?? 0);
        return (
          <div className="bar-row" key={`${String(item.date)}-${index}`}>
            <span>{String(item.date)}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${Math.min(value, 100)}%` }} />
            </div>
            <strong>{value || "-"}</strong>
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

  return (
    <div className="page-grid">
      <Card title="대시보드 요약">
        <pre>{JSON.stringify(summary, null, 2)}</pre>
      </Card>
      <Card title="혈당 추이">
        <Bars items={trends.glucose ?? []} />
      </Card>
      <Card title="체중 추이">
        <Bars items={trends.weight ?? []} />
      </Card>
      <Card title="챌린지 수행률">
        <Bars items={trends.challenge_completion_rate ?? []} />
      </Card>
      <Card title="식단 점수">
        <Bars items={trends.diet_score ?? []} />
      </Card>
    </div>
  );
}
