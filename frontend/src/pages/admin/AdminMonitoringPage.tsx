import { useEffect, useState } from "react";

import { AdminSystemHealth, getAdminSystemHealth } from "../../api/admin";

function statusLabel(value: string): string {
  const normalized = value.toLowerCase();
  if (normalized === "ok" || normalized === "configured") return "정상";
  if (normalized === "not_configured" || normalized === "disabled") return "비활성";
  if (normalized === "degraded" || normalized === "misconfigured") return "주의";
  return value;
}

export default function AdminMonitoringPage() {
  const [health, setHealth] = useState<AdminSystemHealth | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        const data = await getAdminSystemHealth();
        if (alive) {
          setHealth(data);
          setError(null);
          setLoading(false);
        }
      } catch (error) {
        if (alive) {
          setError(error instanceof Error ? error.message : "시스템 상태를 불러오지 못했습니다.");
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="admin-section">
      <div className="page-header">
        <div>
          <h1>모니터링</h1>
          <p>API, 데이터베이스, Redis, 이메일 서비스 상태를 확인합니다.</p>
        </div>
      </div>

      {loading && <div className="card">시스템 상태를 확인하는 중입니다...</div>}
      {error && <div className="error-message">{error}</div>}

      {health && (
        <>
          <article className="card admin-health-overview">
            <span className={`badge status-${health.status}`}>{statusLabel(health.status)}</span>
            <div>
              <h2>{health.service}</h2>
              <p>환경: {health.environment}</p>
            </div>
          </article>

          <div className="admin-card-grid">
            {Object.entries(health.checks).map(([key, value]) => (
              <article className="card metric-card" key={key}>
                <span className="metric-label">{key}</span>
                <strong className={`admin-check-value status-${value}`}>{statusLabel(value)}</strong>
                {health.details?.[key] && <p className="muted-text">{health.details[key]}</p>}
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
