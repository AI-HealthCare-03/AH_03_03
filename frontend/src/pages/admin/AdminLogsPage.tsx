import { useEffect, useState } from "react";

import {
  AdminSensitiveAccessLog,
  AdminSystemErrorLog,
  getAdminSensitiveAccessLogs,
  getAdminSystemErrors,
} from "../../api/admin";
import { useAuth } from "../../auth/AuthContext";
import { formatDateTime, formatRelativeTime } from "../../utils/format";

export default function AdminLogsPage() {
  const { backendUser } = useAuth();
  const canViewSensitiveLogs = ["ADMIN", "SUPER_ADMIN"].includes(String(backendUser?.role ?? "").toUpperCase());
  const [systemErrors, setSystemErrors] = useState<AdminSystemErrorLog[]>([]);
  const [sensitiveLogs, setSensitiveLogs] = useState<AdminSensitiveAccessLog[]>([]);
  const [systemErrorTotal, setSystemErrorTotal] = useState(0);
  const [sensitiveTotal, setSensitiveTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        const [errors, accesses] = await Promise.all([
          getAdminSystemErrors(50),
          canViewSensitiveLogs ? getAdminSensitiveAccessLogs(50) : Promise.resolve({ items: [], total: 0 }),
        ]);
        if (alive) {
          setSystemErrors(errors.items);
          setSensitiveLogs(accesses.items);
          setSystemErrorTotal(errors.total);
          setSensitiveTotal(accesses.total);
          setError(null);
          setLoading(false);
        }
      } catch (error) {
        if (alive) {
          setError(error instanceof Error ? error.message : "운영 로그를 불러오지 못했습니다.");
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      alive = false;
    };
  }, [canViewSensitiveLogs]);

  return (
    <section className="admin-section">
      <div className="page-header">
        <div>
          <h1>운영 로그</h1>
          <p>장애 추적과 민감정보 접근 이력을 원문 데이터 없이 확인합니다.</p>
        </div>
      </div>

      {loading && <div className="card">운영 로그를 불러오는 중입니다...</div>}
      {error && <div className="error-message">{error}</div>}

      {!loading && !error && (
        <div className="admin-log-stack">
          <article className="card">
            <div className="section-title-row">
              <h2>시스템 오류 로그</h2>
              <span className="badge badge-muted">총 {systemErrorTotal.toLocaleString()}건</span>
            </div>
            {systemErrors.length === 0 ? (
              <div className="empty-state">최근 시스템 오류 로그가 없습니다.</div>
            ) : (
              <div className="admin-table-wrap">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>시각</th>
                      <th>상태</th>
                      <th>경로</th>
                      <th>오류 유형</th>
                      <th>요청 ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {systemErrors.map((item) => (
                      <tr key={item.id}>
                        <td title={formatDateTime(item.created_at)}>{formatRelativeTime(item.created_at)}</td>
                        <td>{item.status_code}</td>
                        <td>{item.path}</td>
                        <td>{item.error_type}</td>
                        <td>{item.request_id ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </article>

          <article className="card">
            <div className="section-title-row">
              <h2>민감정보 접근 로그</h2>
              <span className="badge badge-muted">총 {sensitiveTotal.toLocaleString()}건</span>
            </div>
            {!canViewSensitiveLogs ? (
              <div className="empty-state">민감정보 접근 로그 상세 목록은 ADMIN 이상 권한에서 확인할 수 있습니다.</div>
            ) : sensitiveLogs.length === 0 ? (
              <div className="empty-state">최근 민감정보 접근 로그가 없습니다.</div>
            ) : (
              <div className="admin-table-wrap">
                <table className="admin-table">
                  <thead>
                    <tr>
                      <th>시각</th>
                      <th>행위자</th>
                      <th>대상</th>
                      <th>리소스</th>
                      <th>행동</th>
                      <th>경로</th>
                      <th>요청 ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sensitiveLogs.map((item) => (
                      <tr key={item.id}>
                        <td title={formatDateTime(item.created_at)}>{formatRelativeTime(item.created_at)}</td>
                        <td>
                          #{item.actor_user_id} {item.actor_role ? `(${item.actor_role})` : ""}
                        </td>
                        <td>#{item.target_user_id}</td>
                        <td>{item.resource_type}</td>
                        <td>{item.action_type}</td>
                        <td>{item.path}</td>
                        <td>{item.request_id ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        </div>
      )}
    </section>
  );
}
