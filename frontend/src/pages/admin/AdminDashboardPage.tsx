import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { AdminSummary, AdminUsersSummary, getAdminSummary, getAdminUsersSummary } from "../../api/admin";

type AdminDashboardState = {
  summary: AdminSummary | null;
  users: AdminUsersSummary | null;
  error: string | null;
  loading: boolean;
};

const summaryCards = [
  ["total_users", "전체 사용자"],
  ["active_users", "활성 사용자"],
  ["today_new_users", "오늘 가입"],
  ["total_analysis_results", "분석 결과"],
  ["total_exam_reports", "검진표"],
  ["total_medications", "복약/영양제"],
] as const;

export default function AdminDashboardPage() {
  const [state, setState] = useState<AdminDashboardState>({
    summary: null,
    users: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        const [summary, users] = await Promise.all([getAdminSummary(), getAdminUsersSummary()]);
        if (alive) {
          setState({ summary, users, error: null, loading: false });
        }
      } catch (error) {
        if (alive) {
          setState({
            summary: null,
            users: null,
            error: error instanceof Error ? error.message : "관리자 요약을 불러오지 못했습니다.",
            loading: false,
          });
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
          <h1>운영 대시보드</h1>
          <p>서비스 운영 상태와 주요 지표를 민감정보 없이 요약합니다.</p>
        </div>
        <Link className="btn-secondary" to="/admin/monitoring">
          시스템 상태 보기
        </Link>
      </div>

      {state.loading && <div className="card">운영 지표를 불러오는 중입니다...</div>}
      {state.error && <div className="error-message">{state.error}</div>}

      {state.summary && (
        <>
          <div className="admin-card-grid">
            {summaryCards.map(([key, label]) => (
              <article className="card metric-card" key={key}>
                <span className="metric-label">{label}</span>
                <strong className="metric-value">{state.summary?.[key].toLocaleString()}</strong>
              </article>
            ))}
          </div>

          <div className="admin-two-column">
            <article className="card">
              <h2>오늘의 운영 로그</h2>
              <div className="admin-inline-stats">
                <span>
                  시스템 오류 <strong>{state.summary.system_error_count_today.toLocaleString()}</strong>
                </span>
                <span>
                  민감정보 접근 <strong>{state.summary.sensitive_access_count_today.toLocaleString()}</strong>
                </span>
              </div>
              <Link className="btn-secondary" to="/admin/logs">
                로그 확인
              </Link>
            </article>
            <article className="card">
              <h2>운영 환경</h2>
              <div className="admin-status-list">
                <span>
                  환경 <strong>{state.summary.environment}</strong>
                </span>
                <span>
                  이메일 서비스 <strong>{state.summary.email_service_status}</strong>
                </span>
              </div>
            </article>
          </div>
        </>
      )}

      {state.users && (
        <article className="card">
          <h2>관리자 role 요약</h2>
          <div className="admin-role-grid">
            <span>MONITOR {state.users.monitor_users}</span>
            <span>OPERATOR {state.users.operator_users}</span>
            <span>ADMIN {state.users.admin_users}</span>
            <span>SUPER_ADMIN {state.users.super_admin_users}</span>
          </div>
        </article>
      )}
    </section>
  );
}
