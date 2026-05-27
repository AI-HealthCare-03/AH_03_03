import { Link, Navigate, Outlet } from "react-router-dom";

import { useAuth } from "./AuthContext";

const ADMIN_ROLES = new Set(["MONITOR", "OPERATOR", "ADMIN", "SUPER_ADMIN"]);
const OPERATOR_ROLES = new Set(["OPERATOR", "ADMIN", "SUPER_ADMIN"]);

export function isAdminConsoleRole(role?: string | null): boolean {
  return ADMIN_ROLES.has(String(role ?? "").toUpperCase());
}

export function isOperatorRole(role?: string | null): boolean {
  return OPERATOR_ROLES.has(String(role ?? "").toUpperCase());
}

export default function AdminRoute() {
  const { backendUser, isAuthenticated, loading } = useAuth();

  if (loading) {
    return <div className="page-loading">관리자 권한을 확인하고 있습니다...</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (!isAdminConsoleRole(backendUser?.role)) {
    return (
      <main className="admin-access-denied">
        <section className="card">
          <span className="badge badge-muted">403</span>
          <h1>관리자 권한이 필요합니다</h1>
          <p>운영 콘솔은 승인된 관리자만 접근할 수 있습니다.</p>
          <Link className="btn-primary" to="/">
            일반 앱으로 돌아가기
          </Link>
        </section>
      </main>
    );
  }

  return <Outlet />;
}
