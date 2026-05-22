import { NavLink, Outlet } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import ThemeToggle from "./ThemeToggle";

const adminLinks = [
  { to: "/admin", label: "운영 대시보드", icon: "📊" },
  { to: "/admin/monitoring", label: "모니터링", icon: "🩺" },
  { to: "/admin/logs", label: "운영 로그", icon: "📋" },
];

export default function AdminLayout() {
  const { backendUser, logout } = useAuth();

  return (
    <div className="admin-shell">
      <aside className="admin-sidebar">
        <NavLink className="admin-brand" to="/admin">
          <span className="brand-mark">H</span>
          Admin
        </NavLink>
        <nav className="admin-nav">
          {adminLinks.map((link) => (
            <NavLink key={link.to} to={link.to} end={link.to === "/admin"}>
              <span>{link.icon}</span>
              {link.label}
            </NavLink>
          ))}
        </nav>
        <NavLink className="admin-return-link" to="/">
          일반 앱으로 돌아가기
        </NavLink>
      </aside>
      <div className="admin-content-shell">
        <header className="admin-topbar">
          <div>
            <strong>관리자 콘솔</strong>
            <span>{backendUser?.name ?? backendUser?.email} · {backendUser?.role}</span>
          </div>
          <div className="admin-topbar-actions">
            <ThemeToggle />
            <button type="button" className="btn-secondary" onClick={logout}>
              로그아웃
            </button>
          </div>
        </header>
        <main className="admin-page">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
