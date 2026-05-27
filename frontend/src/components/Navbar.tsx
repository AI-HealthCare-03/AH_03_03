import { useEffect, useState } from "react";
import { Link, NavLink } from "react-router-dom";

import { listUnreadNotifications } from "../api/notifications";
import { isAdminConsoleRole } from "../auth/AdminRoute";
import { useAuth } from "../auth/AuthContext";
import ThemeToggle from "./ThemeToggle";

export default function Navbar() {
  const { backendUser, isAuthenticated, logout } = useAuth();
  const [unreadCount, setUnreadCount] = useState(0);
  const showAdminLink = isAdminConsoleRole(backendUser?.role);

  useEffect(() => {
    if (!isAuthenticated) {
      setUnreadCount(0);
      return undefined;
    }

    const loadUnreadCount = async () => {
      try {
        const unreadItems = await listUnreadNotifications<unknown[]>();
        setUnreadCount(Array.isArray(unreadItems) ? unreadItems.length : 0);
      } catch {
        setUnreadCount(0);
      }
    };

    void loadUnreadCount();
    const intervalId = window.setInterval(() => {
      void loadUnreadCount();
    }, 60_000);

    return () => window.clearInterval(intervalId);
  }, [isAuthenticated]);

  return (
    <header className="topbar">
      <Link className="brand" to="/">
        <span className="brand-mark">H</span>
        HealthCare
      </Link>
      <div className="navbar-actions">
        <ThemeToggle />
        {isAuthenticated ? (
          <>
            <Link className="icon-button" to="/notifications" aria-label="알림">
              알림
              {unreadCount > 0 && <span className="notification-badge">{unreadCount > 99 ? "99+" : unreadCount}</span>}
            </Link>
            <Link className="icon-button" to="/chatbot" aria-label="AI 건강 상담">
              상담
            </Link>
            {showAdminLink && (
              <Link className="icon-button" to="/admin" aria-label="관리자 콘솔">
                관리자
              </Link>
            )}
            <Link className="user-chip" to="/mypage">
              <span className="avatar">{(backendUser?.nickname ?? backendUser?.name ?? "U").slice(0, 1)}</span>
              <span>{backendUser?.nickname ?? backendUser?.name ?? backendUser?.email}</span>
            </Link>
            <button type="button" onClick={logout}>
              로그아웃
            </button>
          </>
        ) : (
          <>
            <NavLink to="/">서비스 소개</NavLink>
            <NavLink to="/faqs">FAQ</NavLink>
            <NavLink to="/login">로그인</NavLink>
            <NavLink className="button" to="/signup">
              회원가입
            </NavLink>
          </>
        )}
      </div>
    </header>
  );
}
