import { useEffect, useState } from "react";
import { Link, NavLink } from "react-router-dom";

import { listUnreadNotifications } from "../api/notifications";
import { useAuth } from "../auth/AuthContext";
import ThemeToggle from "./ThemeToggle";

type NavbarProps = {
  isMobileMenuOpen?: boolean;
  onMobileMenuOpen?: () => void;
  showMobileMenuButton?: boolean;
};

export default function Navbar({ isMobileMenuOpen = false, onMobileMenuOpen, showMobileMenuButton = false }: NavbarProps) {
  const { isAuthenticated, logout } = useAuth();
  const [unreadCount, setUnreadCount] = useState(0);

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
        Health Ladder
      </Link>
      <div className="navbar-actions">
        <ThemeToggle />
        {isAuthenticated ? (
          <>
            <NavLink className="icon-button desktop-nav-action" to="/about">
              서비스 소개
            </NavLink>
            <NavLink className={({ isActive }) => `icon-button navbar-notification-link desktop-nav-action${isActive ? " active" : ""}`} to="/notifications" aria-label="알림">
              알림
              {unreadCount > 0 && <span className="notification-badge">{unreadCount > 99 ? "99+" : unreadCount}</span>}
            </NavLink>
            <button className="nav-logout-btn" type="button" onClick={logout}>
              로그아웃
            </button>
            {showMobileMenuButton && (
              <button
                aria-controls="mobile-navigation-drawer"
                aria-expanded={isMobileMenuOpen}
                aria-label="모바일 메뉴 열기"
                className="icon-button mobile-menu-button"
                onClick={onMobileMenuOpen}
                type="button"
              >
                <span aria-hidden="true">☰</span>
              </button>
            )}
          </>
        ) : (
          <>
            <NavLink className="icon-button desktop-nav-action" to="/">
              서비스 소개
            </NavLink>
            <NavLink className="icon-button desktop-nav-action" to="/faqs">
              FAQ
            </NavLink>
            <NavLink className="icon-button desktop-nav-action" to="/login">
              로그인
            </NavLink>
            <NavLink className="button mobile-core-action" to="/signup">
              회원가입
            </NavLink>
          </>
        )}
      </div>
    </header>
  );
}
