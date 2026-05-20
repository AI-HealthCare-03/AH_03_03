import { Link, NavLink } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";

export default function Navbar() {
  const { backendUser, isAuthenticated, logout } = useAuth();

  return (
    <header className="topbar">
      <Link className="brand" to="/">
        <span className="brand-mark">H</span>
        HealthCare
      </Link>
      <div className="navbar-actions">
        {isAuthenticated ? (
          <>
            <Link className="icon-button" to="/notifications" aria-label="알림">
              알림
            </Link>
            <Link className="icon-button" to="/inquiries" aria-label="채팅">
              상담
            </Link>
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
