import { Link, NavLink } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";

export default function Navbar() {
  const { backendUser, isAuthenticated, logout } = useAuth();

  return (
    <header className="navbar">
      <Link className="brand" to="/">
        AI Health
      </Link>
      <div className="navbar-actions">
        {isAuthenticated ? (
          <>
            <span>{backendUser?.nickname ?? backendUser?.name ?? backendUser?.email}</span>
            <button type="button" onClick={logout}>
              로그아웃
            </button>
          </>
        ) : (
          <>
            <NavLink to="/login">로그인</NavLink>
            <NavLink to="/signup">회원가입</NavLink>
          </>
        )}
      </div>
    </header>
  );
}
