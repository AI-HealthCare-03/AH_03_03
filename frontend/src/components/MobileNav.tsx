import { useEffect } from "react";
import { NavLink, useLocation } from "react-router-dom";

import { isAdminConsoleRole } from "../auth/AdminRoute";
import { useAuth } from "../auth/AuthContext";
import { sidebarLinks } from "./Sidebar";

type MobileNavProps = {
  isOpen: boolean;
  onClose: () => void;
};

const bottomLinks = [
  { to: "/", icon: "🏠", label: "홈" },
  { to: "/health", icon: "🧭", label: "분석" },
  { to: "/challenges", icon: "✅", label: "챌린지" },
  { to: "/diets", icon: "🥗", label: "식단" },
  { to: "/ocr/exam", icon: "📄", label: "검진표" },
];

const linkByPath = new Map(sidebarLinks.map((link) => [link.to, link]));
const pickLinks = (paths: string[]) => paths.flatMap((path) => {
  const link = linkByPath.get(path);
  return link ? [link] : [];
});

const drawerSections = [
  {
    links: pickLinks(["/", "/health", "/ocr/exam", "/diets", "/challenges", "/chatbot"]),
    title: "주요 기능",
  },
  {
    links: pickLinks(["/dashboard", "/medications", "/family"]),
    title: "기록/관리",
  },
  {
    links: [
      { to: "/notifications", icon: "🔔", label: "알림" },
      { to: "/mypage", icon: "👤", label: "마이페이지" },
      ...pickLinks(["/inquiries", "/faq", "/settings"]),
    ],
    title: "계정/지원",
  },
];

export default function MobileNav({ isOpen, onClose }: MobileNavProps) {
  const { backendUser, logout } = useAuth();
  const location = useLocation();
  const showAdminLink = isAdminConsoleRole(backendUser?.role);

  useEffect(() => {
    onClose();
  }, [location.pathname, onClose]);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", closeOnEscape);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [isOpen, onClose]);

  const handleLogout = () => {
    onClose();
    void logout();
  };

  return (
    <>
      <nav aria-label="모바일 주요 메뉴" className="mobile-bottom-nav">
        {bottomLinks.map((link) => (
          <NavLink className="mobile-bottom-nav-item" key={link.to} to={link.to}>
            <span aria-hidden="true">{link.icon}</span>
            <span>{link.label}</span>
          </NavLink>
        ))}
      </nav>

      {isOpen && <button aria-label="모바일 메뉴 닫기" className="mobile-nav-backdrop" onClick={onClose} type="button" />}

      <aside
        aria-hidden={!isOpen}
        className={`mobile-nav-drawer${isOpen ? " mobile-nav-drawer-open" : ""}`}
        id="mobile-navigation-drawer"
      >
        <div className="mobile-nav-drawer-header">
          <div>
            <strong>전체 메뉴</strong>
            <span>{backendUser?.nickname ?? backendUser?.name ?? backendUser?.email}</span>
          </div>
          <button aria-label="모바일 메뉴 닫기" className="icon-button" onClick={onClose} type="button">
            <span aria-hidden="true">×</span>
          </button>
        </div>

        <div className="mobile-nav-drawer-links">
          {drawerSections.map((section) => (
            <section className="mobile-nav-section" key={section.title}>
              <h2>{section.title}</h2>
              <div className="mobile-nav-section-links">
                {section.links.map((link) => (
                  <NavLink className="mobile-drawer-link" key={link.to} to={link.to}>
                    <span aria-hidden="true">{link.icon}</span>
                    <span>{link.label}</span>
                  </NavLink>
                ))}
              </div>
            </section>
          ))}
          {showAdminLink && (
            <NavLink className="mobile-drawer-link" to="/admin">
              <span aria-hidden="true">🛡️</span>
              <span>관리자 콘솔</span>
            </NavLink>
          )}
        </div>

        <button className="mobile-nav-logout" onClick={handleLogout} type="button">
          로그아웃
        </button>
      </aside>
    </>
  );
}
