import { NavLink } from "react-router-dom";

import { isAdminConsoleRole } from "../auth/AdminRoute";
import { useAuth } from "../auth/AuthContext";

export type SidebarLink = {
  icon: string;
  label: string;
  to: string;
};

export const sidebarLinks: SidebarLink[] = [
  { to: "/", icon: "🏠", label: "홈" },
  { to: "/health", icon: "🧭", label: "건강 분석" },
  { to: "/ocr", icon: "📄", label: "검진·복약 등록" },
  { to: "/chatbot", icon: "🤖", label: "AI 건강 상담" },
  { to: "/diets", icon: "🥗", label: "식단 분석" },
  { to: "/dashboard", icon: "📊", label: "건강 리포트" },
  { to: "/challenges", icon: "✅", label: "챌린지" },
  { to: "/medications", icon: "💊", label: "복약/영양제" },
  { to: "/mypage", icon: "👤", label: "마이페이지" },
  { to: "/inquiries", icon: "💬", label: "1:1 문의" },
  { to: "/faq", icon: "?", label: "FAQ" },
  { to: "/settings", icon: "⚙️", label: "설정" },
];

export default function Sidebar() {
  const { backendUser } = useAuth();
  const showAdminLink = isAdminConsoleRole(backendUser?.role);
  const getLinkClass = ({ isActive }: { isActive: boolean }) =>
    `sidebar-link${isActive ? " sidebar-link-active" : ""}`;

  return (
    <aside className="sidebar">
      {sidebarLinks.map((link) => (
        <NavLink aria-label={link.label} className={getLinkClass} key={link.to} title={link.label} to={link.to}>
          <span aria-hidden="true" className="sidebar-active-indicator" />
          <span className="sidebar-link-icon">{link.icon}</span>
          <span className="sidebar-link-label">{link.label}</span>
        </NavLink>
      ))}
      {showAdminLink && (
        <NavLink aria-label="관리자 콘솔" className={getLinkClass} title="관리자 콘솔" to="/admin">
          <span aria-hidden="true" className="sidebar-active-indicator" />
          <span className="sidebar-link-icon">🛡️</span>
          <span className="sidebar-link-label">관리자 콘솔</span>
        </NavLink>
      )}
    </aside>
  );
}
