import { NavLink } from "react-router-dom";

import { isAdminConsoleRole } from "../auth/AdminRoute";
import { useAuth } from "../auth/AuthContext";

export type SidebarLink = {
  icon: string;
  label: string;
  to: string;
};

type SidebarSection = {
  title: string;
  links: SidebarLink[];
};

export const sidebarSections: SidebarSection[] = [
  {
    title: "핵심 기능",
    links: [
      { to: "/", icon: "🏠", label: "홈" },
      { to: "/health", icon: "🧭", label: "건강 분석" },
      { to: "/dashboard", icon: "📊", label: "건강 리포트" },
    ],
  },
  {
    title: "기록/관리",
    links: [
      { to: "/ocr", icon: "📄", label: "검진·복약 등록" },
      { to: "/medications", icon: "💊", label: "복약/영양제" },
      { to: "/diets", icon: "🥗", label: "식단 분석" },
      { to: "/challenges", icon: "✅", label: "챌린지" },
      { to: "/chatbot", icon: "🤖", label: "AI 건강 상담" },
    ],
  },
  {
    title: "계정/지원",
    links: [
      { to: "/mypage", icon: "👤", label: "마이페이지" },
      { to: "/settings", icon: "⚙️", label: "설정" },
      { to: "/inquiries", icon: "💬", label: "문의/FAQ" },
    ],
  },
];

export const sidebarLinks: SidebarLink[] = sidebarSections.flatMap((s) => s.links);

export default function Sidebar() {
  const { backendUser } = useAuth();
  const showAdminLink = isAdminConsoleRole(backendUser?.role);
  const getLinkClass = ({ isActive }: { isActive: boolean }) =>
    `sidebar-link${isActive ? " sidebar-link-active" : ""}`;

  return (
    <aside className="sidebar">
      {sidebarSections.map((section) => (
        <div key={section.title}>
          <p className="sidebar-section-title">{section.title}</p>
          {section.links.map((link) => (
            <NavLink aria-label={link.label} className={getLinkClass} key={link.to} title={link.label} to={link.to}>
              <span aria-hidden="true" className="sidebar-active-indicator" />
              <span className="sidebar-link-icon">{link.icon}</span>
              <span className="sidebar-link-label">{link.label}</span>
            </NavLink>
          ))}
        </div>
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
