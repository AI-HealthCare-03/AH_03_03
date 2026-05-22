import { NavLink } from "react-router-dom";

import { isAdminConsoleRole } from "../auth/AdminRoute";
import { useAuth } from "../auth/AuthContext";

const links = [
  { to: "/", icon: "🏠", label: "홈" },
  { to: "/health", icon: "🧭", label: "건강 분석" },
  { to: "/ocr", icon: "📄", label: "OCR 입력" },
  { to: "/chatbot", icon: "🤖", label: "AI 건강 상담" },
  { to: "/diets", icon: "🥗", label: "식단 분석" },
  { to: "/dashboard", icon: "📊", label: "추적 대시보드" },
  { to: "/challenges", icon: "✅", label: "챌린지" },
  { to: "/medications", icon: "💊", label: "복약/영양제" },
  { to: "/family", icon: "👥", label: "가족 관리" },
  { to: "/inquiries", icon: "💬", label: "1:1 문의" },
  { to: "/faq", icon: "?", label: "FAQ" },
  { to: "/settings", icon: "⚙️", label: "설정" },
];

export default function Sidebar() {
  const { backendUser } = useAuth();
  const showAdminLink = isAdminConsoleRole(backendUser?.role);

  return (
    <aside className="sidebar">
      <div className="sidebar-title">
        <span className="sidebar-title-icon">H</span>
        <span className="sidebar-label">HealthCare</span>
      </div>
      {links.map((link) => (
        <NavLink key={link.to} to={link.to}>
          <span className="sidebar-icon">{link.icon}</span>
          <span className="sidebar-label">{link.label}</span>
        </NavLink>
      ))}
      {showAdminLink && (
        <NavLink to="/admin">
          <span className="sidebar-icon">🛡️</span>
          <span className="sidebar-label">관리자 콘솔</span>
        </NavLink>
      )}
    </aside>
  );
}
