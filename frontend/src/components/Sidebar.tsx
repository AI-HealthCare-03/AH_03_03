import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "메인" },
  { to: "/dashboard", label: "대시보드" },
  { to: "/health", label: "건강정보" },
  { to: "/analysis", label: "건강분석" },
  { to: "/challenges", label: "챌린지" },
  { to: "/diets", label: "식단" },
  { to: "/medications", label: "복약" },
  { to: "/notifications", label: "알림" },
  { to: "/faqs", label: "FAQ" },
  { to: "/settings", label: "설정" },
  { to: "/mypage", label: "마이페이지" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      {links.map((link) => (
        <NavLink key={link.to} to={link.to}>
          {link.label}
        </NavLink>
      ))}
    </aside>
  );
}
