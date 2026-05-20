import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "홈" },
  { to: "/health", label: "건강 분석" },
  { to: "/diets", label: "식단 분석" },
  { to: "/dashboard", label: "추적 대시보드" },
  { to: "/challenges", label: "챌린지" },
  { to: "/medications", label: "복약/영양제" },
  { to: "/sleep", label: "수면", disabled: true },
  { to: "/exercise", label: "운동", disabled: true },
  { to: "/water", label: "수분섭취", disabled: true },
  { to: "/family", label: "가족 관리", disabled: true },
  { to: "/inquiries", label: "1:1 문의" },
  { to: "/faq", label: "FAQ" },
  { to: "/settings", label: "설정" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-title">HealthCare</div>
      {links.map((link) => (
        link.disabled ? (
          <span className="sidebar-disabled" key={link.to}>
            {link.label}
          </span>
        ) : (
          <NavLink key={link.to} to={link.to}>
            {link.label}
          </NavLink>
        )
      ))}
    </aside>
  );
}
