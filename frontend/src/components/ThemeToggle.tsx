import { useEffect, useState } from "react";
import { MoonStar, Sun } from "lucide-react";

type Theme = "light" | "dark";

const THEME_STORAGE_KEY = "ai_health_theme";

function getInitialTheme(): Theme {
  if (typeof window === "undefined") {
    return "light";
  }
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  return storedTheme === "dark" ? "dark" : "light";
}

function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  window.localStorage.setItem(THEME_STORAGE_KEY, theme);
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const nextTheme = theme === "dark" ? "light" : "dark";

  return (
    <button
      aria-label={theme === "dark" ? "라이트모드로 전환" : "다크모드로 전환"}
      className="theme-toggle"
      onClick={() => setTheme(nextTheme)}
      type="button"
    >
      <span aria-hidden="true">
  {theme === "dark" ? <Sun size={16} /> : <MoonStar size={16} />}
</span>
      <span className="theme-toggle-label">{theme === "dark" ? "라이트" : "다크"}</span>
    </button>
  );
}
